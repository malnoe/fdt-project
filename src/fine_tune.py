from __future__ import annotations

import os
import random
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

from tqdm import tqdm

from config import Config
from llm_classifier import LLMClassifier

from transformers import (
    AutoTokenizer,
    AutoModel,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup,
)

ASPECTS = ["Prix", "Service", "Cuisine"]
LABELS = ["Négative", "Neutre", "Positive", "NE"]  # ordre fixe

label2id = {l: i for i, l in enumerate(LABELS)}
id2label = {i: l for l, i in label2id.items()}

# Optionnel : normalisation si tes données ont des variantes (sinon laisse tel quel)
NORMALIZE_LABEL = {
    "Negatif": "Négative",
    "Négatif": "Négative",
    "Negative": "Négative",
    "Neutral": "Neutre",
    "Positif": "Positive",
    "Positive#NE": "Positive",
    "Non exprimé": "NE",
    "Non-exprimé": "NE",
    "N/E": "NE",
}

def _norm_label(x: str) -> str:
    x = (x or "").strip()
    return NORMALIZE_LABEL.get(x, x)


def _get_cfg(cfg: Config, name: str, default: Any) -> Any:
    return getattr(cfg, name, default)


def _resolve_device(device: int) -> torch.device:
    if device >= 0 and torch.cuda.is_available():
        return torch.device(f"cuda:{device}")
    return torch.device("cpu")


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =========================
# PLMFT model: encoder + 3 heads
# =========================
class AspectSentimentModel(nn.Module):
    def __init__(self, base_model_name: str, dropout: float = 0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(base_model_name)
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.heads = nn.ModuleDict({asp: nn.Linear(hidden, len(LABELS)) for asp in ASPECTS})

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)

        # pooling : pooler_output si dispo, sinon mean pooling masqué
        pooled = getattr(out, "pooler_output", None)
        if pooled is None:
            last_hidden = out.last_hidden_state  # (B, T, H)
            mask = attention_mask.unsqueeze(-1)  # (B, T, 1)
            pooled = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

        pooled = self.dropout(pooled)
        logits = {asp: self.heads[asp](pooled) for asp in ASPECTS}  # each (B, 4)
        return logits


# =========================
# PLMFT dataset
# =========================
class ReviewsDataset(Dataset):
    """
    attend une liste de dicts:
      {"Avis": "...", "Prix": "...", "Service": "...", "Cuisine": "..."}
    """

    def __init__(self, data: list[dict], tokenizer, max_length: int):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, i: int) -> dict[str, Any]:
        ex = self.data[i]
        text = ex["Avis"]

        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors=None,
        )

        labels = {}
        for asp in ASPECTS:
            lab = _norm_label(ex[asp])
            if lab not in label2id:
                raise ValueError(
                    f"Label inconnu '{ex[asp]}' (normalisé='{lab}') pour aspect '{asp}'. "
                    f"Labels attendus: {LABELS}"
                )
            labels[asp] = label2id[lab]

        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "labels": labels,
        }


def _plmft_collate(batch: list[dict], data_collator: DataCollatorWithPadding) -> dict[str, Any]:
    # padding dynamique via HF
    features = [{"input_ids": x["input_ids"], "attention_mask": x["attention_mask"]} for x in batch]
    padded = data_collator(features)  # tensors

    # labels -> tensors (B,) - use tensor constructor directly for efficiency
    labels = {
        asp: torch.tensor([x["labels"][asp] for x in batch], dtype=torch.long) 
        for asp in ASPECTS
    }
    padded["labels"] = labels
    return padded


def _evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    ce = nn.CrossEntropyLoss(reduction='sum')
    model.eval()

    total_loss = 0.0
    total = 0

    correct = {asp: 0 for asp in ASPECTS}

    with torch.no_grad():
        for batch in tqdm(loader, desc="Validating", leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            batch_size = input_ids.size(0)
            labels = {asp: batch["labels"][asp].to(device) for asp in ASPECTS}

            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = sum(ce(logits[asp], labels[asp]) for asp in ASPECTS) / len(ASPECTS)

            total_loss += loss.item() * batch_size
            total += batch_size

            for asp in ASPECTS:
                preds = torch.argmax(logits[asp], dim=-1)
                correct[asp] += (preds == labels[asp]).sum().item()

    divisor = max(1, total)
    out = {
        "val_loss": total_loss / divisor,
        "acc_mean": np.mean([correct[asp] / divisor for asp in ASPECTS]),
    }
    for asp in ASPECTS:
        out[f"acc_{asp.lower()}"] = correct[asp] / divisor

    model.train()
    return out

class PLMFTClassifier:
    def __init__(self, cfg: Config):
        self.cfg = cfg

        self.tokenizer: Optional[AutoTokenizer] = None
        self.plm_model: Optional[AspectSentimentModel] = None

        # Réglages / chemins
        self.model_name: str = _get_cfg(cfg, "model_name", "cmarkea/distilcamembert-base")
        self.output_dir: str = _get_cfg(cfg, "output_dir", "./plmft_ckpt")
        os.makedirs(self.output_dir, exist_ok=True)
        self.ckpt_path: str = os.path.join(self.output_dir, "best.pt")
        self._init_plmft()

    def _init_plmft(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.plm_model = AspectSentimentModel(
            base_model_name=self.model_name,
            dropout=float(_get_cfg(self.cfg, "dropout", 0.1)),
        )


    def train(self, train_data: list[dict], val_data: list[dict], device: int) -> None:
        """
        :param train_data:
        :param val_data:
        :param device: un nombre qui identifie le numéro de la gpu sur laquelle le traitement doit se faire
        -1 veut dire que le device est la cpu, et un nombre entier >= 0 indiquera le numéro de la gpu
        :return:
        """
        if self.plm_model is None or self.tokenizer is None:
            self._init_plmft()

        assert self.plm_model is not None
        assert self.tokenizer is not None

        torch_device = _resolve_device(device)
        self.plm_model.to(torch_device)
        self.plm_model.train()

        seed = int(_get_cfg(self.cfg, "seed", 42))
        _set_seed(seed)

        max_len = int(_get_cfg(self.cfg, "max_length", 256))
        train_bs = int(_get_cfg(self.cfg, "batch_size", 16))
        eval_bs = int(_get_cfg(self.cfg, "eval_batch_size", 32))
        lr = float(_get_cfg(self.cfg, "lr", 2e-5))
        epochs = int(_get_cfg(self.cfg, "epochs", 5))
        weight_decay = float(_get_cfg(self.cfg, "weight_decay", 0.01))
        warmup_ratio = float(_get_cfg(self.cfg, "warmup_ratio", 0.1))

        train_ds = ReviewsDataset(train_data, self.tokenizer, max_len)
        val_ds = ReviewsDataset(val_data, self.tokenizer, max_len)

        collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

        train_loader = DataLoader(
            train_ds,
            batch_size=train_bs,
            shuffle=True,
            collate_fn=lambda b: _plmft_collate(b, collator),
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=eval_bs,
            shuffle=False,
            collate_fn=lambda b: _plmft_collate(b, collator),
        )

        # Optim + scheduler
        opt = AdamW(self.plm_model.parameters(), lr=lr, weight_decay=weight_decay)
        total_steps = epochs * max(1, len(train_loader))
        warmup_steps = int(warmup_ratio * total_steps)

        sched = get_linear_schedule_with_warmup(
            opt, num_warmup_steps=warmup_steps, num_training_steps=total_steps
        )

        ce = nn.CrossEntropyLoss(reduction='sum')
        best_val = float("inf")

        for ep in range(1, epochs + 1):
            running_loss = 0.0
            seen = 0

            for batch in tqdm(train_loader, desc=f"PLMFT train epoch {ep}/{epochs}"):
                opt.zero_grad(set_to_none=True)

                input_ids = batch["input_ids"].to(torch_device)
                attention_mask = batch["attention_mask"].to(torch_device)
                batch_size = input_ids.size(0)
                labels = {asp: batch["labels"][asp].to(torch_device) for asp in ASPECTS}

                logits = self.plm_model(input_ids=input_ids, attention_mask=attention_mask)
                loss = sum(ce(logits[asp], labels[asp]) for asp in ASPECTS) / len(ASPECTS)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.plm_model.parameters(), 1.0)
                opt.step()
                sched.step()

                running_loss += loss.item() * batch_size
                seen += batch_size

            train_loss = running_loss / max(1, seen)
            metrics = _evaluate(self.plm_model, val_loader, torch_device)

            print(
                f"[epoch {ep}/{epochs}] "
                f"train_loss={train_loss:.4f} "
                f"val_loss={metrics['val_loss']:.4f} "
                f"acc_mean={metrics['acc_mean']:.4f} "
                f"acc_prix={metrics['acc_prix']:.4f} "
                f"acc_service={metrics['acc_service']:.4f} "
                f"acc_cuisine={metrics['acc_cuisine']:.4f}"
            )

            # save best (by val_loss)
            if metrics["val_loss"] < best_val:
                best_val = metrics["val_loss"]
                torch.save(
                    {
                        "model_state_dict": self.plm_model.state_dict(),
                        "model_name": self.model_name,
                        "label2id": label2id,
                        "id2label": id2label,
                    },
                    self.ckpt_path,
                )

        # reload best
        if os.path.isfile(self.ckpt_path):
            state = torch.load(self.ckpt_path, map_location="cpu")
            self.plm_model.load_state_dict(state["model_state_dict"])
            self.plm_model.eval()

    def predict(self, texts: list[str], device: int) -> list[dict]:
        """
        :param texts:
        :param device: device: un nombre qui identifie le numéro de la gpu sur laquelle le traitement doit se faire
        -1 veut deire que le device est la cpu, et un nombre entier >= 0 indiquera le numéro de la gpu à utiliser
        :return:
        """
        if self.plm_model is None or self.tokenizer is None:
            self._init_plmft()

        assert self.plm_model is not None
        assert self.tokenizer is not None

        torch_device = _resolve_device(device)
        self.plm_model.to(torch_device)
        self.plm_model.eval()

        bs = int(_get_cfg(self.cfg, "eval_batch_size", _get_cfg(self.cfg, "batch_size", 32)))
        max_len = int(_get_cfg(self.cfg, "max_length", 256))

        all_outputs: list[dict] = []

        for i in tqdm(range(0, len(texts), bs), desc="PLMFT predict"):
            batch_texts = texts[i : i + bs]

            enc = self.tokenizer(
                batch_texts,
                truncation=True,
                max_length=max_len,
                padding=True,
                return_tensors="pt",
            )
            enc = {k: v.to(torch_device) for k, v in enc.items()}

            with torch.no_grad():
                logits = self.plm_model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
                # logits[asp] : (B,4)
                preds = {asp: torch.argmax(logits[asp], dim=-1).tolist() for asp in ASPECTS}

            for j in range(len(batch_texts)):
                out = {asp: id2label[int(preds[asp][j])] for asp in ASPECTS}
                all_outputs.append(out)

        return all_outputs