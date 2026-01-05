# Import des modules et librairies nécessaires
from __future__ import annotations
from jinja2 import Template
from openai import OpenAI
import re
import json

from config import Config

import os
import random
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

from tqdm import tqdm

from transformers import (
    AutoTokenizer,
    AutoModel,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup,
    AutoConfig,
)

from transformers import TrainingArguments, Trainer
import evaluate
import numpy as np

import torch
from transformers import AutoConfig, AutoTokenizer, AutoModel, pipeline
from transformers import logging as hflogging

class AspectSentimentDataset(Dataset):
    """Custom Dataset for aspect-based sentiment analysis"""
    
    ASPECTS = ['Prix', 'Cuisine', 'Service']
    OPINION_TO_ID = {
        'Positive': 0,
        'Négative': 1,
        'Neutre': 2,
        'NE': 3
    }
    ID_TO_OPINION = {v: k for k, v in OPINION_TO_ID.items()}
    
    def __init__(self, data: list[dict], tokenizer, max_length: int = 512):
        """
        :param data: list of dicts with 'Avis', 'Prix', 'Cuisine', 'Service' keys
        :param tokenizer: huggingface tokenizer
        :param max_length: max token length for padding/truncation
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        text = sample['Avis']
        
        # Tokenize
        encoded = self.tokenizer(
            text,
            truncation=True,
            add_special_tokens=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt',
            return_attention_mask=True,
            return_token_type_ids=False
        )
        
        # Prepare labels for each aspect
        labels = {}
        for aspect in self.ASPECTS:
            opinion = sample.get(aspect, 'NE')
            labels[aspect] = self.OPINION_TO_ID.get(opinion, self.OPINION_TO_ID['NE'])
        
        return {
            'input_ids': encoded['input_ids'].squeeze(0),
            'attention_mask': encoded['attention_mask'].squeeze(0),
            'labels_prix': torch.tensor(labels['Prix'], dtype=torch.long),
            'labels_cuisine': torch.tensor(labels['Cuisine'], dtype=torch.long),
            'labels_service': torch.tensor(labels['Service'], dtype=torch.long),
        }


class MultiAspectClassifier(nn.Module):
    """Multi-task classifier for aspect-based sentiment analysis"""
    
    def __init__(self, model_name: str, num_classes: int = 4, dropout: float = 0.2):
        """
        :param model_name: pretrained model name (e.g., 'cmarkea/distilcamembert-base')
        :param num_classes: number of sentiment classes (Positive, Négative, Neutre, NE)
        :param dropout: dropout rate
        """
        super().__init__()
        self.plm = AutoModel.from_pretrained(model_name, output_attentions=False)
        hidden_size = self.plm.config.hidden_size
        
        # Classification heads for each aspect
        self.prix_classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes)
        )
        
        self.cuisine_classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes)
        )
        
        self.service_classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes)
        )
        
        self.loss_fn = nn.CrossEntropyLoss()
        
    def forward(self, input_ids, attention_mask, 
                labels_prix=None, labels_cuisine=None, labels_service=None):
        # Get [CLS] token representation
        outputs = self.plm(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        
        # Predict each aspect
        prix_logits = self.prix_classifier(cls_output)
        cuisine_logits = self.cuisine_classifier(cls_output)
        service_logits = self.service_classifier(cls_output)
        
        loss = None
        if labels_prix is not None and labels_cuisine is not None and labels_service is not None:
            loss_prix = self.loss_fn(prix_logits, labels_prix)
            loss_cuisine = self.loss_fn(cuisine_logits, labels_cuisine)
            loss_service = self.loss_fn(service_logits, labels_service)
            loss = (loss_prix + loss_cuisine + loss_service) / 3
        
        return {
            'loss': loss,
            'prix_logits': prix_logits,
            'cuisine_logits': cuisine_logits,
            'service_logits': service_logits,
        }

class PLMFTClassifier:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.model_name = "cmarkea/distilcamembert-base"
        self.lmconfig = AutoConfig.from_pretrained(self.model_name)
        self.lmtokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = None
        self.device = None
        self.checkpoint_dir = "plmft_ckpt"
        
        # Create checkpoint directory if it doesn't exist
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def train(self, train_data: list[dict], val_data: list[dict], device: int) -> None:
        """
        Fine-tune the pretrained language model on aspect-based sentiment analysis task.
        
        :param train_data: list of dicts with 'Avis', 'Prix', 'Cuisine', 'Service' keys
        :param val_data: validation data with same structure
        :param device: GPU device number (-1 for CPU)
        :return: None (saves best model to checkpoint)
        """
        # Training hyperparameters
        batch_size = 16
        learning_rate = 2e-5
        num_epochs = 5
        
        # Set device
        if device >= 0 and torch.cuda.is_available():
            self.device = torch.device(f'cuda:{device}')
        else:
            self.device = torch.device('cpu')
        
        print(f"Using device: {self.device}")
        
        # Create datasets
        train_dataset = AspectSentimentDataset(train_data, self.lmtokenizer, max_length=512)
        val_dataset = AspectSentimentDataset(val_data, self.lmtokenizer, max_length=512)
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0
        )
        
        # Initialize model
        self.model = MultiAspectClassifier(self.model_name, num_classes=4, dropout=0.2)
        self.model.to(self.device)
        
        # Setup optimizer and scheduler
        optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        total_steps = len(train_loader) * num_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps
        )
        
        # Training loop
        best_val_loss = float('inf')
        best_model_path = os.path.join(self.checkpoint_dir, 'best.pt')
        
        for epoch in range(num_epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            
            with tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}") as pbar:
                for batch in pbar:
                    optimizer.zero_grad()
                    
                    # Move batch to device
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels_prix = batch['labels_prix'].to(self.device)
                    labels_cuisine = batch['labels_cuisine'].to(self.device)
                    labels_service = batch['labels_service'].to(self.device)
                    
                    # Forward pass
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels_prix=labels_prix,
                        labels_cuisine=labels_cuisine,
                        labels_service=labels_service
                    )
                    
                    loss = outputs['loss']
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    
                    train_loss += loss.item()
                    pbar.set_postfix({'loss': loss.item()})
            
            avg_train_loss = train_loss / len(train_loader)
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels_prix = batch['labels_prix'].to(self.device)
                    labels_cuisine = batch['labels_cuisine'].to(self.device)
                    labels_service = batch['labels_service'].to(self.device)
                    
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels_prix=labels_prix,
                        labels_cuisine=labels_cuisine,
                        labels_service=labels_service
                    )
                    
                    val_loss += outputs['loss'].item()
            
            avg_val_loss = val_loss / len(val_loader)
            
            print(f"Epoch {epoch+1}/{num_epochs} - "
                  f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
            
            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(self.model.state_dict(), best_model_path)
                print(f"✓ Saved best model with val loss: {avg_val_loss:.4f}")
        
        # Load best model
        self.model.load_state_dict(torch.load(best_model_path))
        print("Training complete. Best model loaded.")

    def predict(self, texts: list[str]) -> list[dict[str, str]]:
        """
        Predict sentiment for each aspect of the given texts using the fine-tuned model.
        
        :param texts: list of review texts
        :return: list of dicts with keys 'Prix', 'Cuisine', 'Service' and values in 
                 {'Positive', 'Négative', 'Neutre', 'NE'}
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first or load a checkpoint.")
        
        self.model.eval()
        
        # Create dataset without labels
        class PredictionDataset(Dataset):
            def __init__(self, texts, tokenizer, max_length=512):
                self.texts = texts
                self.tokenizer = tokenizer
                self.max_length = max_length
            
            def __len__(self):
                return len(self.texts)
            
            def __getitem__(self, idx):
                text = self.texts[idx]
                encoded = self.tokenizer(
                    text,
                    truncation=True,
                    add_special_tokens=True,
                    padding='max_length',
                    max_length=self.max_length,
                    return_tensors='pt',
                    return_attention_mask=True,
                    return_token_type_ids=False
                )
                
                return {
                    'input_ids': encoded['input_ids'].squeeze(0),
                    'attention_mask': encoded['attention_mask'].squeeze(0),
                }
        
        pred_dataset = PredictionDataset(texts, self.lmtokenizer, max_length=512)
        batch_size = 16
        pred_loader = DataLoader(
            pred_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0
        )
        
        all_predictions = []
        
        with torch.no_grad():
            for batch in pred_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                # Get predictions for each aspect
                prix_preds = torch.argmax(outputs['prix_logits'], dim=1)
                cuisine_preds = torch.argmax(outputs['cuisine_logits'], dim=1)
                service_preds = torch.argmax(outputs['service_logits'], dim=1)
                
                # Convert to opinion strings
                for p, c, s in zip(prix_preds, cuisine_preds, service_preds):
                    all_predictions.append({
                        'Prix': AspectSentimentDataset.ID_TO_OPINION[p.item()],
                        'Cuisine': AspectSentimentDataset.ID_TO_OPINION[c.item()],
                        'Service': AspectSentimentDataset.ID_TO_OPINION[s.item()],
                    })
        
        return all_predictions
    