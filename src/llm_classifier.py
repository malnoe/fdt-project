from jinja2 import Template
from openai import OpenAI
import re
import json


from config import Config

_PROMPT_TEMPLATE = """Considérez l'avis suivant:

"{{text}}"

Quelle est le sens de l'opinion exprimée sur chacun des aspects suivants : "Prix", "Cuisine", "Service" ?
L'aspect "Prix" fait référence aux prix des plats et boissons. 
L'aspect "Cuisine" fait référence à la qualité et la quantité de la nourriture et boissons. 
L'aspect "Service" fait référence à la qualité et l'efficacité du service et de l'accueil.

Pour chaque aspect, le sens de l'opinion doit être une des valeurs suivantes: "Positive", "Négative", "Neutre", ou "NE".
La valeur "Positive" concerne un avis avec une ou plusieurs opinions positive(s) sur l'aspect en question, et aucune opinions négatives sur ce même aspect.
La valeur "Négative" concerne un avis avec une ou plusieurs opinions négative(s) sur l'aspect en question, et aucune opinions positives sur ce même aspect.
La valeur "Neutre" concerne un avis avec au moins une opinion positive et une opinion négative opur le même aspect.
La valeur "NE" (qui signifie Non Exprimée) concerne un avis qui ne contient aucune opinion exprimée sur l'aspect en question.

La réponse doit se limiter au format json suivant:
{"Prix": opinion, "Cuisine": opinion, "Service": opinion}."""



class LLMClassifier:

    def __init__(self, cfg: Config):
        self.cfg = cfg
        # Instantiate an ollama client
        self.llmclient = OpenAI(base_url=cfg.ollama_url+'/v1',api_key='EMPTY')
        self.model_name = 'qwen3:1.7b'
        self.model_options = {
            'max_tokens': 500,  # max number of tokens to predict
            'temperature': 0.1,
            'top_p': 0.9,
        }
        self.jtemplate = Template(_PROMPT_TEMPLATE)
        

    EXPECTED_ASPECTS = ("Prix", "Cuisine", "Service")

    def _extract_first_json_object(self, text: str) -> str | None:
        """Extrait le premier {...} complet en gérant accolades imbriquées et strings JSON."""
        start = text.find("{")
        if start == -1:
            return None

        depth = 0
        in_str = False
        esc = False

        for i in range(start, len(text)):
            ch = text[i]

            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue

            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start:i + 1]

        return None

    def _coerce_to_json(self, blob: str) -> str:
        """
        Rend 'blob' parsable par json.loads dans des cas fréquents de sorties LLM:
        - quotes simples
        - valeurs non quotées: {"Prix": opinion} -> {"Prix": "opinion"}
        """
        s = blob.strip()

        # 1) Remplacer quotes simples par doubles (heuristique simple)
        # (suffisant dans la majorité des cas LLM)
        s = re.sub(r"(?<!\\)'", '"', s)

        # 2) Mettre entre guillemets les valeurs non-quotées (mots) après :
        #    {"Prix": opinion} ou {"Prix": NE} etc.
        #    On évite de toucher aux nombres, true/false/null, ou déjà quoté, ou objets/listes.
        def repl(m):
            val = m.group(1)
            return f': "{val}"'

        s = re.sub(
            r':\s*([A-Za-zÀ-ÖØ-öø-ÿ_][A-Za-zÀ-ÖØ-öø-ÿ0-9_\-]*)\s*(?=,|\})',
            repl,
            s
        )

        return s

    def _normalize_opinion(self, opinion: object) -> str:
        """Normalise vers {Positive, Négative, Neutre, NE}."""
        if opinion is None:
            return "NE"
        raw = str(opinion).strip()
        low = raw.lower()

        if "non exprim" in low:  # non exprimé / non exprimée / non exprimé.
            return "NE"
        # accepte déjà les labels attendus (robuste casse / accents simplifiés ici)
        if low in {"ne","non exprimé", "non exprimée", "non exprimé."}:
            return "NE"
        if low in {"positive", "positif", "positif.", "pos"}:
            return "Positive"
        if low in {"negative", "négative", "negatif", "négatif", "neg"}:
            return "Négative"
        if low in {"neutre", "neutral"}:
            return "Neutre"
        # Par défaut
        return "NE"

    def parse_json_response(self, response: str) -> dict[str, str]|None:
        blob = self._extract_first_json_object(response)
        if not blob:
            # aucun JSON détecté -> on renvoie tout à NE
            return {a: "NE" for a in self.EXPECTED_ASPECTS}

        # vérifiaction si JSON valide, sinon tentative de coercition
        try:
            jresp = json.loads(blob)
        except Exception:
            try:
                jresp = json.loads(self._coerce_to_json(blob))
            except Exception:
                return {a: "NE" for a in self.EXPECTED_ASPECTS}
        if not isinstance(jresp, dict):
            return {a: "NE" for a in self.EXPECTED_ASPECTS}

        # Normalisation des sorties + complétion des aspects manquants
        out = {a: "NE" for a in self.EXPECTED_ASPECTS}

        for aspect, opinion in jresp.items():
            a = str(aspect).strip()
            if a in out:
                out[a] = self._normalize_opinion(opinion)

        return out
    
    
    def predict(self, text: str) -> dict[str,str]:
        """
        Lance au LLM une requête contenant le texte de l'avis et les instructions pour extraire
        les opinions sur les aspects sous forme d'objet json
        :param text: le texte de l'avis
        :return: un dictionnaire python avec une entrée pour chacun des 3 aspects ayant pour valeur une des
        4 valeurs possibles pour l'opinion (Positive, Négative, Neutre et NE)
        """
        prompt = self.jtemplate.render(text=text)
        result = self.llmclient.chat.completions.create(model=self.model_name, messages=[{"role": "user", "content": prompt}], **self.model_options)
        response = result.choices[0].message.content
        jresp = self.parse_json_response(response)
        return jresp

    





## C'est un EXEMPLE de classe de classifieur avec juste un prompt sans fine-tunning.


















