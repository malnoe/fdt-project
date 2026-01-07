from jinja2 import Template
from openai import OpenAI
import re
import json


from config import Config

# Template général
_PROMPT_TEMPLATE = """Considere the following review:

    "{{text}}"
    What is the meaning of the opinion expressed on each of the following aspects: "Prix", "Cuisine", "Service"?
    The "Prix" aspect refers to the prices of food and beverages. 
    The "Cuisine" aspect refers to the quality and quantity of food and beverages. 
    The “Service” aspect refers to the quality and efficiency of the service and reception.

    For each aspect, the meaning of the opinion must be one of the following values: "Positive", "Négative", "Neutre", or "NE".
    The value “Positive” concerns a review with one or more positive opinions on the aspect in question, and no negative opinions on the same aspect.
    The value “Négative” concerns a review with one or more negative opinions on the aspect in question, and no positive opinions on the same aspect.
    The value “Neutre” concerns a review with at least one positive opinion and one negative opinion on the same aspect.
    The value "NE" (which stands for Non Expressed) refers to a review that does not contain any opinion expressed on the aspect in question.

    The response must be limited to the following json format:
    {"Prix": opinion, "Cuisine": opinion, "Service": opinion}."""

# Templates par aspect
_PROMPT_TEMPLATE_SERVICE = """
    Considère l'avis suivant :

    "{{text}}"
    Quel est l'opinion exprimée pour le service ? Le client est-il satisfait ou insatisfait du service et de l'accueil que ce soit la qualité ou l'efficacité ?
    
    L'opinion pour le service doit être l'une des valeurs suivantes : "Positive", "Négative", "Neutre", ou "NE".
    La valeur “Positive” concerne un avis avec une ou plusieurs opinions positives sur le service, et aucune opinion négative sur le même aspect.
    La valeur “Négative” concerne un avis avec une ou plusieurs opinions négatives sur le service, et aucune opinion positive sur le même aspect.
    La valeur “Neutre” concerne un avis avec au moins une opinion positive et une opinion négative sur le service.
    La valeur "NE" (qui signifie Non Exprimé) fait référence à un avis qui ne contient aucune opinion exprimée sur le service.
    
    Commence par détailler brièvement les éléments de l'avis qui t'ont permis de déterminer cette opinion puis donne la réponse de l'opinion sous la forme {"Service": opinion}.
"""
    
_PROMPT_TEMPLATE_CUISINE = """
Considère l'avis suivant :

    "{{text}}"
    Quel est l'opinion exprimée pour la cuisine ? Le client est-il satisfait ou insatisfait de la qualité et de la quantité des plats et boissons proposées ?
    
    L'opinion pour la cuisine doit être l'une des valeurs suivantes : "Positive", "Négative", "Neutre", ou "NE".
    La valeur “Positive” concerne un avis avec une ou plusieurs opinions positives sur la cuisine, et aucune opinion négative sur le même aspect.
    La valeur “Négative” concerne un avis avec une ou plusieurs opinions négatives sur la cuisine, et aucune opinion positive sur le même aspect.
    La valeur “Neutre” concerne un avis avec au moins une opinion positive et une opinion négative sur la cuisine.
    La valeur "NE" (qui signifie Non Exprimé) fait référence à un avis qui ne contient aucune opinion exprimée sur la cuisine.
    
    Commence par détailler brièvement les éléments de l'avis qui t'ont permis de déterminer cette opinion puis donne la réponse de l'opinion sous la forme {"Cuisine": opinion}.
"""

_PROMPT_TEMPLATE_PRIX = """
Considère l'avis suivant :

    "{{text}}"
    Quel est l'opinion exprimée pour le prix ? Le client est-il satisfait ou insatisfait des prix proposé ou du rapport qualité/prix ?
    
    L'opinion pour le prix doit être l'une des valeurs suivantes : "Positive", "Négative", "Neutre", ou "NE".
    La valeur “Positive” concerne un avis avec une ou plusieurs opinions positives sur le prix, et aucune opinion négative sur le même aspect.
    La valeur “Négative” concerne un avis avec une ou plusieurs opinions négatives sur le prix, et aucune opinion positive sur le même aspect.
    La valeur “Neutre” concerne un avis avec au moins une opinion positive et une opinion négative sur le prix.
    La valeur "NE" (qui signifie Non Exprimé) fait référence à un avis qui ne contient aucune opinion exprimée sur le prix.

    Commence par détailler brièvement les éléments de l'avis qui t'ont permis de déterminer cette opinion puis donne la réponse de l'opinion sous la forme {"Prix": opinion}.
    """

# Definition d'une nouvelle class LLMClassifier avec une méthode __init__ et predict pour l'utilisation dans classifier_wrapper.py
class LLMClassifier:
    
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.llmclient = OpenAI(base_url=cfg.ollama_url+'/v1',api_key='EMPTY')
        self.model_name = 'gemma2:2b'
        self.model_options = {
            'max_tokens': 1000,  # max number of tokens to predict
            'temperature': 0.1,
            'top_p': 0.5,
        }
        self.jtemplate = Template(_PROMPT_TEMPLATE)
        self.jtemplate_prix = Template(_PROMPT_TEMPLATE_PRIX)
        self.jtemplate_cuisine = Template(_PROMPT_TEMPLATE_CUISINE)
        self.jtemplate_service = Template(_PROMPT_TEMPLATE_SERVICE)

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

        # 1. Remplacer quotes simples par doubles (heuristique simple)
        # (suffisant dans la majorité des cas LLM)
        s = re.sub(r"(?<!\\)'", '"', s)

        # 2. Mettre entre guillemets les valeurs non-quotées (mots) après :
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
        """Extrait et parse le premier objet JSON dans la réponse du LLM avec normalisation."""
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
    
    # 1. Méthode avec un seul appel au LLM et parsing JSON complet
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
    
    # 2. Méthode avec trois fois appel au même LLM et vote majoritaire
    #def predict(self, text: str) -> dict[str,str]:
        """
        Lance le LLM trois fois pour chaque contribution.
        Récupère pour chaque aspect l'opinion prédite et fait un vote majoritaire.

        :param text: le texte de l'avis
        :return: un dictionnaire python avec une entrée pour chacun des 3 aspects ayant pour valeur une des
        4 valeurs possibles pour l'opinion (Positive, Négative, Neutre et NE)
        """
        # Initialisation des listes de réponses
        prix_responses = []
        cuisine_responses = []
        service_responses = []
        # Appel LLM trois fois
        for i in range(3):
            prompt = self.jtemplate.render(text=text)
            result = self.llmclient.chat.completions.create(model=self.model_name, messages=[{"role": "user", "content": prompt}], **self.model_options)
            response = result.choices[0].message.content
            jresp = self.parse_json_response(response)
            # Append aux listes de réponses
            prix_responses.append(jresp["Prix"])
            cuisine_responses.append(jresp["Cuisine"])
            service_responses.append(jresp["Service"])

        # Vote majoritaire
        def vote_majority(responses):
            counts = {}
            for r in responses:
                counts[r] = counts.get(r, 0) + 1
            return max(counts, key=lambda k: counts.get(k, 0))
        
        # Retour du json final avec les votes majoritaires
        return {
            "Prix": vote_majority(prix_responses),
            "Cuisine": vote_majority(cuisine_responses),
            "Service": vote_majority(service_responses)
        }
    
    # 3. Méthode avec 1 appel par aspect
    #def predict(self, text: str) -> dict[str,str]:
        """
        Lance le LLM une fois pour chaque aspect d'une contribution.
        Récupère pour chaque aspect l'opinion prédite.

        :param text: le texte de l'avis
        :return: un dictionnaire python avec une entrée pour chacun des 3 aspects ayant pour valeur une des
        4 valeurs possibles pour l'opinion (Positive, Négative, Neutre et NE)
        """
        # Prix
        prompt_prix = self.jtemplate_prix.render(text=text)
        result_prix = self.llmclient.chat.completions.create(model=self.model_name, messages=[{"role": "user", "content": prompt_prix}], **self.model_options)
        response_prix = result_prix.choices[0].message.content
        prix_response_json = self.parse_json_response(response_prix)
        prix_response = prix_response_json["Prix"]
        # Cuisine
        prompt_cuisine = self.jtemplate_cuisine.render(text=text)
        result_cuisine = self.llmclient.chat.completions.create(model=self.model_name, messages=[{"role": "user", "content": prompt_cuisine}], **self.model_options)
        response_cuisine = result_cuisine.choices[0].message.content
        cuisine_response_json = self.parse_json_response(response_cuisine)
        cuisine_response = cuisine_response_json["Cuisine"]
        # Service
        prompt_service = self.jtemplate_service.render(text=text)
        result_service = self.llmclient.chat.completions.create(model=self.model_name, messages=[{"role": "user", "content": prompt_service}], **self.model_options)
        response_service = result_service.choices[0].message.content
        service_response_json = self.parse_json_response(response_service)
        service_response = service_response_json["Service"]
        
        # Retour du json final avec les votes majoritaires
        return {
            "Prix": prix_response,
            "Cuisine": cuisine_response,
            "Service": service_response
        }
    

















