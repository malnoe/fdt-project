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
        self.model_name = 'llama3.2:1b'
        self.model_options = {
            'max_tokens': 500,  # max number of tokens to predict
            'temperature': 0.1,
            'top_p': 0.9,
        }
        self.jtemplate = Template(_PROMPT_TEMPLATE)


    def predict(self, text: str) -> dict[str,str]:
        """
        Lance au LLM une requête contenant le texte de l'avis et les instructions pour extraire
        les opinions sur les aspects sous forme d'objet json
        :param text: le texte de l'avis
        :return: un dictionnaire python avec une entrée pour chacun des 4 aspects ayant pour valeur une des
        4 valeurs possibles pour l'opinion (Positive, Négative, Neutre et NE)
        """
        prompt = self.jtemplate.render(text=text)
        result = self.llmclient.chat.completions.create(model=self.model_name, messages=[{"role": "user", "content": prompt}], **self.model_options)
        response = result.choices[0].message.content
        jresp = self.parse_json_response(response)
        return jresp

    def parse_json_response(self, response: str) -> dict[str, str] | None:
        m = re.findall(r"\{[^\{\}]+\}", response, re.DOTALL)
        if m:
            try:
                jresp = json.loads(m[0])
                for aspect, opinion in jresp.items():
                    if "non exprim" in opinion.lower():
                        jresp[aspect] = "NE"
                return jresp
            except:
                return None
        else:
            return None





## C'est un EXEMPLE de classe de classifieur avec juste un prompt sans fine-tunning.


















