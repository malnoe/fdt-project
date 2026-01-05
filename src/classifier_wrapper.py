from pandas import DataFrame
from tqdm import tqdm

from config import Config

from llm_classifier import LLMClassifier
from fine_tune import PLMFTClassifier


class ClassifierWrapper:

    # METTRE LA BONNE VALEUR ci-dessous en fonction de la méthode utilisée
    METHOD: str = 'PLMFT'  # or 'LLM'
    #############################################################################################
    # NE PAS MODIFIER LA SIGNATURE DE CETTE FONCTION, Vous pouvez modifier son contenu si besoin
    #############################################################################################
    def __init__(self, cfg: Config):
        print("ClassifierWrapper using method:", self.METHOD)
        self.cfg = cfg
        if self.METHOD=='LLM' :
            self.classifier = LLMClassifier(cfg)
        else:
            self.classifier = PLMFTClassifier(cfg)


    #############################################################################################
    # NE PAS MODIFIER LA SIGNATURE DE CETTE FONCTION, Vous pouvez modifier son contenu si besoin
    #############################################################################################
    def train(self, train_data: list[dict], val_data: list[dict], device: int) -> None:
        """
        :param train_data:
        :param val_data:
        :param device: un nombre qui identifie le numéro de la gpu sur laquelle le traitement doit se faire
        -1 veut deire que le device est la cpu, et un nombre entier >= 0 indiquera le numéro de la gpu
        :return:
        """
        if self.METHOD=='PLMFT':
            self.classifier.train(train_data, val_data, device)
        else:
            pass


    #############################################################################################
    # NE PAS MODIFIER LA SIGNATURE DE CETTE FONCTION, Vous pouvez modifier son contenu si besoin
    #############################################################################################
    def predict(self, texts: list[str], device: int) -> list[dict]:
        """
        :param texts:
        :param device: device: un nombre qui identifie le numéro de la gpu sur laquelle le traitement doit se faire
        -1 veut deire que le device est la cpu, et un nombre entier >= 0 indiquera le numéro de la gpu à utiliser
        :return:
        """
        if self.METHOD=='PLMFT':
            # Si fine-tuned PLM, on peut faire du batch processing
            all_opinions = self.classifier.predict(texts)
            return all_opinions
        else:
            # Si LLM, on fait du one-by-one
            all_opinions = []
            for text in tqdm(texts):
                opinions = self.classifier.predict(text)
                all_opinions.append(opinions)
            return all_opinions
        