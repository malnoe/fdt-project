Garance MALNOË & Matthias MAZET
M2 SSD


##### Approche par LLM sans entraînement #####

Notre approche s'est construite en plusieurs étapes :
    Étape 1. Tests de différents prompts sur les différents LLMs.
Nous avons rédigé différents prompts afin d'aborder le problème sous différents angles. Nous avons testé, pour chaque LLM : 
  - Un prompt basique détaillant les différents aspects à noter (Cuisine, Prix et Service) et les différents grades de notations (Positive, Neutre, Négative et Non Exprimée).
  - Une version anglaise du prompt précédent, avec les tags en français pour conserver la cohérence de sortie.
  - Une version plus détaillée du premier prompt, avec notamment l'ajout de notions de références et de quelques mots exemples pour chaque aspect (e.g. : "L'aspect "Prix" fait référence aux prix des plats et boissons. Il concerne tout ce qui parle d'argent, de coût, de tarifs, etc., avec un vocabulaire tel que "trop cher", "prix", "rapport qualité/prix", etc.").
  - Une version anglaise du prompt précédent.
  - Une version du premier prompt où nous demandions au LLM d'expliquer sa démarche (certaines études ont montraient que les performances d'un LLM augmentaient simplement grâce à cette demande).
  - Une version du premier prompt avec l'ajout d'exemples de classifications d'opinions.
  - Une combinaison de plusieurs approches, i.e. un prompt plus détaillé avec l'ajout d'exemples de classification d'opinions. Nous n'avons pas ajouté l'explication de la démarche car le temps de calcul était trop long par rapport à l'efficacité du prompt.
Cette première étape nous a permis de choisir la meilleure combinaison prompt/LLM. À quelques exceptions près, le LLM avec les meilleures performances en termes d'accuracy moyenne et de temps de calcul était le gemma2:2b. La meilleure accuracy moyenne avec ce LLM a été obtenue avec le prompt basique en anglais. Nous avons pu constater que l'ajout d'exemples détériorait significativement l'accuracy moyenne, tandis que l'explication de la démarche augmentait grandement le temps de calcul. L'ajout de détails n'a eu que peu d'impact sur l'accuracy moyenne mais augmentait le temps d'exécution, c'est pourquoi nous ne l'avons pas conservé.

    Étape 2. Modifications des hyperparamètres.
Une fois la meilleure combinaison prompt/LLM retenue (prompt basique en anglais avec gemma2:2b), nous avons testé différentes combinaisons de valeurs pour les hyperparamètres de température `temp` et `top_p` afin d'observer leur impact sur l'accuracy moyenne et le temps d'exécution. Ainsi, nous avons noté un intérêt à diminuer la top_p jusqu'à un certain seuil (environ 0.5).

    Étape 3. Répétition de LLMs.
Toujours avec la combinaison prompt/LLM retenue à l'étape 1, nous avons ensuite tenté de répéter trois fois le même LLM avec une température élevée afin d'observer les variations de classifications. Par un vote à la majorité sur les trois exécutions, nous n'avons finalement pas constaté de réel intérêt à cette approche, et nous n'avons donc pas poussé la démarche plus loin (notamment en raison de temps d'exécution plutôt conséquents).
    
    Étape 4. Un prompt par aspect.
La dernière étape tentée pour cette approche a été la construction de prompt individuel pour chaque aspect.
Nous avons testé des prompts basiques en français et en anglais pour chaque aspect (même prompts que précédemment mais redécoupé pour chaque aspect), dont nous avons ensuite combiner les prédictions afin d'obtenir le format final souhaité. Cette approche n'ayant pas donné de bons résultats en termes d'accuracy moyenne, nous n'avons pas poussé la réflexion plus loin.


Pour cette approche, nous avons finalement retenu le LLM gemma2:2b avec notre prompt en anglais, une température de 0.1 et une top_p de 0.5. Avec cette combinaison, nous avons obtenu une accuracy moyenne de 83.78 pour un temps d'exécution de 601 secondes. Le prompt utilisé était le suivant :

"""Considere the following review:

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
    {"Prix": opinion, "Cuisine": opinion, "Service": opinion}.”””


________________________________________________________________________________
________________________________________________________________________________


##### Approche par PLMFT #####





