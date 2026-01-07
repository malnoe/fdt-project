Garance MALNOË & Matthias MAZET
M2 SSD
##### Introduction #####

Expliquer l'objectif du projet, la démarche globale pour mener le projet et les 2 approches testés.

##### Approche par LLM sans entraînement #####

Notre approche s'est construite en plusieurs étapes :
    Étape 1. Tests de différents prompts sur les différents modèles de LLMs.
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

L'ensemble des autres prompts testés sont disponibles en annexe de ce read-me.
________________________________________________________________________________
________________________________________________________________________________


##### Approche par PLMFT #####



________________________________________________________________________________
________________________________________________________________________________


##### Conclusion #####
...

________________________________________________________________________________
________________________________________________________________________________

##### Annexe #####

***Prompt 1 :***  
    """Considérez l'avis suivant:

    "{{text}}"

    Quel est le sens de l'opinion exprimée sur chacun des aspects suivants : "Prix", "Cuisine", "Service" ?  
    L'aspect "Prix" fait référence aux prix des plats et boissons.   
    L'aspect "Cuisine" fait référence à la qualité et la quantité de la nourriture et boissons.   
    L'aspect "Service" fait référence à la qualité et l'efficacité du service et de l'accueil.

    Pour chaque aspect, le sens de l'opinion doit être une des valeurs suivantes: "Positive", "Négative", "Neutre", ou "NE".  
    La valeur "Positive" concerne un avis avec une ou plusieurs opinions positives sur l'aspect en question, et aucune opinion négative sur ce même aspect.  
    La valeur "Négative" concerne un avis avec une ou plusieurs opinions négatives sur l'aspect en question, et aucune opinion positive sur ce même aspect.  
    La valeur "Neutre" concerne un avis avec au moins une opinion positive et une opinion négative pour le même aspect.  
    La valeur "NE" (qui signifie Non Exprimée) concerne un avis qui ne contient aucune opinion exprimée sur l'aspect en question.

    La réponse doit se limiter au format json suivant:  
    {"Prix": opinion, "Cuisine": opinion, "Service": opinion}."""

---

***Prompt 2 (version légèrement modifié du Prompt 1) :***  
"""Considérez l'avis suivant:

   "{{text}}"

   Quel est le sens de l'opinion générale exprimée sur chacun des aspects suivants : "Prix", "Cuisine", "Service" ?  
   L'aspect "Prix" fait référence aux prix des plats et boissons.  
   L'aspect "Cuisine" fait référence à la qualité et la quantité de la nourriture et boissons.  
   L'aspect "Service" fait référence à la qualité et l'efficacité du service et de l'accueil.

   Pour chaque aspect, le sens de l'opinion doit être une des valeurs suivantes: "Positive", "Négative", "Neutre", ou "NE".  
   La valeur "Positive" concerne un aspect avec une ou plusieurs opinions positives et aucune opinion négative.  
   La valeur "Négative" concerne un aspect avec une ou plusieurs opinions négatives et aucune opinion positive.  
   La valeur "Neutre" concerne un aspect avec au moins une opinion positive et une opinion négative exprimées.  
   La valeur "NE" (qui signifie Non Exprimée) concerne un aspect sans opinion exprimée dans l’avis. C’est la valeur par défaut si aucune décision n’est prise.

   La réponse doit se limiter au format json suivant:  
   {"Prix": opinion, "Cuisine": opinion, "Service": opinion}."""

---

***Prompt 3 (Version en anglais du Prompt 1 avec tags en français) :***  
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
---

***Prompt 4 (version plus détaillée du Prompt 1) :***  
"""Considérez l'avis suivant:

   "{{text}}"

   Quel est le sens de l'opinion générale exprimée sur chacun des aspects suivants : "Prix", "Cuisine", "Service" ?  
    L'aspect "Prix" fait référence aux prix des plats et boissons. Il concerne tout ce qui parle d’argent, de coût, de tarifs, etc., avec un vocabulaire tel que “trop cher”, “prix”, “rapport qualité/prix”, etc.  
   L'aspect "Cuisine" fait référence à la qualité et la quantité de la nourriture et boissons. Il concerne notamment le goût, le visuel et la contenance des plats, avec un vocabulaire tel que “quantité”, “goût”, “visuel”, etc.  
   L'aspect "Service" fait référence à la qualité et l'efficacité du service et de l'accueil. Il concerne la rapidité du service, la qualité de l’accueil, etc., avec un vocabulaire tel que “accueil”, “personnel”, “service”, “efficace”, etc.

    Pour chaque aspect, le sens de l'opinion doit être une des valeurs suivantes: "Positive", "Négative", "Neutre", ou "NE".  
    La valeur "Positive" concerne un avis avec une ou plusieurs opinions positives sur l'aspect en question, et aucune opinion négative sur ce même aspect.  
    La valeur "Négative" concerne un avis avec une ou plusieurs opinions négatives sur l'aspect en question, et aucune opinion positive sur ce même aspect.  
    La valeur "Neutre" concerne un avis avec au moins une opinion positive et une opinion négative pour le même aspect.  
    La valeur "NE" (qui signifie Non Exprimée) concerne un avis qui ne contient aucune opinion exprimée sur l'aspect en question.

    La réponse doit se limiter au format json suivant:  
    {"Prix": opinion, "Cuisine": opinion, "Service": opinion}."""

---

***Prompt 5 (Prompt 1 avec explication de la démarche) :***  
    """Considérez l'avis suivant:

    "{{text}}"

    Quel est le sens de l'opinion exprimée sur chacun des aspects suivants : "Prix", "Cuisine", "Service" ?  
    L'aspect "Prix" fait référence aux prix des plats et boissons.   
    L'aspect "Cuisine" fait référence à la qualité et la quantité de la nourriture et boissons.   
    L'aspect "Service" fait référence à la qualité et l'efficacité du service et de l'accueil.

    Pour chaque aspect, le sens de l'opinion doit être une des valeurs suivantes: "Positive", "Négative", "Neutre", ou "NE".  
    La valeur "Positive" concerne un avis avec une ou plusieurs opinions positives sur l'aspect en question, et aucune opinion négative sur ce même aspect.  
    La valeur "Négative" concerne un avis avec une ou plusieurs opinions négatives sur l'aspect en question, et aucune opinion positive sur ce même aspect.  
    La valeur "Neutre" concerne un avis avec au moins une opinion positive et une opinion négative pour le même aspect.  
    La valeur "NE" (qui signifie Non Exprimée) concerne un avis qui ne contient aucune opinion exprimée sur l'aspect en question.

    Explique ton raisonnement pour chaque prise de décision sur chaque aspect dans une réponse intermédiaire, puis donne ta réponse finale.

    La réponse doit se limiter au format json suivant:  
    {"Prix": opinion, "Cuisine": opinion, "Service": opinion}."""

---

***Prompt 6 (Prompt 1 avec exemple) :***  
    """Considérez l'avis suivant:

    "{{text}}"

    Quel est le sens de l'opinion exprimée sur chacun des aspects suivants : "Prix", "Cuisine", "Service" ?  
    L'aspect "Prix" fait référence aux prix des plats et boissons.   
    L'aspect "Cuisine" fait référence à la qualité et la quantité de la nourriture et boissons.   
    L'aspect "Service" fait référence à la qualité et l'efficacité du service et de l'accueil.

    Pour chaque aspect, le sens de l'opinion doit être une des valeurs suivantes: "Positive", "Négative", "Neutre", ou "NE".  
    La valeur "Positive" concerne un avis avec une ou plusieurs opinions positives sur l'aspect en question, et aucune opinion négative sur ce même aspect.  
    La valeur "Négative" concerne un avis avec une ou plusieurs opinions négatives sur l'aspect en question, et aucune opinion positive sur ce même aspect.  
    La valeur "Neutre" concerne un avis avec au moins une opinion positive et une opinion négative pour le même aspect.  
    La valeur "NE" (qui signifie Non Exprimée) concerne un avis qui ne contient aucune opinion exprimée sur l'aspect en question.

    La réponse doit se limiter au format json suivant:  
    {"Prix": “opinion”, "Cuisine": “opinion”, "Service": “opinion”}.

    Par exemple, la réponse de l’avis “Trop trop long. Repas correct et pas exceptionnel Salle a l etage pas vraiement adapte por le service d un groupe tel que nous etions” serait {"Prix": “NE”, "Cuisine": “Neutre”, "Service": “Négative”}.   
    La réponse pour l’avis "Les glaces sont très très chères, comptez 5€ pour une glace 2 boules\! Pas de terrasse et service à la chaîne. Bref, les glaces vendues en grande surface ont le même goût et sont beaucoup moins chères" serait {"Prix": “Négative”, "Cuisine": “Neutre”, "Service": “Négative”}."""

---

***Prompt 7 (Mix 1 (sans explications du raisonnement car trop long)) :***  
    """Considérez l'avis suivant:

    "{{text}}"

    Quel est le sens de l'opinion générale exprimée sur chacun des aspects suivants : "Prix", "Cuisine", "Service" ?  
    L'aspect "Prix" fait référence aux prix des plats et boissons. Il concerne tout ce qui parle d’argent, de coût, de tarifs, etc., avec un vocabulaire tel que “trop cher”, “prix”, “rapport qualité/prix”, etc.  
   L'aspect "Cuisine" fait référence à la qualité et la quantité de la nourriture et boissons. Il concerne notamment le goût, le visuel et la contenance des plats, avec un vocabulaire tel que “quantité”, “goût”, “visuel”, etc.  
   L'aspect "Service" fait référence à la qualité et l'efficacité du service et de l'accueil. Il concerne la rapidité du service, la qualité de l’accueil, etc., avec un vocabulaire tel que “accueil”, “personnel”, “service”, “efficace”, etc.

    Pour chaque aspect, le sens de l'opinion doit être une des valeurs suivantes: "Positive", "Négative", "Neutre", ou "NE".  
    La valeur "Positive" concerne un avis avec une ou plusieurs opinions positives sur l'aspect en question, et aucune opinion négative sur ce même aspect.  
    La valeur "Négative" concerne un avis avec une ou plusieurs opinions négatives sur l'aspect en question, et aucune opinion positive sur ce même aspect.  
    La valeur "Neutre" concerne un avis avec au moins une opinion positive et une opinion négative pour le même aspect.  
    La valeur "NE" (qui signifie Non Exprimée) concerne un avis qui ne contient aucune opinion exprimée sur l'aspect en question.

    La réponse doit se limiter au format json suivant:  
    {"Prix": “opinion”, "Cuisine": “opinion”, "Service": “opinion”}.

    Par exemple, la réponse de l’avis “Trop trop long. Repas correct et pas exceptionnel Salle a l etage pas vraiement adapte por le service d un groupe tel que nous etions” serait {"Prix": “NE”, "Cuisine": “Neutre”, "Service": “Négative”}.   
    La réponse pour l’avis "Les glaces sont très très chères, comptez 5€ pour une glace 2 boules\! Pas de terrasse et service à la chaîne. Bref, les glaces vendues en grande surface ont le même goût et sont beaucoup moins chères" serait {"Prix": “Négative”, "Cuisine": “Neutre”, "Service": “Négative”}."""

---

***Prompt 8 (Mix 2 (anglais + détails)) :***  
 """Consider the following review:

   "{{text}}"

What is the sentiment associated with each of the following aspects : Price (“Prix”), Cuisine/Cooking (“Cuisine”) and Service (“Service”) ?  
The price aspect ("Prix") refers to the price of the food and beverages. It refers to everything related to money, cost, rate, etc. The associated vocabulary can be "trop cher", "prix", "rapport qualité/prix",etc.  
The cuisine/cooking aspect ("Cuisine") refers to the quality and quantity of food and beverages. It refers to the taste, appearance and size of dishes, with vocabulary such as "quantité", "goût", "visuel", etc.  
The "Service" aspect refers to the quality and efficiency of service and hospitality/reception. It concerns the speed of service, the quality of hospitality, etc., with vocabulary such as "acceuil", "personnel"', "service", "efficace","rapide","agréable", etc.  
For each aspect, the meaning of the opinion must be one of the following values: "Positive", "Négative", "Neutre", or "NE".  
    The value "Positive" concerns a review with one or more positive opinions on the aspect in question, and no negative opinions on the same aspect.  
    The value "Négative" concerns a review with one or more negative opinions on the aspect in question, and no positive opinions on the same aspect.  
    The value "Neutre" concerns a review with at least one positive opinion and one negative opinion on the same aspect.  
    The value "NE" (which stands for Non Expressed) refers to a review that does not contain any opinion expressed on the aspect in question.

    The response must be limited to the following json format:  
    {"Prix": opinion, "Cuisine": opinion, "Service": opinion}."""

---

***Prompt Prix 1 :***  
"""  
    Considere the following review:

    "{{text}}"  
    What is the meaning of the opinion expressed the "Prix" aspect ? The "Prix" aspect refers to the prices of food and beverages.

    For each aspect, the meaning of the opinion must be one of the following values: "Positive", "Négative", "Neutre", or "NE".  
    The value “Positive” concerns a review with one or more positive opinions on the aspect in question, and no negative opinions on the same aspect.  
    The value “Négative” concerns a review with one or more negative opinions on the aspect in question, and no positive opinions on the same aspect.  
    The value “Neutre” concerns a review with at least one positive opinion and one negative opinion on the same aspect.  
    The value "NE" (which stands for Non Expressed) refers to a review that does not contain any opinion expressed on the aspect in question.

    The response must be limited to the opinion value only."""  
---

***Prompt Cuisine 1 :***  
"""  
    Considere the following review:

    "{{text}}"  
    What is the meaning of the opinion expressed the "Service" aspect ? The “Service” aspect refers to the quality and efficiency of the service and reception.

    For each aspect, the meaning of the opinion must be one of the following values: "Positive", "Négative", "Neutre", or "NE".  
    The value “Positive” concerns a review with one or more positive opinions on the aspect in question, and no negative opinions on the same aspect.  
    The value “Négative” concerns a review with one or more negative opinions on the aspect in question, and no positive opinions on the same aspect.  
    The value “Neutre” concerns a review with at least one positive opinion and one negative opinion on the same aspect.  
    The value "NE" (which stands for Non Expressed) refers to a review that does not contain any opinion expressed on the aspect in question.

    The response must be limited to the opinion value only."""

---

***Prompt Service 1 :***  
"""  
    Considere the following review:

    "{{text}}"  
    What is the meaning of the opinion expressed the "Service" aspect ? The “Service” aspect refers to the quality and efficiency of the service and reception.

    For each aspect, the meaning of the opinion must be one of the following values: "Positive", "Négative", "Neutre", or "NE".  
    The value “Positive” concerns a review with one or more positive opinions on the aspect in question, and no negative opinions on the same aspect.  
    The value “Négative” concerns a review with one or more negative opinions on the aspect in question, and no positive opinions on the same aspect.  
    The value “Neutre” concerns a review with at least one positive opinion and one negative opinion on the same aspect.  
    The value "NE" (which stands for Non Expressed) refers to a review that does not contain any opinion expressed on the aspect in question.

    The response must be limited to the opinion value only."""

---

***Prompt Cuisine 2 :***

"""  
Considère l'avis suivant :

    "{{text}}"  
    Quel est l'opinion exprimée pour la cuisine ? Le client est-il satisfait ou insatisfait de la qualité et de la quantité des plats et boissons proposées ?  
      
    L'opinion pour la cuisine doit être l'une des valeurs suivantes : "Positive", "Négative", "Neutre", ou "NE".  
    La valeur “Positive” concerne un avis avec une ou plusieurs opinions positives sur la cuisine, et aucune opinion négative sur le même aspect.  
    La valeur “Négative” concerne un avis avec une ou plusieurs opinions négatives sur la cuisine, et aucune opinion positive sur le même aspect.  
    La valeur “Neutre” concerne un avis avec au moins une opinion positive et une opinion négative sur la cuisine.  
    La valeur "NE" (qui signifie Non Exprimé) fait référence à un avis qui ne contient aucune opinion exprimée sur la cuisine.  
      
    La réponse doit se limiter à la valeur de l'opinion uniquement.  
"""  
---

***Prompt Service 2 :***

"""  
    Considère l'avis suivant :

    "{{text}}"  
    Quel est l'opinion exprimée pour le service ? Le client est-il satisfait ou insatisfait du service et de l'accueil que ce soit la qualité ou l'efficacité ?  
      
    L'opinion pour le service doit être l'une des valeurs suivantes : "Positive", "Négative", "Neutre", ou "NE".  
    La valeur “Positive” concerne un avis avec une ou plusieurs opinions positives sur le service, et aucune opinion négative sur le même aspect.  
    La valeur “Négative” concerne un avis avec une ou plusieurs opinions négatives sur le service, et aucune opinion positive sur le même aspect.  
    La valeur “Neutre” concerne un avis avec au moins une opinion positive et une opinion négative sur le service.  
    La valeur "NE" (qui signifie Non Exprimé) fait référence à un avis qui ne contient aucune opinion exprimée sur le service.  
      
    La réponse doit se limiter à la valeur de l'opinion uniquement."""  
---

***Prompt Prix 2 :***

"""  
Considère l'avis suivant :

    "{{text}}"  
    Quel est l'opinion exprimée pour le prix ? Le client est-il satisfait ou insatisfait des prix proposé ou du rapport qualité/prix ?  
      
    L'opinion pour le prix doit être l'une des valeurs suivantes : "Positive", "Négative", "Neutre", ou "NE".  
    La valeur “Positive” concerne un avis avec une ou plusieurs opinions positives sur le prix, et aucune opinion négative sur le même aspect.  
    La valeur “Négative” concerne un avis avec une ou plusieurs opinions négatives sur le prix, et aucune opinion positive sur le même aspect.  
    La valeur “Neutre” concerne un avis avec au moins une opinion positive et une opinion négative sur le prix.  
    La valeur "NE" (qui signifie Non Exprimé) fait référence à un avis qui ne contient aucune opinion exprimée sur le prix.

    La réponse doit se limiter à la valeur de l'opinion uniquement.  
    """