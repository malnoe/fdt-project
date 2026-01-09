Garance MALNOË & Matthias MAZET
M2 SSD


________________________________________________________________________________


##### Introduction #####

L'objectif de ce projet est de construire un classifieur capable de déterminer l'opinion exprimée sur trois aspects (le prix, la cuisine et le service) dans des avis de clients de restaurants. L'opinion sur chaque aspect est référencée parmi l'une des quatre catégories suivantes : Positive, Négative, Neutre ou Non Exprimée (NE).
Pour cela, nous avons construit et testé deux approches différentes de classifieur : une approche utilisant un Large Language Model (LLM) sans entraînement, et une approche par Fine-Tuning d'un modèle déjà pré-entrainé (PLMFT).
La démarche globale ainsi que les résultats obtenus et le modèle retenu pour chaque approche sont détaillés dans les sections suivantes.
Le modèle final retenu pour le projet est détaillé dans la conclusion.


________________________________________________________________________________


##### Approche par LLM sans entraînement #####

Notre approche s'est construite en plusieurs étapes :
    
* Étape 1. Tests de différents prompts sur les différents modèles de LLMs.
	Nous avons rédigé différents prompts pour essayer d'augmenter les performances en testant plusieurs angles.
	Pour chaque prompt, nous avons testé l'ensemble des LLM autorisés par le cadre du projet (gemma2:2b, gemma3:1b, llama3.2:1b, qwen2.5:1.5b et qwen3:1.7b) avec les mêmes hyperparamètres (temp=0.1 et top_p=0.9) afin de comparer les performances en termes d'accuracy moyenne globale et de temps d'exécution.

	Les prompts testés étaient les suivants :
        	a. Un prompt basique détaillant les différents aspects à noter (Cuisine, Prix et Service) et les différents grades de notations (Positive, Neutre, Négative et Non Exprimée).
        	b. Une version anglaise du prompt précédent, avec les tags en français pour conserver la cohérence de sortie.
		c. Une version plus détaillée du premier prompt, avec notamment l'ajout de notions de références et de quelques mots exemples pour chaque aspect (e.g. : "L'aspect "Prix" fait référence aux prix des plats et boissons. Il concerne tout ce qui parle d'argent, de coût, de tarifs, etc., avec un vocabulaire tel que "trop cher", "prix", "rapport qualité/prix", etc.").
        	d. Une version anglaise du prompt précédent.
        	e. Une version du premier prompt où nous demandions au LLM d'expliquer sa démarche (certaines études ont montré que les performances d'un LLM augmentaient simplement grâce à cette demande).
        	f. Une version du premier prompt avec l'ajout d'exemples de classifications d'opinions.
        	g. Une combinaison de plusieurs approches, i.e. un prompt plus détaillé avec l'ajout d'exemples de classification d'opinions. Nous n'avons pas ajouté l'explication de la démarche car le temps de calcul était trop long par rapport à l'efficacité du prompt.

	Cette première étape nous a permis de sélectionner une première combinaison prompt/LLM à retenir pour les étapes suivantes.
	À quelques exceptions près, le LLM avec les meilleures performances en termes d'accuracy moyenne et de temps de calcul était toujours le gemma2:2b, et la meilleure accuracy moyenne (83,61% pour 686s de prédiction) avec ce LLM a été obtenue avec le prompt basique en anglais. 
	Nous avons pu constater que l'ajout d'exemples détériorait significativement l'accuracy moyenne, tandis que l'explication de la démarche augmentait grandement le temps de calcul. L'ajout de détails n'a eu que peu d'impact sur l'accuracy moyenne mais augmentait grandement le temps d'exécution, c'est pourquoi nous ne l'avons pas conservé.


* Étape 2. Recherche des meilleurs hyperparamètres.
	Une fois la meilleure combinaison prompt/LLM retenue (prompt basique en anglais avec gemma2:2b), nous avons testé différentes combinaisons de valeurs pour les hyperparamètres de température `temp` et `top_p` afin d'observer leur impact sur l'accuracy moyenne et le temps d'exécution (temp = [0.1,0.5,0.8] et top_p = [0.3,0.5,0.9]). Nous n'avons pas testé plus de valeurs pour la température et la top_p car nous n'avons observé que des variations minimales (<0.1%) de l'accuracy moyenne entre ces différentes valeurs.
	Les variations de température n'ont pas eu d'impact significatif sur les performances mais nous avons noté un léger intérêt à diminuer la top_p pour augmenter l'accuracy moyenne (81,83% au moins => 83,78% au plus).
	Les meilleures performances (accuracy moyenne = 83,78, temps d'exécution de 601 secondes) ont été obtenues avec une température de 0.1 et une top_p de 0.5, ces paramètres ont donc été retenus.


* Étape 3. Vote de majorité après répétition de plusieurs appels d'un même prompt.
	Nous avons ensuite testé une approche de vote à la majorité en répétant trois fois l'appel d'un même LLM avec le même prompt et une température élevée (temp=0.7 pour les meilleures performances et top_p=0.5) afin peut-être d'obtenir une meilleure précision. 
	Nous avons de nouveau utilisé la combinaison prompt en anglais / gemma2:2b retenue à l'étape 1.
	N'ayant pas constaté de réel intérêt à cette approche (accuracy moyenne = 83,67 au mieux), nous avons donc décidé de ne pas la retenir, notamment en raison de temps d'exécution plutôt conséquents (moyenne = 2062 secondes contre environ 600 secondes) liés à la répétition des appels LLM pour une seule prédiction.


* Étape 4. Un prompt par aspect.
	La dernière étape tentée pour cette approche a été la construction de prompt individuel pour chaque aspect.
	Nous avons testé des prompts basiques en français et en anglais pour chaque aspect (même forme que précédemment mais redécoupé pour chaque aspect), dont nous avons ensuite combiné les prédictions afin d'obtenir le format final souhaité. 
	Cette approche n'ayant pas donné de bons résultats en termes d'accuracy moyenne (accuracy moyenne max = 68,11%), nous n'avons pas poussé la démarche plus loin et avons décidé de ne pas retenir cette approche.


Pour l'approche par LLM, nous avons donc finalement retenu le LLM gemma2:2b avec le prompt en anglais (prompt n°3 en Annexe) ci-dessous, avec une température de 0.1 et une top_p de 0.5. Nous avons obtenu une accuracy moyenne de 83.78 pour un temps d'exécution de 601 secondes.

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
    {"Prix": opinion, "Cuisine": opinion, "Service": opinion}."""

L'ensemble des autres prompts testés sont disponibles en annexe de ce document.


________________________________________________________________________________


##### Approche par PLMFT #####

Pour cette approche, nous avons testés plusieurs modèles masqués pré-entrainés (au moins en partie) sur des données en français, qui sont disponibles sur Hugging Face et qui comprennent moins de 300 millions de paramètres : 
	- google-bert/bert-base-multilingual-cased
	- almanach/camembert-base
	- almanach/moderncamembert-base
	- FacebookAI/xlm-roberta-base
	- cmarkea/distilcamembert-base

L'ensemble de ces modèles ont été fine-tunés sur les jeux d'entraînement et de validation fournis pour la tâche de classification des avis en 4 classes (Positive, Neutre, Négative, NE) pour chacun des aspects (Prix, Cuisine, Service) avec 3 epochs et les mêmes hyperparamètres. Pour chaque modèle, les résultats obtenus sur le jeu de validation sont recensés dans le tableau suivant :

| Modèle                                   | Accuracy moyenne (%) | Temps d'entrainement (s) | Temps de prédiction (s) |
|------------------------------------------|----------------------|--------------------------|-------------------------|
| google-bert/bert-base-multilingual-cased | 81,56                | 7970                     | 131                     |
| almanach/camembert-base                  | 81,50                | 5144                     | 75                      |
| almanach/moderncamembert-base            | 83,95                | 5704                     | 107                     |
| FacebookAI/xlm-roberta-base              | 83,11                | 7520                     | 80                      |
| cmarkea/distilcamembert-base             | 84,56                | 3516                     | 53                      |


Nous avons ensuite testé les trois modèles ayant obtenu les meilleurs résultats sur le jeu de validation (moderncamembert-base, xlm-roberta-base et distilcamembert-base) sur 5 epochs (avec toujours les mêmes hyperparamètres) afin de voir si les performances pouvaient être encore améliorées. Les résultats obtenus sont recensés dans le tableau suivant :

| Modèle                                   | Accuracy moyenne (%) | Temps d'entrainement (s) | Temps de prédiction (s) |
|------------------------------------------|----------------------|--------------------------|-------------------------|
| almanach/moderncamembert-base            | 84,17                | 10300                    | 107                     |
| FacebookAI/xlm-roberta-base              | 84,34                | 11300                    | 80                      |
| cmarkea/distilcamembert-base             | 85,94                | 5730                     | 53                      |


Le modèle retenu à la fin de cette étape est donc le cmarkea/distilcamembert-base fine-tuné. 
Pour compléter cette approche, nous avons  finalement entraîné ce modèle sur 10 epochs plusieurs fois pour choisir à quelle epoch s'arrêter. À chaque fois, c'est l'epoch 5 qui a été retenu, avec une accuracy moyenne de 86.39, 86,28 et 85,94 respectivement pour les trois essais. Les epochs suivantes présentent un loss plus important sur le jeu de validation.

Pour l'approche par PLMFT, le modèle retenu est donc celui entrainé sur cmarkea/distilcamembert-base fine-tuné sur 5 epochs, avec lequel nous obtenons une accuracy moyenne de 86,20%.

Il est important de noter que les entraînements ont été réalisés sur CPU (GPU non disponible) et que les temps pourraient être significativement réduits avec l'utilisation d'un GPU. 
Ce temps de calcul très important nous a limités dans le nombre d'expérimentations que nous avons pu faire sur cette approche.


________________________________________________________________________________


##### Conclusion #####

Au vu des résultats, le choix entre les deux approches dépendra du contexte d'utilisation du classifieur : 
	- L'approche par LLM sans entraînement est pertinente si nous souhaitons faire peu de prédictions grâce à l'absence de temps d'entraînement, le temps de prédiction restant raisonnable et une accuracy moyenne correcte assez proche de l'approche par PLMFT (83,78% contre 86,20%).
	- L'approche par PLMFT est plus adaptée si nous souhaitons réaliser beaucoup de prédictions et que nous avons accès à un GPU permettant de réduire le temps d'entraînement, ce qui permet d'obtenir une accuracy moyenne plus élevée (86,20%).

Une plus grande importance étant accordée à l'accuracy moyenne et l'utilisation possible d'un GPU nous ont finalement poussés à retenir l'approche par PLMFT pour ce projet.

* Description du modèle final retenu :
	Le classifieur retenu repose sur un modèle de langue transformer pré-entraîné (cmarkea/distilcamembert-base) fine-tuné. L'architecture du modèle est basée sur un encodeur partagé composé de trois têtes de classification (une par aspect), chacune composée d'une couche linéaire suivie d'une fonction d'activation softmax afin d'obtenir une distribution de probabilité sur les quatre classes à prédire (Positive, Négative, Neutre, NE).
Le modèle a été fine-tuné sur les jeux d'entraînement et de validation fournis dans le projet, avec les hyperparamètres suivants :
	- 5 epochs (avec sauvegarde du modèle à chaque epoch).
	- Une longueur maximale des séquences de 256 tokens.
	- Un batch size de 16 pour l'entraînement et de 32 pour la validation.
	- Un taux d'apprentissage de 2*10^(-5).
	- Un optimiseur AdamW.
	- Un drop-out de 0.1 (warmup).
	- Une weight decay de 0.01.
	- Une fonction de perte d'entropie croisée (cross-entropy).

Le modèle final atteint une accuracy moyenne de 86,20% sur le jeu de test.

________________________________________________________________________________
________________________________________________________________________________


##### Annexe #####

* Prompt 1 :
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

-----

* Prompt 2 (version légèrement modifié du Prompt 1) :
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

-----

* Prompt 3 (Version en anglais du Prompt 1, avec tags en français) :
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

-----

* Prompt 4 (version plus détaillée du Prompt 1) :  
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

-----

* Prompt 5 (Prompt 1 avec explication de la démarche) :
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

-----

* Prompt 6 (Prompt 1 avec exemple) :  
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

-----

* Prompt 7 (Mix 1 (sans explications du raisonnement car trop long)) :
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

-----

* Prompt 8 (Mix 2 (anglais + détails)) : 
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

-----

* Prompt Prix 1 : 
	"""Considere the following review:

	"{{text}}"  
	What is the meaning of the opinion expressed the "Prix" aspect ? The "Prix" aspect refers to the prices of food and beverages.

	For each aspect, the meaning of the opinion must be one of the following values: "Positive", "Négative", "Neutre", or "NE".  
	The value “Positive” concerns a review with one or more positive opinions on the aspect in question, and no negative opinions on the same aspect.  
	The value “Négative” concerns a review with one or more negative opinions on the aspect in question, and no positive opinions on the same aspect.  
	The value “Neutre” concerns a review with at least one positive opinion and one negative opinion on the same aspect.  
	The value "NE" (which stands for Non Expressed) refers to a review that does not contain any opinion expressed on the aspect in question.

	The response must be limited to the opinion value only."""  

-----

* Prompt Cuisine 1 :*
	"""Considere the following review:

	"{{text}}"  
	What is the meaning of the opinion expressed the "Service" aspect ? The “Service” aspect refers to the quality and efficiency of the service and reception.

	For each aspect, the meaning of the opinion must be one of the following values: "Positive", "Négative", "Neutre", or "NE".  
	The value “Positive” concerns a review with one or more positive opinions on the aspect in question, and no negative opinions on the same aspect.  
	The value “Négative” concerns a review with one or more negative opinions on the aspect in question, and no positive opinions on the same aspect.  
	The value “Neutre” concerns a review with at least one positive opinion and one negative opinion on the same aspect.  
	The value "NE" (which stands for Non Expressed) refers to a review that does not contain any opinion expressed on the aspect in question.

	The response must be limited to the opinion value only."""

-----

* Prompt Service 1 :
	"""Considere the following review:

	"{{text}}"  
	What is the meaning of the opinion expressed the "Service" aspect ? The “Service” aspect refers to the quality and efficiency of the service and reception.

	For each aspect, the meaning of the opinion must be one of the following values: "Positive", "Négative", "Neutre", or "NE".  
	The value “Positive” concerns a review with one or more positive opinions on the aspect in question, and no negative opinions on the same aspect.  
	The value “Négative” concerns a review with one or more negative opinions on the aspect in question, and no positive opinions on the same aspect.  
	The value “Neutre” concerns a review with at least one positive opinion and one negative opinion on the same aspect.  
	The value "NE" (which stands for Non Expressed) refers to a review that does not contain any opinion expressed on the aspect in question.

	The response must be limited to the opinion value only."""

-----

* Prompt Cuisine 2 :
	"""Considère l'avis suivant :

	"{{text}}"  
	Quel est l'opinion exprimée pour la cuisine ? Le client est-il satisfait ou insatisfait de la qualité et de la quantité des plats et boissons proposées ?  
      
	L'opinion pour la cuisine doit être l'une des valeurs suivantes : "Positive", "Négative", "Neutre", ou "NE".  
	La valeur “Positive” concerne un avis avec une ou plusieurs opinions positives sur la cuisine, et aucune opinion négative sur le même aspect.  
	La valeur “Négative” concerne un avis avec une ou plusieurs opinions négatives sur la cuisine, et aucune opinion positive sur le même aspect.  
	La valeur “Neutre” concerne un avis avec au moins une opinion positive et une opinion négative sur la cuisine.  
	La valeur "NE" (qui signifie Non Exprimé) fait référence à un avis qui ne contient aucune opinion exprimée sur la cuisine.  
      
	La réponse doit se limiter à la valeur de l'opinion uniquement."""  

-----

* Prompt Service 2 :
	"""Considère l'avis suivant :

	"{{text}}"  
	Quel est l'opinion exprimée pour le service ? Le client est-il satisfait ou insatisfait du service et de l'accueil que ce soit la qualité ou l'efficacité ?  
      
	L'opinion pour le service doit être l'une des valeurs suivantes : "Positive", "Négative", "Neutre", ou "NE".  
	La valeur “Positive” concerne un avis avec une ou plusieurs opinions positives sur le service, et aucune opinion négative sur le même aspect.  
	La valeur “Négative” concerne un avis avec une ou plusieurs opinions négatives sur le service, et aucune opinion positive sur le même aspect.  
	La valeur “Neutre” concerne un avis avec au moins une opinion positive et une opinion négative sur le service.  
	La valeur "NE" (qui signifie Non Exprimé) fait référence à un avis qui ne contient aucune opinion exprimée sur le service.  
      
	La réponse doit se limiter à la valeur de l'opinion uniquement."""  

-----

* Prompt Prix 2 :
	"""Considère l'avis suivant :

	"{{text}}"  
	Quel est l'opinion exprimée pour le prix ? Le client est-il satisfait ou insatisfait des prix proposé ou du rapport qualité/prix ?  
      
	L'opinion pour le prix doit être l'une des valeurs suivantes : "Positive", "Négative", "Neutre", ou "NE".  
	La valeur “Positive” concerne un avis avec une ou plusieurs opinions positives sur le prix, et aucune opinion négative sur le même aspect.  
	La valeur “Négative” concerne un avis avec une ou plusieurs opinions négatives sur le prix, et aucune opinion positive sur le même aspect.  
	La valeur “Neutre” concerne un avis avec au moins une opinion positive et une opinion négative sur le prix.  
	La valeur "NE" (qui signifie Non Exprimé) fait référence à un avis qui ne contient aucune opinion exprimée sur le prix.

	La réponse doit se limiter à la valeur de l'opinion uniquement."""


________________________________________________________________________________


##### Utilisation de l'IA #####

Dans le cadre de ce projet, nous avons utilisé l'IA pour nous aider à comprendre comment construire un classifieur fine-tuné avec plusieurs classes avec les librairies torch et transformers et pour écrire du code, notamment pour l'implémentation du classifieur PLMFT et certaines fonctions de parse du classifieur LLM. 
Les modèles utilisés sont ChatGPT (GPT-5.1) et Copilot.
