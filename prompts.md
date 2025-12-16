1e tentative :
    """Considérez l'avis suivant:

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
    { "Prix": opinion, "Cuisine": opinion, "Service": opinion}."""

Résultats :
    - gemma2:2b -> AVG MACRO ACC: 84 (sur 50 tests) -> 83.39 (tous les tests) 
    - gemma3:1b -> AVG MACRO ACC: 58.0 (sur 50 tests)
    - llama3.2:1b -> AVG MACRO ACC: 27.33 (sur 50 tests)
    - qwen2.5:1.5b -> AVG MACRO ACC: 51.33 (sur 50 tests)
    - qwen3:1.7b -> AVG MACRO ACC: 40.0 (sur 50 tests)


################################################################################################
################################################################################################
################################################################################################

1e tentative :


Résultats :
    - gemma2:2b -> AVG MACRO ACC: 84 (sur 50 tests) -> 83.39 (tous les tests) 
    - gemma3:1b -> AVG MACRO ACC: 58.0 (sur 50 tests)
    - llama3.2:1b -> AVG MACRO ACC: 27.33 (sur 50 tests)
    - qwen2.5:1.5b -> AVG MACRO ACC: 51.33 (sur 50 tests)
    - qwen3:1.7b -> AVG MACRO ACC: 40.0 (sur 50 tests)

