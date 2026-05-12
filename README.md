# Seattle Collision Risk Dashboard

## Présentation du projet

Ce projet présente un tableau de bord interactif de science des données appliqué à l’analyse du risque de collision routière à Seattle.

L’objectif est d’utiliser des techniques d’intelligence artificielle et d’apprentissage automatique pour comparer plusieurs modèles de classification, sélectionner automatiquement le modèle le plus performant et présenter les résultats dans une interface web accessible.

Le dashboard permet d’explorer différents scénarios routiers à partir de variables comme la météo, l’état de la route, la luminosité, le type d’adresse, le jour de la semaine et la localisation. Les prédictions affichées dans l’interface proviennent d’un fichier JSON généré par un script Python après l’entraînement des modèles.

Site publié avec GitHub Pages :

https://aidatamodels.github.io/seattle-collision-risk-dashboard/

## Objectif

Le projet vise à démontrer un flux complet de science des données, depuis la préparation des données jusqu’à la visualisation des résultats dans une application web.

Les objectifs principaux sont :

- analyser des données historiques de collisions routières ;
- préparer un jeu analytique exploitable pour l’apprentissage automatique ;
- entraîner et comparer plusieurs modèles de classification ;
- sélectionner automatiquement le meilleur modèle selon une métrique de performance ;
- générer un fichier JSON contenant les résultats nécessaires au dashboard ;
- publier une interface web interactive avec GitHub Pages.

## Contexte

Les collisions routières sont influencées par plusieurs facteurs : conditions météorologiques, état de la chaussée, luminosité, type d’intersection, période de l’année et localisation.

Dans ce projet, ces informations sont utilisées pour estimer le risque associé à différents scénarios. L’approche repose sur des méthodes de classification supervisée, un domaine de l’intelligence artificielle utilisé pour apprendre des relations à partir de données historiques.

Le projet ne vise pas à produire un outil opérationnel de sécurité routière. Il s’agit d’un projet démonstratif permettant d’illustrer une démarche de science des données complète et reproductible.

## Données utilisées

Le projet utilise le jeu de données `Data-Collisions.csv`, associé aux collisions routières de Seattle.

Le script Python prépare les données afin de construire un jeu analytique intégrant notamment :

- la localisation ;
- le jour de la semaine ;
- la saison ;
- la météo ;
- l’état de la route ;
- la luminosité ;
- le type d’adresse ;
- le type de jonction ;
- le mois ;
- l’indicateur de fin de semaine ;
- le jour de l’année ;
- des variables cycliques liées à la saisonnalité.

Le projet distingue trois classes principales :

| Classe | Description |
|---|---|
| 0 | Aucune collision observée |
| 1 | Collision avec dommages matériels |
| 2 | Collision avec blessures |

## Modèles comparés

Le script Python entraîne et compare trois modèles de classification :

| Modèle | Description |
|---|---|
| Régression logistique équilibrée | Modèle linéaire utilisé comme référence de comparaison |
| Forêt aléatoire | Modèle d’ensemble basé sur plusieurs arbres de décision |
| Extra Trees | Modèle d’ensemble utilisant des arbres extrêmement randomisés |

Le meilleur modèle est sélectionné automatiquement à partir d’une métrique de performance, notamment le `f1_macro`, qui permet d’évaluer la qualité globale du modèle sur plusieurs classes.

Dans l’exécution actuelle, le modèle retenu est :

```text
Forêt aléatoire
