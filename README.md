# Seattle Collision Risk Dashboard

## Présentation du projet

Seattle Collision Risk Dashboard est un projet de science des données et d’intelligence artificielle appliqué à l’analyse du risque de collision routière à Seattle.

Le projet utilise des données historiques de collisions afin de construire un jeu analytique, entraîner plusieurs modèles de classification supervisée, comparer leurs performances et présenter les résultats dans un tableau de bord web interactif.

L’objectif est de démontrer un flux complet de projet en science des données, depuis la préparation des données jusqu’à la publication d’une interface web avec GitHub Pages.

## Dashboard en ligne

Le dashboard est publié avec GitHub Pages et peut être consulté ici :

[Ouvrir le Seattle Collision Risk Dashboard](https://aidatamodels.github.io/seattle-collision-risk-dashboard/)

## Objectif

Ce projet vise à estimer le risque associé à différents scénarios routiers à partir de variables contextuelles telles que :

- la localisation ;
- la météo ;
- l’état de la route ;
- la luminosité ;
- le type d’adresse ;
- le type de jonction ;
- le jour de la semaine ;
- la saison ;
- le mois ;
- l’indicateur de fin de semaine.

Le dashboard permet d’explorer les résultats produits par le meilleur modèle entraîné en Python.

## Contexte

Les collisions routières ne dépendent pas d’un seul facteur. Elles peuvent être influencées par les conditions météorologiques, la visibilité, l’état de la chaussée, le type d’intersection, la période de l’année et les caractéristiques propres à certaines localisations.

Dans ce projet, ces éléments sont utilisés comme variables explicatives dans une approche d’apprentissage automatique supervisé. Le but n’est pas de prédire avec certitude qu’un accident va se produire, mais d’estimer statistiquement le niveau de risque associé à un scénario donné.

Ce projet est donc une démonstration méthodologique de science des données, d’intelligence artificielle appliquée et de visualisation interactive.

## Mise en situation

Le dashboard peut être utilisé pour simuler un scénario routier précis.

Par exemple, l’utilisateur peut sélectionner les conditions suivantes :

- Jour : lundi ;
- Météo : temps clair ;
- État de la route : chaussée sèche ;
- Luminosité : lumière du jour ;
- Type d’adresse : intersection ;
- Type de jonction : intersection ;
- Localisation : 6TH AVE AND JAMES ST.

La question analysée par le modèle devient alors :

> Si une journée de type lundi présente ces conditions — temps clair, route sèche, lumière du jour, intersection et localisation donnée — quel serait le risque estimé de collision selon les tendances observées dans les données historiques ?

Le dashboard retourne une estimation statistique du risque associé à ce scénario. Il indique notamment la classe prédite, la distribution des probabilités entre les classes et la probabilité de blessures en cas de collision.

Il ne s’agit pas d’une prédiction en temps réel pour une date future précise. Le modèle ne connaît pas les conditions réelles de la semaine prochaine, comme la météo exacte, le trafic, les travaux, les événements ou les comportements des conducteurs. Il estime plutôt le niveau de risque d’un scénario défini par l’utilisateur, à partir des données historiques disponibles.

## Sources des données

Le jeu de données utilisé correspond au jeu de données annuel des collisions routières du Seattle Department of Transportation.

Il regroupe des événements enregistrés par la Ville de Seattle et décrit, pour chaque collision, la gravité observée ainsi que plusieurs informations contextuelles utiles à l’analyse, notamment la localisation, les conditions météorologiques, l’état de la route, la luminosité, le type d’adresse et le type de jonction.

Dans ce projet, ces données sont utilisées afin de construire un jeu analytique destiné à l’entraînement, à la comparaison et à l’évaluation de modèles de classification supervisée.

Pour plus de détails sur le jeu de données, consulter le document de référence :_ [Data Set Summary](https://www.seattle.gov/Documents/Departments/SDOT/GIS/Collisions_OD.pdf).

## Variables utilisées

Les principales variables utilisées dans le modèle sont les suivantes :

| Variable | Description |
|---|---|
| `LOCATION` | Localisation associée au scénario |
| `jour_semaine` | Jour de la semaine |
| `saison` | Saison associée à la date |
| `WEATHER` | Conditions météorologiques |
| `ROADCOND` | État de la route |
| `LIGHTCOND` | Conditions de luminosité |
| `ADDRTYPE` | Type d’adresse ou de lieu |
| `JUNCTIONTYPE` | Type de jonction |
| `mois` | Mois du scénario |
| `weekend` | Indicateur de fin de semaine |
| `jour_annee` | Jour de l’année |
| `sin_jour_annee` | Variable cyclique liée à la saisonnalité |
| `cos_jour_annee` | Variable cyclique liée à la saisonnalité |

## Variable cible

Le problème est formulé comme une classification multiclasse.

Le jeu de données brut contient principalement des collisions observées. Afin de permettre au modèle d’estimer non seulement la gravité d’une collision, mais aussi la probabilité qu’aucune collision ne soit observée dans un scénario donné, le script construit un jeu analytique à partir des données historiques.

Cette construction permet d’intégrer des scénarios sans collision observée, en combinant des localisations, des dates et des conditions contextuelles. Ces scénarios sont représentés par la classe `0`.

Les classes utilisées sont les suivantes :

| Classe | Description |
|---|---|
| 0 | Aucune collision observée |
| 1 | Collision avec dommages matériels |
| 2 | Collision avec blessures |

Cette structure permet de distinguer les situations sans collision, les collisions avec dommages matériels et les collisions impliquant des blessures.

## Approche méthodologique

Le projet suit un flux classique de science des données.

### 1. Chargement des données

Le script `build_dashboard_data.py` charge le fichier `Data-Collisions.csv`.

Si le fichier n’est pas disponible localement, le script peut tenter de le récupérer depuis une source distante.

### 2. Nettoyage et préparation

Les données sont filtrées, nettoyées et transformées afin de conserver les colonnes utiles pour la modélisation.

Cette étape prépare les variables catégorielles, les variables numériques et les informations temporelles.

### 3. Construction du jeu analytique

Le script construit un jeu analytique combinant les localisations et les dates.

Cette étape permet d’intégrer des scénarios sans collision observée, afin de ne pas travailler uniquement sur les événements où une collision a déjà eu lieu.

### 4. Ingénierie des variables

Des variables supplémentaires sont créées pour enrichir l’analyse :

- le mois ;
- le jour de la semaine ;
- la saison ;
- l’indicateur de fin de semaine ;
- le jour de l’année ;
- deux variables cycliques basées sur le sinus et le cosinus du jour de l’année.

Ces variables permettent au modèle de mieux tenir compte des effets temporels et saisonniers.

### 5. Entraînement des modèles

Trois modèles de classification sont entraînés :

| Modèle | Rôle |
|---|---|
| Régression logistique équilibrée | Modèle de référence simple et interprétable |
| Forêt aléatoire | Modèle d’ensemble basé sur plusieurs arbres de décision |
| Extra Trees | Modèle d’ensemble avec arbres extrêmement randomisés |

### 6. Comparaison des modèles

Les modèles sont comparés à partir de métriques de classification.

La métrique principale utilisée pour la sélection du meilleur modèle est le `f1_macro`.

Cette métrique est pertinente dans un problème multiclasse, car elle permet d’évaluer la performance globale du modèle sans se limiter à la classe majoritaire.

### 7. Sélection du meilleur modèle

Le script sélectionne automatiquement le modèle le plus performant.

Dans l’exécution actuelle du projet, le modèle retenu est :

```text
Forêt aléatoire
```

### 8. Génération du fichier JSON

Après l’entraînement et la sélection du meilleur modèle, le script génère le fichier :

```text
dashboard_data.json
```

Ce fichier contient les résultats utilisés par le dashboard web.

### 9. Visualisation dans le dashboard

Le fichier `index.html` lit le fichier `dashboard_data.json` et affiche les résultats dans une interface web interactive.

Le navigateur ne réentraîne pas les modèles. Toute la partie science des données est exécutée en amont par Python.

## Fonctionnement général

Le fonctionnement du projet peut être résumé ainsi :

```text
Données historiques
        ↓
Préparation des données avec Python
        ↓
Création du jeu analytique
        ↓
Entraînement de trois modèles
        ↓
Comparaison des performances
        ↓
Sélection du meilleur modèle
        ↓
Génération de dashboard_data.json
        ↓
Lecture du JSON par index.html
        ↓
Affichage des résultats dans le dashboard
```

## Résultats de l’exécution actuelle

Lors de l’exécution actuelle du script :

```text
Modèles comparés :
- Régression logistique équilibrée
- Forêt aléatoire
- Extra Trees

Modèle retenu :
- Forêt aléatoire

Fichier généré :
- dashboard_data.json
```

Le dashboard affiche les prédictions issues du meilleur modèle retenu par le script Python.

## Architecture du projet

```text
seattle-collision-risk-dashboard/
│
├── index.html
├── build_dashboard_data.py
├── dashboard_data.json
├── requirements.txt
├── README.md
└── LICENSE
```

## Description des fichiers

| Fichier | Description |
|---|---|
| `index.html` | Interface web du dashboard |
| `build_dashboard_data.py` | Script Python de préparation des données, entraînement des modèles et génération du JSON |
| `dashboard_data.json` | Fichier généré par Python et lu par le dashboard |
| `requirements.txt` | Dépendances Python nécessaires |
| `README.md` | Documentation du projet |
| `LICENSE` | Licence MIT du projet |

## Technologies utilisées

Le projet utilise les technologies suivantes :

- Python ;
- Pandas ;
- NumPy ;
- Scikit-learn ;
- HTML5 ;
- CSS3 ;
- JavaScript ;
- Git ;
- GitHub ;
- GitHub Pages.

## Intelligence artificielle et apprentissage automatique

Ce projet utilise des modèles d’apprentissage automatique supervisé, un domaine de l’intelligence artificielle.

Les modèles sont entraînés à partir de données historiques afin d’identifier des relations entre les conditions observées et les classes de collision.

La forêt aléatoire, modèle retenu dans l’exécution actuelle, combine plusieurs arbres de décision afin de produire une prédiction plus robuste qu’un arbre isolé.

## Fonctionnalités du dashboard

Le dashboard permet de visualiser :

- le scénario sélectionné ;
- la localisation analysée ;
- le risque estimé de collision ;
- la classe prédite ;
- la probabilité associée à chaque classe ;
- la probabilité de blessure en cas de collision ;
- le Top 5 des localisations les plus exposées pour un scénario donné ;
- l’indication que les résultats proviennent du fichier `dashboard_data.json`.

## Exécution locale

### 1. Cloner le dépôt

```bash
git clone https://github.com/aidatamodels/seattle-collision-risk-dashboard.git
```

### 2. Accéder au dossier du projet

```bash
cd seattle-collision-risk-dashboard
```

### 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

Sur Windows, il est aussi possible d’utiliser :

```bash
py -m pip install -r requirements.txt
```

### 4. Générer le fichier JSON

```bash
python build_dashboard_data.py
```

Sur Windows :

```bash
py build_dashboard_data.py
```

Cette commande génère le fichier :

```text
dashboard_data.json
```

### 5. Lancer un serveur local

```bash
python -m http.server 8000
```

Sur Windows :

```bash
py -m http.server 8000
```

### 6. Ouvrir le dashboard

Dans le navigateur :

```text
http://localhost:8000
```

## Publication avec GitHub Pages

Le projet est publié avec GitHub Pages à partir de la branche `main`.

Configuration utilisée :

```text
Source : Deploy from a branch
Branch : main
Folder : /root
```

URL du dashboard :

```text
https://aidatamodels.github.io/seattle-collision-risk-dashboard/
```

## Remarque sur le fichier JSON

Le fichier `dashboard_data.json` peut être volumineux, car il contient les scénarios pré-calculés utilisés par le dashboard.

Cette approche permet d’avoir un site statique compatible avec GitHub Pages, sans serveur Python en production.

En contrepartie, le temps de chargement peut être plus long si le fichier JSON est lourd.

## Limites du projet

Ce projet doit être interprété comme une démonstration de science des données et d’intelligence artificielle appliquée.

Les principales limites sont les suivantes :

- les résultats dépendent de la qualité et de la structure des données historiques ;
- le dashboard affiche des résultats pré-calculés ;
- aucune API de prédiction dynamique n’est utilisée dans cette version ;
- le fichier JSON peut avoir un impact sur le temps de chargement ;
- les prédictions ne doivent pas être utilisées comme outil officiel de sécurité routière ;
- le modèle ne remplace pas une expertise métier en transport, urbanisme ou sécurité routière.

## Améliorations possibles

Des améliorations futures pourraient inclure :

- réduire la taille du fichier `dashboard_data.json` ;
- ajouter une matrice de confusion ;
- afficher les métriques détaillées de chaque modèle ;
- ajouter des graphiques de comparaison entre les modèles ;
- intégrer une carte interactive des localisations ;
- ajouter une API Python pour les prédictions dynamiques ;
- enrichir l’analyse exploratoire des données ;
- documenter plus en détail les choix de modélisation ;
- ajouter des tests automatisés ;
- améliorer la performance de chargement du dashboard.

## Avertissement

Les résultats présentés dans ce dashboard sont des estimations statistiques fondées sur des données historiques.

Ils ne constituent pas une prédiction certaine et ne doivent pas être utilisés comme outil officiel de sécurité routière, de planification urbaine ou de prise de décision opérationnelle.

Ce projet est fourni à des fins éducatives, démonstratives et méthodologiques.

## Licence

Ce projet est distribué sous licence MIT.

La licence MIT autorise l’utilisation, la modification, la distribution et la réutilisation du code, à condition de conserver l’avis de copyright et le texte de la licence.

Voir le fichier [LICENSE](LICENSE) pour plus de détails.

Cette licence s’applique au code source du projet. Les données externes utilisées, notamment le fichier `Data-Collisions.csv`, restent soumises aux conditions d’utilisation de leur source d’origine.

## Auteur

Projet développé par `aidatamodels` dans le cadre d’une démarche de science des données, d’intelligence artificielle appliquée et de visualisation interactive de modèles de classification.
