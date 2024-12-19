# Plateforme-de-detection-des-anomalies-du-sommeil/ML.

![IntroGitHub](https://github.com/user-attachments/assets/cf0b05f9-5018-45fb-995c-2ad4218c13e9)

# I. Introduction

Les troubles du sommeil constituent un problème de santé publique majeur, touchant
une grande partie de la population mondiale et entraînant des complications graves
comme l'anxiété, la dépression et les maladies cardiovasculaires. Il est crucial de
comprendre les facteurs influençant la qualité du sommeil pour développer des
interventions efficaces. L'analyse de données permet d'identifier les relations entre les
habitudes de vie, les comportements et les caractéristiques démographiques. En
analysant le dataset "Sleep_health_and_lifestyle", l'étude utilise des techniques de
clustering et d'analyse factorielle pour modéliser ces relations.
L'objectif est de fournir des insights pour améliorer la santé des personnes touchées par ces
troubles.

# I.1. Présentation du Jeu de Données

Nous explorons les données provenant d'une enquête auprès de 400 individus dans un jeu de
données appelé "Sleep_health_and_lifestyle" qui est composé d'un ensemble
d'observations collectées auprès d'individus concernant divers aspects de leur santé et de leur
mode de vie liés au sommeil.Cette enquête recueille des informations sur divers aspects de la
santé et du mode de vie de chaque individu.

# I.2. La variable cible (Target)

Dans notre étude, la variable cible est le trouble du sommeil (Sleep Disorder), Cette variable catégorielle nous permet de classifier les
individus en fonction des troubles du sommeil qu'ils peuvent avoir.
La variable Sleep Disorder comporte trois classes :

   ● None : Absence de trouble du sommeil.
   
   ● Sleep Apnea : Présence d'apnée du sommeil.
   
   ● Insomnia : Présence d'insomnie.
   
Ces classes nous aident à identifier les types de troubles du sommeil présents chez les
individus et à prédire ces troubles en fonction des autres variables explicatives, telles que les
habitudes de vie et les caractéristiques médicales.

# I.3. Les variables qui participe à la classification(les Inputs)
Nous utilisons dans ce jeu de données, plusieurs variables explicatives. Ces variables nous
permettent de modéliser et de mieux comprendre les facteurs qui influencent les troubles du
sommeil. Voici une description des principales variables utilisées excepté la variable

# Description des Variables du Dataset

| **Nom de la Variable**                  | **Type**                 | **Rôle**                                                                                      |
|-----------------------------------------|--------------------------|----------------------------------------------------------------------------------------------|
| **Gender (Genre)**                      | Catégorielle (Nominale)  | Indique le sexe de l'individu : Male (Homme) ou Female (Femme).                              |
| **Age (Âge)**                           | Numérique (Discrète)     | Indique l'âge de l'individu en années.                                                       |
| **Occupation (Profession)**             | Catégorielle (Nominale)  | Indique la profession ou l'emploi de l'individu.                                             |
| **Sleep Duration (Durée du Sommeil)**    | Numérique (Continue)     | Indique la durée du sommeil de l'individu en heures.                                         |
| **Quality of Sleep (Qualité du Sommeil)**| Numérique (Discrète)     | Évalue la qualité du sommeil de l'individu sur une échelle de 1 à 10.                        |
| **Physical Activity Level (Niveau d'Activité Physique)** | Numérique (Discrète) | Indique le niveau d'activité physique de l'individu sur une échelle déterminée.             |
| **Stress Level (Niveau de Stress)**      | Numérique (Discrète)     | Indique le niveau de stress de l'individu, mesuré sur une échelle donnée.                    |
| **BMI Category (Catégorie IMC)**         | Catégorielle (Nominale)  | Classe l'individu en fonction de son IMC : "Sous-poids", "Normal", "Surpoids", ou "Obésité". |
| **Pressure Systolic (Pression Artérielle Systolique)** | Numérique (Continue) | Indique la pression artérielle systolique de l'individu en mmHg.                             |
| **Pressure Diastolic (Pression Artérielle Diastolique)** | Numérique (Continue) | Indique la pression artérielle diastolique de l'individu en mmHg.                            |
| **Heart Rate (Rythme Cardiaque)**        | Numérique (Continue)     | Indique le rythme cardiaque de l'individu, mesuré en battements par minute.                  |
| **Daily Steps (Nombre de Pas Quotidiens)**| Numérique (Discrète)     | Indique le nombre de pas effectués quotidiennement par l'individu.                           |


# I.4. Statistiques descriptives
cette analyse résume les principales caractéristiques du dataset et fournit un aperçu des données utilisées pour les analyses et la modélisation des troubles du sommeil:

## Statistiques descriptives des données

| **Attribut**              | **count** | **mean**   | **std**    | **min**  | **25%**   | **50%**   | **75%**   | **max**    |
|----------------------------|-----------|------------|------------|----------|-----------|-----------|-----------|------------|
| **Age**                   | 374.000   | 42.184     | 8.673      | 27.000   | 35.250    | 43.000    | 50.000    | 59.000     |
| **Sleep Duration**        | 374.000   | 7.132      | 0.796      | 5.800    | 6.400     | 7.200     | 7.800     | 8.500      |
| **Quality of Sleep**      | 374.000   | 7.313      | 1.197      | 4.000    | 6.000     | 7.000     | 8.000     | 9.000      |
| **Physical Activity Level** | 374.000 | 59.171     | 20.831     | 30.000   | 45.000    | 60.000    | 75.000    | 90.000     |
| **Stress Level**          | 374.000   | 5.385      | 1.775      | 3.000    | 4.000     | 5.000     | 7.000     | 8.000      |
| **Pressure Systolic**     | 374.000   | 128.553    | 7.748      | 115.000  | 125.000   | 130.000   | 135.000   | 142.000    |
| **Pressure Diastolic**    | 374.000   | 84.650     | 6.162      | 75.000   | 80.000    | 85.000    | 90.000    | 95.000     |
| **Heart Rate**            | 374.000   | 70.166     | 4.136      | 65.000   | 68.000    | 70.000    | 72.000    | 86.000     |
| **Daily Steps**           | 374.000   | 6816.845   | 1617.916   | 3000.000 | 5600.000  | 7000.000  | 8000.000  | 10000.000  |


- Age : L'âge moyen est de 42 ans, avec un minimum de 27 ans et un maximum de 59 ans.

- Sleep Duration : La durée moyenne du sommeil est de 7.1 heures, avec une variation entre 5.8 et 8.5 heures.

- Quality of Sleep : La qualité moyenne du sommeil est évaluée à 7.3 sur 10.

- Physical Activity Level : Les niveaux d'activité physique varient considérablement, avec une moyenne de 59.

- Stress Level : Le niveau de stress moyen est de 5.4 sur une échelle allant jusqu'à 8.

- Pressure Systolic/Diastolic : La pression artérielle systolique moyenne est de 128.6 mmHg, tandis que la diastolique est de 84.6 mmHg.

- Heart Rate : Le rythme cardiaque moyen est de 70 battements par minute.

- Daily Steps : Le nombre moyen de pas quotidiens est de 6817, avec un minimum de 3000 et un maximum de 10000.
  
# I.5. Statistiques univariées
- Sleep Duration

![image](https://github.com/user-attachments/assets/f2dd4001-afd6-4de4-b30a-e716c262d355)

**Interprétation**

**5.1. Répartition Majoritaire :**

Les Engineers ont la plus grande proportion avec 22.7 % de la durée totale de sommeil, ce qui peut indiquer qu'ils dorment le plus parmi les professions représentées.
Les Sales Representatives suivent avec 18.2 %.

**Répartition Moyenne :**

Les Doctors représentent 13.6 %.
Les Software Engineers et Nurses sont à 9.1 % chacun.

**Répartition Minoritaire :**

Les Teachers, Salespersons, Lawyers, Scientists, Accountants, et Managers ont chacun une proportion de 4.5 %, ce qui indique une durée de sommeil relativement faible par rapport aux autres professions.

**2. Utilisation**

Ce diagramme circulaire est utile pour visualiser la répartition de la durée de sommeil par profession de manière claire et concise. Il permet de :

Comparer rapidement les durées de sommeil entre différentes professions.
Identifier les professions avec des durées de sommeil significativement plus élevées ou plus faibles.
Mettre en évidence des catégories spécifiques (comme les Teachers dans ce graphique).

**3. Conclusion**

Le diagramme circulaire montre que les Engineers et les Sales Representatives ont les durées de sommeil les plus longues, tandis que les autres professions ont des durées de sommeil plus courtes, avec des proportions similaires pour certaines catégories.

# I.6. Statistiques bivariées

![image](https://github.com/user-attachments/assets/060f7d7f-e485-429b-a9b2-628891df166f)

En examinant la répartition de la durée de sommeil par genre,
on observe des médianes similaires pour les hommes et les femmes (environ 7.2 heures). Cependant, la dispersion des données révèle que les femmes présentent une variabilité légèrement plus élevée,
avec une étendue maximale atteignant 8.5 heures, tandis que celle des hommes s’arrête à 8.1 heures. Le mode de sommeil dominant reste situé dans la tranche de 6.5 à 8 heures pour les deux genres,
traduisant une consistance globale. Cette analyse permet d’identifier les éventuelles différences entre les groupes étudiés et d’évaluer si des comportements spécifiques ou des facteurs contextuels influencent la durée totale de sommeil.

# II. Prétraitement des données

**II.1. Relation entre les variables**

Comprendre les liens et corrélations entre les variables revêt une importance cruciale dans le domaine du machine learning.
Cette analyse permet de réduire la dimensionnalité en éliminant des variables redondantes, simplifiant ainsi le modèle tout en préservant ses performances.

![image](https://github.com/user-attachments/assets/7e83c28e-4df8-4be7-8449-c06869d8689c)

**1. Encodage des Variables Catégorielles**
Les variables catégorielles, comme Sleep Disorder, doivent être converties en format numérique pour être compréhensibles par les algorithmes de machine learning.
Les variables catégorielles, comme Sleep Disorder, doivent être converties en format numérique pour être compréhensibles par les algorithmes de machine learning.

*Label Encoding :* Attribue un entier unique à chaque catégorie, utile pour des relations ordonnées.

*OneHot Encoding :* Crée des colonnes binaires pour chaque catégorie, permettant aux modèles de mieux comprendre les relations non ordonnées.

*Importance :* Ces techniques évitent que les modèles interprètent à tort les catégories comme des valeurs numériques continues.

**2. Mise à l'Échelle des Données**

Les colonnes numériques, telles qu'Age, Sleep Duration et Stress Level, ont été mises à l'échelle à l'aide de MinMaxScaler pour les normaliser entre 0 et 1.
*Importance :* Cela garantit que toutes les variables sont sur le même ordre de grandeur, empêchant qu'une variable à grande échelle (comme l'âge) domine les autres dans le processus d'apprentissage.

**3. Traitement des Valeurs Manquantes (NaN)**

La colonne Sleep Disorder contenait des valeurs manquantes (NaN), qui ont été remplacées par un label descriptif comme No Disorder.
Importance : Attribuer une signification aux valeurs manquantes évite que les algorithmes considèrent ces entrées comme des erreurs ou omettent des données importantes.

**4. Encodage de la Colonne Sleep Disorder**

Une fois les valeurs NaN remplacées, la colonne Sleep Disorder a été encodée pour transformer les catégories (No Disorder, Sleep Apnea, Insomnia) en valeurs numériques.
*Importance :* Cela permet de définir des classes claires pour les algorithmes de classification, simplifiant leur compréhension et leur traitement des données cibles.

# III.Séparation de donnée

Nous avons divisé notre jeu de données en ensembles d'entraînement (70%) et de test
(30%). Cela permet d'évaluer la performance de nos modèles sur des données non vues
et de s'assurer que le modèle est capable de généraliser à de nouveaux cas.

● Feature Engineering
Dans le cadre de cette étude, il pourrait être utile de créer de nouvelles variables à partir
des variables existantes. Par exemple, nous pourrions créer une nouvelle variable basée
sur l'interaction entre la qualité du sommeil et la durée du sommeil, car ces deux facteurs
peuvent influencer ensemble la survenue d'un trouble du sommeil.

● **Validation Croisée 10-Fold**

Dans notre étude, nous avons utilisé la validation croisée 10-Fold pour évaluer les performances du modèle. Voici une brève explication :

*Principe :*

Les données sont divisées en 10 sous-ensembles égaux appelés folds.
Le modèle est entraîné sur 9 folds et validé sur le fold restant. Ce processus est répété 10 fois, en utilisant à chaque itération un fold différent pour la validation.

*Avantage :*

Chaque échantillon apparaît une fois dans le jeu de validation et 9 fois dans le jeu d’entraînement.
Cela maximise l’utilisation des données disponibles et fournit une évaluation robuste et fiable des performances.
Réduction du biais et de la variance :

La répétition sur plusieurs folds permet d’obtenir des résultats plus stables et moins dépendants d’une seule division des données.

*Visualisation :*

Le graphique illustre la répartition des échantillons dans chaque fold :
Les zones claires représentent les échantillons utilisés pour la validation.
Les zones sombres correspondent aux échantillons utilisés pour l’entraînement.

![image](https://github.com/user-attachments/assets/89682b6a-9508-4c16-a995-378ad964e7f6)

*Interprétation des résultats :*

Les performances du modèle sont calculées pour chaque fold, puis la moyenne des résultats donne une estimation fiable de la capacité du modèle à généraliser sur des données non vues.

# IV. Modélisation

**IV.1. Modèles évalués :**

**K-Nearest Neighbors (KNN):**  Un modèle de classification basé sur la proximité, où les prédictions sont faites en fonction des classes des k voisins les plus proches.

**Arbres de Décision :** Un ensemble d'arbres de décision qui agrège leurs prédictions pour améliorer la robustesse et la généralisation du modèle.

**Naive Bayes :** Naive Bayes est un modèle basé sur le théorème de Bayes en supposant que les caractéristiques sont indépendantes les unes des autres (hypothèse naïve).

**Random Forest :** Random Forest est un ensemble d'arbres de décision. Il utilise la technique de bagging (Bootstrap Aggregating) pour combiner plusieurs arbres et améliorer la précision.

**XGBoost :** Une implémentation optimisée de l'algorithme de gradient boosting, connue pour sa vitesse et sa performance.

**Support Vector Machine (SVM) :** Un modèle de classification qui cherche à trouver un hyperplan optimal pour séparer les classes dans l'espace des caractéristiques.
Chaque modèle a été évalué avec plusieurs métriques pour assurer une comparaison complète.

*Optimisation :* Utilisation de GridSearchCV pour régler les hyperparamètres des modèles afin d’améliorer leurs performances.

**IV.2. Évaluation et Comparaison des Modèles :**
Les performances des modèles ont été mesurées en utilisant les métriques suivantes :

**-Précision (accuracy)**

**-Rappel (recall)**

**-Score F1 (harmonisation entre précision et rappel)**


## Matrice de confusion pour analyser les erreurs par classe

**K-Nearest Neighbors (KNN)**

![image](https://github.com/user-attachments/assets/915b2bfb-9311-4fce-9396-b7d3b14a0abc)

**Arbres de Décision**

![image](https://github.com/user-attachments/assets/c4a88421-c7a3-498a-9350-3207341e73b4)

**Naive Bayes**

![image](https://github.com/user-attachments/assets/06bab0ed-7982-4530-a1e6-5306bf1eb1e1)

**Random Forest**

![image](https://github.com/user-attachments/assets/afde3c13-0f59-43e3-acef-da6675dd31fa)

**XGBoost**

![image](https://github.com/user-attachments/assets/3057dd4b-74b0-4488-a873-85ea601f1ad3)

**Support Vector Machine (SVM)**

![image](https://github.com/user-attachments/assets/51d21b63-b623-4d06-8cc4-257cbdda1e89)



## V. Performances des Modèles de Machine Learning

| Modèle           | Accuracy   | Precision | Recall  | F1-Score |
|-------------------|------------|-----------|---------|----------|
| KNN              | 86.67%     | 0.87      | 0.87    | 0.87     |
| Decision Tree    | 87.00%     | 0.87      | 0.87    | 0.86     |
| Random Forest    | 88.00%     | 0.88      | 0.88    | 0.88     |
| Naive Bayes      | 88.00%     | 0.89      | 0.88    | 0.88     |
| SVM              | 91.00%     | 0.91      | 0.91    | 0.91     |
| XGBoost          | 93.33%     | 0.94      | 0.93    | 0.93     |

**Observations :** Nous constatons que XGBoost montre la meilleure performance globale avec une précision de 93.33% et des scores élevés sur toutes les métriques.

et SVM est le deuxième meilleur modèle avec une précision de 91.00%, offrant des performances équilibrées.

nous avons aussi Naive Bayes et Random Forest ont obtenu des performances similaires avec une précision de 88.00%, mais Naive Bayes a une meilleure précision.

tandis que le Decision Tree et KNN sont efficaces mais légèrement inférieurs en termes de précision et de robustesse.

## V.1. Comparaison des Modèles

| Modèle            | Précision  | Avantages                                   | Inconvénients                             |
|--------------------|------------|---------------------------------------------|-------------------------------------------|
| KNN               | 86.67%     | Simple à implémenter                        | Moins robuste aux grandes dimensions      |
| Arbres de Décision| 86.67%     | Interprétabilité facile                     | Moins performant pour les classes minoritaires |
| Naive Bayes       | 88%        | Rapide et simple                            | Hypothèse d'indépendance des features     |
| Random Forest     | 88%        | Bonne gestion du surapprentissage           | Temps de calcul élevé                     |
| XGBoost           | 93.33%     | Meilleure performance globale               | Optimisation complexe                     |
| SVM (Optimisé)    | 90.67%     | Bonne précision sur toutes les classes      | Temps d’entraîment long                   |

## VI Conclusion et Recommandations

Sur la base de cette analyse, nous pouvons conclure en disant que XGBoost s’impose comme le meilleur modèle pour la détection des troubles du sommeil, offrant une précision élevée et des performances équilibrées sur toutes les classes.
Toutefois, si la complexité du modèle ou les temps de calcul sont des contraintes, le Random Forest ou le SVM optimisé constituent d'excellentes alternatives.

**Pour l’amélioration continue :**

Collecter davantage de données pour éviter le surapprentissage.

Explorer l’importance des features avec Random Forest ou XGBoost pour réduire la complexité du modèle.


## VII. Déploiement des Modèles dans une Application Flask

Une interface utilisateur a été créée pour intégrer les modèles entraînés dans une application web avec Flask.
L'utilisateur peut entrer les paramètres d'un patient (âge, IMC, heures de sommeil et niveau de stress) pour prédire les anomalies du sommeil.

Architecture de l'Application :
Backend : Flask
Frontend : HTML, CSS

## Features
- User-friendly interface to input data for prediction
- Integration of trained machine learning models
- Flask-based backend with routes for prediction and result visualization
- Pre-trained models loaded for fast and efficient inference

## Prerequisites
To clone and run the project, ensure you have the following installed on your system:

- Python
- Flask
- Joblib
- Scikit-learn
- NumPy
- Pandas
- Any other required libraries listed in `requirements.txt`

## How to Clone the Project

1. **Clone the repository**:

   ```bash
   git clone https://github.com/TressyIssa/Plateforme-de-detection-des-anomalies-du-sommeil.git
   cd sleep-disorder-prediction
   ```

2. **Set up a virtual environment (optional but recommended)**:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. **Install the required dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

4. **Prepare the trained models**:

   - Place the trained machine learning models (e.g., `XGBoost_model.pkl`) in the `models/` directory. If you don’t have the models locally, refer to the Colab notebook to export and download them using `joblib` or `pickle`.

5. **Run the Flask application**:

   ```bash
   python app.py
   ```

6. **Access the application**:

   Open your browser and navigate to `http://127.0.0.1:5000` to use the application.

## Directory Structure

```
project-directory/
|-- app.py                # Main Flask application
|-- models/               # Directory to store trained ML models
|   |-- XGBoost_model.pkl
|-- templates/            # HTML templates for Flask
|   |-- predict_form.html
|-- static/               # Static files (CSS, JavaScript, images)
|-- requirements.txt      # Python dependencies
|-- README.md             # Project documentation
```

## How to Train the Models
If you wish to retrain the models:

1. Use the provided Colab notebook or Python script for training.
2. Preprocess the dataset (normalize, encode variables, handle NaN values).
3. Train models like Random Forest, XGBoost, SVM, etc.
4. Save the trained models using `joblib`:

   ```python
   import joblib
   joblib.dump(model, "models/XGBoost_model.pkl")
   ```

5. Replace the existing model files in the `models/` directory with the new ones.

## Contributions
Feel free to contribute by submitting issues or pull requests. For major changes, please open an issue first to discuss what you would like to change.


