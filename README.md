# Regression Linéaire / Logistique 
## Machine Learning

### Implémentation dans deux registres différents: 
1. Estimation d'un prix, dans le contexte ou il s'agit de vente de voitures
Il s'agit ici d'un cas de regression linéaire multivariés avec plusieurs features (kilomètres parcourues, ancienneté de la voiture, type du moteur etc.)

2. Estimation de cas avéré ou non d'une maladie, dans le contexte médiacal ou les patients peuvent être atteints de trouble cardio-vasulaires
C'est une classification réalisée à partir d'un set de donnée de 70'000 échantillons, et d'une dizaines de features.

### La méthode appliquée dans les deux registres est sensiblement la même:
- Analyse, preprocessing, visualisation des données initiales (NumPy, Pandas, Matplotlib, Seaborn)
- Implémentation d'un modèle de machine learning via des librairies (SKLearn, SciPy, NumPy)
- Implémentation d'un modèle de machine learning à l'aide d'un algorithme homemade de descente de gradient
- Analyse, Test, Comparaison des résultats
