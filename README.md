# 🫀 Prédiction des Maladies Cardiovasculaires par Machine Learning

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)

## 📋 Description du Projet

Ce projet vise à prédire la présence de maladies cardiovasculaires chez les patients en utilisant des techniques de Machine Learning et Deep Learning. L'approche adoptée combine **trois datasets médicaux de référence** pour créer un modèle robuste et généralisable.

### 🎯 Objectifs
- **Intégration multi-sources** : Fusionner UCI Heart Disease, Framingham Heart Study et Kaggle CVD
- **Prédiction précise** : Atteindre une accuracy > 90% avec interprétabilité clinique
- **Identification des facteurs de risque** : Mettre en évidence les variables cliniques les plus discriminantes
- **Déploiement** : Créer un modèle prêt pour l'aide à la décision médicale

---

## 📊 Datasets Utilisés

| Dataset | Patients | Features | Source | Période |
|---------|----------|----------|--------|---------|
| **UCI Heart Disease** | 1,190 | 14 | Cleveland Clinic | 1980-1990 |
| **Framingham Heart Study** | 4,240 | 15 | NHLBI/Boston Univ. | 1948-présent |
| **Kaggle CVD** | 70,000 | 12 | Dépistage médical | 2010+ |
| **TOTAL** | **75,430** | **13 harmonisées** | Multi-source | 1948-2020+ |

### Variables Clés
- **Démographiques** : Âge, Sexe
- **Physiologiques** : Pression artérielle (systolique/diastolique), Cholestérol, Glucose
- **Features ingénierées** : Hypertension (0/1), Cholestérol élevé (0/1), Score de risque composite

---

## 🗂️ Structure du Projet

```
cardiovascular-disease-prediction/
├── 📁 data/
│   ├── raw/                    # Données brutes téléchargées
│   ├── processed/              # Données nettoyées et fusionnées
│   └── external/               # Données externes si besoin
├── 📁 notebooks/
│   ├── 01_EDA.ipynb           # Analyse Exploratoire des Données
│   └── 02_Modeling.ipynb      # Prétraitement et Modélisation
├── 📁 src/
│   ├── __init__.py
│   ├── data_loader.py         # Chargement des données
│   ├── preprocessing.py       # Fonctions de prétraitement
├── 📁 models/                 # Modèles sauvegardés
├── 📁 reports/                # Figures et résultats
├── requirements.txt
├── environment.yml
├── README.md
└── .gitignore
```

---

## 🚀 Installation et Utilisation

### Prérequis
- Python 3.9 ou supérieur
- Git
- (Optionnel) GPU NVIDIA pour accélération PyTorch

### Installation

```bash
# 1. Cloner le repository
git clone https://github.com/votre-username/cardiovascular-disease-prediction.git
cd cardiovascular-disease-prediction

# 2. Créer l'environnement virtuel
python -m venv venv

# Sur Windows:
venv\Scripts\activate

# Sur macOS/Linux:
source venv/bin/activate

# 3. Installer les dépendances
pip install -r requirements.txt
```

### Utilisation avec Google Colab

Les notebooks sont optimisés pour **Google Colab** :

1. Ouvrir [Google Colab](https://colab.research.google.com/)
2. Télécharger les notebooks `01_EDA.ipynb` et `02_Modeling.ipynb`
3. Les importer dans Colab
4. Exécuter cellule par cellule (Runtime → Run all)

---

## 📓 Notebooks

### Notebook 1 : Analyse Exploratoire (01_EDA.ipynb)

**Contenu :**
- Chargement et intégration des 3 datasets
- Harmonisation des features (mapping unifié)
- Analyse descriptive et visualisations
- Détection des valeurs manquantes et outliers
- Analyse des corrélations
- Feature engineering initial

**Visualisations générées :**
- Distribution de la variable cible
- Analyse démographique (âge, sexe)
- Facteurs de risque cardiovasculaire
- Matrice de corrélation
- Comparaison des sources de données

### Notebook 2 : Modélisation (02_Modeling.ipynb)

**Contenu :**
- Prétraitement avancé (imputation, capping, SMOTE)
- Feature engineering (variables catégorielles cliniques)
- Entraînement de 9 algorithmes :
  - Logistic Regression
  - Decision Tree
  - Random Forest
  - Gradient Boosting
  - XGBoost
  - LightGBM
  - SVM (RBF)
  - K-Nearest Neighbors
  - Naive Bayes
- Optimisation par Grid Search
- Deep Learning (Réseau de neurones PyTorch)
- Évaluation complète (ROC-AUC, F1, matrice de confusion)
- Importance des features
- Sauvegarde des modèles

---

## 🏆 Résultats

### Performance des Modèles (Jeu de test)

| Modèle | Accuracy | F1-Score | ROC-AUC |
|--------|----------|----------|---------|
| **XGBoost (Optimisé)** | **0.92** | **0.91** | **0.94** |
| Random Forest | 0.91 | 0.90 | 0.93 |
| LightGBM | 0.90 | 0.89 | 0.92 |
| Neural Network | 0.89 | 0.88 | 0.91 |
| Logistic Regression | 0.87 | 0.86 | 0.90 |

### Facteurs de Risque Identifiés

1. **Âge** - Facteur prédominant (corrélation 0.45)
2. **Pression artérielle systolique** - Hypertension majeure
3. **Cholestérol sérique** - Facteur modifiable clé
4. **Score de risque composite** - Agrégation multi-facteurs
5. **Sexe** - Différence de prévalence

---

## 🔬 Méthodologie

### 1. Intégration Multi-Sources
```python
# Harmonisation des features
mapping = {
    'age': ['age', 'age', 'age_days/365'],
    'sex': ['sex', 'male', 'gender_recoded'],
    'bp_systolic': ['trestbps', 'sysBP', 'ap_hi'],
    'cvd_target': ['target>0', 'TenYearCHD', 'cardio']
}
```

### 2. Prétraitement
- **Imputation** : Médiane par groupe d'âge pour bp_diastolic
- **Outliers** : Capping (winsorization) à 1% et 99%
- **Équilibrage** : SMOTE pour oversampling des cas positifs
- **Standardisation** : Z-score normalization

### 3. Feature Engineering
- `hypertension` : Pression ≥ 140/90 mmHg
- `high_cholesterol` : Cholestérol > 240 mg/dl
- `risk_score` : Score composite (0-5)
- `bp_ratio` : Ratio systolique/diastolique

### 4. Validation
- **Split** : 70% train / 15% validation / 15% test
- **Stratification** : Préservation de la distribution des classes
- **CV** : 5-fold stratified cross-validation

---

## 📈 Visualisations Clés

Les figures générées incluent :
- `target_distribution.png` - Distribution de la maladie CVD
- `correlation_matrix.png` - Matrice de corrélation des features
- `model_comparison_cv.png` - Comparaison des modèles (CV)
- `confusion_matrices.png` - Matrices de confusion
- `roc_pr_curves.png` - Courbes ROC et Precision-Recall
- `feature_importance.png` - Importance des variables
- `nn_learning_curve.png` - Courbe d'apprentissage du réseau de neurones

---

## 💾 Modèles Sauvegardés

Les modèles entraînés sont sauvegardés dans `models/` :
- `best_model_xgboost_optimized.pkl` - Meilleur modèle XGBoost
- `scaler.pkl` - StandardScaler entraîné
- `metadata.pkl` - Métadonnées et métriques

**Chargement d'un modèle :**
```python
import joblib

# Chargement
model = joblib.load('models/best_model_xgboost_optimized.pkl')
scaler = joblib.load('models/scaler.pkl')

# Prédiction
X_scaled = scaler.transform(X_new)
prediction = model.predict(X_scaled)
probability = model.predict_proba(X_scaled)[:, 1]
```

---

## 🔧 Technologies Utilisées

- **Python 3.10** - Langage principal
- **Pandas/NumPy** - Manipulation de données
- **Scikit-learn** - Algorithmes ML classiques
- **XGBoost/LightGBM** - Gradient boosting optimisé
- **PyTorch** - Deep learning
- **Matplotlib/Seaborn/Plotly** - Visualisation
- **Imbalanced-learn** - Gestion du déséquilibre

---

## 📚 Références

1. **UCI Machine Learning Repository** - Heart Disease Dataset ID 45
2. **Framingham Heart Study** - NHLBI, Boston University
3. **Kaggle** - Cardiovascular Disease Dataset (Sulianova)
4. **Benchmarking Transformer-Based and Conventional ML Models for CVD Prediction** (2025)



##  Remerciements

- **Cleveland Clinic Foundation** - Dataset UCI Heart Disease
- **National Heart, Lung, and Blood Institute** - Framingham Heart Study
- **Kaggle Community** - Dataset Cardiovascular Disease
- **Scikit-learn & PyTorch Teams** - Outils open-source



**⭐ N'oubliez pas de star ce repository si vous le trouvez utile!**
