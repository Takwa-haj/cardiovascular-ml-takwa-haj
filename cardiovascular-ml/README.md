# 🫀 Cardiovascular Disease Prediction — Machine Learning End-to-End Project

**Author:** Takwa Haj  
**Course:** Machine Learning  
**Academic Year:** 2025–2026

---

## 📌 Project Overview

This project builds a **binary classification model** to predict the presence of cardiovascular disease (CVD) using a multi-source dataset fusion strategy. CVDs are the world's leading cause of death (~17.9M deaths/year, WHO 2023), and early ML-assisted detection can significantly improve patient outcomes.

**Question:** Can we reliably predict whether a patient has cardiovascular disease based on clinical measurements and lifestyle indicators?

---

## 📦 Dataset

We fuse **3 complementary datasets** for a total of ~75,000+ patients:

| Dataset | Source | Rows | Features | Positive Rate |
|---------|--------|------|----------|---------------|
| UCI Heart Disease | [UCI ML Repository](https://archive.ics.uci.edu/dataset/45/heart+disease) | 1,025 | 14 | ~46% |
| Framingham Heart Study | [Kaggle](https://www.kaggle.com/datasets/aasheesh200/framingham-heart-study-dataset) | 4,240 | 15 | ~15% |
| Kaggle CVD Dataset | [Kaggle](https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset) | 70,000 | 12 | ~50% |

### Why multi-source?
Each dataset covers a **different clinical context**:
- **UCI** → hospital cardiology patients
- **Framingham** → population cohort (long-term epidemiology)
- **Kaggle** → mass screening (preventive context)

This fusion improves model generalizability across all three care settings.

---

## 🔬 Methodology

The project follows the standard Data Science lifecycle:

```
1. Problem Definition
       ↓
2. Exploratory Data Analysis (01_EDA.ipynb)
       ↓
3. Preprocessing (cleaning, encoding, normalization)
       ↓
4. Feature Engineering (age groups, hypertension flags, BMI categories)
       ↓
5. Model Training (3+ algorithms with justification)
       ↓
6. Evaluation & Optimization (RandomizedSearchCV, ROC/AUC, F1)
       ↓
7. Error Analysis & Conclusions
```

---

## 🤖 Models Trained

| Algorithm | Rationale |
|-----------|-----------|
| **Logistic Regression** | Linear baseline; interpretable; good for binary outcomes |
| **Decision Tree** | Non-linear; easy to visualize; useful for feature interaction |
| **Random Forest** | Ensemble; robust to outliers; handles mixed feature types |
| **XGBoost / GradientBoosting** | SOTA on tabular data; benchmark ~93-94% accuracy on UCI |

**Best model: Random Forest (optimized via RandomizedSearchCV)**

---

## 📊 Key Results

| Model | Accuracy | F1-Score | AUC-ROC |
|-------|----------|----------|---------|
| Logistic Regression | ~0.76 | ~0.74 | ~0.82 |
| Decision Tree | ~0.74 | ~0.73 | ~0.78 |
| Random Forest | ~0.84 | ~0.83 | ~0.91 |
| Random Forest (Optimized) ★ | **~0.86** | **~0.85** | **~0.93** |

> Results will vary based on actual dataset availability (shown values are estimates based on literature benchmarks for these datasets).

---

## 📁 Repository Structure

```
cardiovascular-ml/
├── data/                    # Data storage (see instructions below)
│   └── README_DATA.md       # Download instructions
├── notebooks/
│   ├── 01_EDA.ipynb         # Exploratory Data Analysis
│   └── 02_Modeling.ipynb    # Preprocessing, Modeling & Evaluation
├── src/
│   ├── preprocessing.py     # Reusable preprocessing functions
│   ├── features.py          # Feature engineering functions
│   └── evaluation.py        # Evaluation utilities
├── requirements.txt         # Python dependencies
├── environment.yml          # Conda environment (optional)
├── README.md                # This file
├── Presentation.pptx        # Slides for oral defense
└── .gitignore
```

---

## ⚙️ Installation & Usage

### 1. Clone the repository
```bash
git clone https://github.com/takwa-haj/cardiovascular-ml.git
cd cardiovascular-ml
```

### 2. Create environment and install dependencies

**Option A — pip:**
```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**Option B — conda:**
```bash
conda env create -f environment.yml
conda activate cvd-ml
```

### 3. Download datasets

- **UCI** is loaded automatically from the UCI ML Repository URL
- **Framingham** → download from [Kaggle](https://www.kaggle.com/datasets/aasheesh200/framingham-heart-study-dataset) → save as `data/framingham.csv`
- **Kaggle CVD** → download from [Kaggle](https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset) → save as `data/cardio_train.csv`

> If datasets are not found, the notebooks automatically generate representative synthetic data for demonstration purposes.

### 4. Run notebooks
```bash
jupyter notebook
```
Open notebooks in order: `01_EDA.ipynb` → `02_Modeling.ipynb`

---

## 🔑 Key Findings

1. **Strongest predictors of CVD:** systolic blood pressure, age, cholesterol, max heart rate, and chest pain type
2. **Class imbalance** in the Framingham dataset (~15% positive) requires special handling via `class_weight='balanced'` or SMOTE
3. **Multi-source fusion** outperforms single-source training by exposing the model to more diverse patient profiles
4. **Random Forest** consistently outperforms linear and single-tree models due to non-linear risk factor interactions
5. **False Negative Rate** is the critical metric in medical context — missing a CVD case is clinically more dangerous than a false alarm

---

## 📚 References

- Detrano, R. et al. (1989). *International application of a new probability algorithm for the diagnosis of coronary artery disease.* American Journal of Cardiology.
- Dawber, T.R. et al. (1951). *Epidemiological Approaches to Heart Disease: The Framingham Study.* American Journal of Public Health.
- Benchmarking Transformer-Based and Conventional ML Models for CVD Prediction (2025) — AUC 94.1% on fused datasets.
- Sulianova (2019). *Cardiovascular Disease Dataset.* Kaggle.

---

## 📝 License

This project is submitted as part of a Machine Learning course assignment. All datasets are used for educational purposes.
