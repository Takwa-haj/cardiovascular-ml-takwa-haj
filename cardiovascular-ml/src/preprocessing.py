"""
preprocessing.py
================
Reusable preprocessing functions for the CVD multi-source dataset fusion.
Author: Takwa Haj
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


def preprocess_uci(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the UCI Heart Disease dataset.

    Steps:
        - Binarize target (0 = no disease, 1-4 = disease → 1)
        - Impute missing values in 'ca' and 'thal' with median
        - Rename columns to unified schema
        - Remove duplicates

    Args:
        df: Raw UCI dataframe with 14 columns.

    Returns:
        Cleaned dataframe with unified column names.
    """
    df = df.copy()
    df['cvd_target'] = (df['target'] > 0).astype(int)

    for col in ['ca', 'thal']:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    df = df.rename(columns={
        'trestbps': 'bp_systolic',
        'chol': 'cholesterol_val',
        'thalach': 'heart_rate_max',
    })

    features = ['age', 'sex', 'bp_systolic', 'cholesterol_val', 'heart_rate_max',
                'cp', 'fbs', 'restecg', 'exang', 'oldpeak', 'slope', 'ca', 'thal',
                'cvd_target']
    df = df[[f for f in features if f in df.columns]].drop_duplicates()
    df['source'] = 'uci'
    return df


def preprocess_framingham(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the Framingham Heart Study dataset.

    Steps:
        - Impute all missing numeric values with column median
        - Rename columns to unified schema
        - Remove duplicates

    Args:
        df: Raw Framingham dataframe (15 columns).

    Returns:
        Cleaned dataframe with unified column names.
    """
    df = df.copy()
    df = df.rename(columns={
        'male': 'sex',
        'TenYearCHD': 'cvd_target',
        'sysBP': 'bp_systolic',
        'totChol': 'cholesterol_val',
        'heartRate': 'heart_rate_max',
    })

    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = df[col].fillna(df[col].median())

    features = ['age', 'sex', 'bp_systolic', 'cholesterol_val', 'heart_rate_max',
                'currentSmoker', 'cigsPerDay', 'BMI', 'diabetes',
                'prevalentHyp', 'prevalentStroke', 'BPMeds', 'glucose', 'cvd_target']
    df = df[[f for f in features if f in df.columns]].drop_duplicates()
    df['source'] = 'framingham'
    return df


def preprocess_kaggle(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the Kaggle Cardiovascular Disease dataset.

    Steps:
        - Convert age from days to years
        - Recode gender (2=Male→1, 1=Female→0)
        - Compute BMI from height/weight
        - Remove physiologically impossible BP values
        - Filter unrealistic BMI values
        - Rename to unified schema

    Args:
        df: Raw Kaggle CVD dataframe (12 columns).

    Returns:
        Cleaned dataframe with unified column names.
    """
    df = df.copy()
    df['age'] = (df['age'] / 365.25).round(0).astype(int)
    df['sex'] = (df['gender'] == 2).astype(int)
    df['BMI'] = df['weight'] / (df['height'] / 100) ** 2

    # Remove impossible BP values
    df = df[(df['ap_hi'] >= 60) & (df['ap_hi'] <= 250)]
    df = df[(df['ap_lo'] >= 40) & (df['ap_lo'] <= 150)]
    df = df[df['ap_hi'] > df['ap_lo']]

    # Realistic BMI
    df = df[df['BMI'].between(15, 55)]

    df = df.rename(columns={
        'ap_hi': 'bp_systolic',
        'ap_lo': 'bp_diastolic',
        'cardio': 'cvd_target',
    })

    df['cholesterol_high'] = (df['cholesterol'] > 1).astype(int)
    df['glucose_high'] = (df['gluc'] > 1).astype(int)

    features = ['age', 'sex', 'bp_systolic', 'bp_diastolic', 'BMI',
                'cholesterol_high', 'glucose_high', 'smoke', 'alco', 'active', 'cvd_target']
    df = df[[f for f in features if f in df.columns]].drop_duplicates()
    df['source'] = 'kaggle'
    return df


def fuse_datasets(*dfs: pd.DataFrame) -> pd.DataFrame:
    """
    Fuse multiple preprocessed datasets into one DataFrame.

    Performs outer-join style concatenation (all rows, NaN for missing columns),
    then fills NaN with column medians.

    Args:
        *dfs: Any number of preprocessed DataFrames.

    Returns:
        Fused DataFrame ready for modeling.
    """
    fused = pd.concat(list(dfs), ignore_index=True, sort=False)
    for col in fused.select_dtypes(include=[np.number]).columns:
        fused[col] = fused[col].fillna(fused[col].median())
    return fused


def scale_features(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """
    Fit StandardScaler on train set and transform both sets.

    Args:
        X_train: Training features.
        X_test:  Test features.

    Returns:
        Tuple of (X_train_scaled, X_test_scaled, fitted_scaler)
    """
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=X_train.columns
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=X_test.columns
    )
    return X_train_scaled, X_test_scaled, scaler
