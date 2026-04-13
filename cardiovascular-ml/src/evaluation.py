"""
evaluation.py
=============
Evaluation utilities for CVD classification models.
Author: Takwa Haj
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    precision_score, recall_score, confusion_matrix,
    classification_report, roc_curve
)


def evaluate_model(model, X_test, y_test, model_name: str = "Model") -> dict:
    """
    Evaluate a trained classifier and return a metrics dictionary.

    Args:
        model:       Trained sklearn-compatible classifier.
        X_test:      Test features.
        y_test:      True labels.
        model_name:  Name label for reporting.

    Returns:
        Dictionary with accuracy, f1, auc, precision, recall.
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        'model_name': model_name,
        'accuracy':   accuracy_score(y_test, y_pred),
        'f1':         f1_score(y_test, y_pred),
        'auc':        roc_auc_score(y_test, y_prob),
        'precision':  precision_score(y_test, y_pred),
        'recall':     recall_score(y_test, y_pred),
        'y_pred':     y_pred,
        'y_prob':     y_prob,
    }
    return metrics


def print_metrics_table(results: dict) -> None:
    """Print a formatted comparison table for multiple models."""
    print(f"\n{'Model':<30} {'Acc':>6} {'F1':>6} {'AUC':>6} {'Prec':>6} {'Rec':>6}")
    print('-' * 62)
    for name, r in results.items():
        print(f"{name:<30} {r['accuracy']:>6.3f} {r['f1']:>6.3f} "
              f"{r['auc']:>6.3f} {r['precision']:>6.3f} {r['recall']:>6.3f}")


def plot_roc_curves(results: dict, y_test) -> None:
    """Plot ROC curves for all models on a single axes."""
    palette = ['#065A82', '#E63946', '#2A9D8F', '#E9C46A', '#6A0572']
    fig, ax = plt.subplots(figsize=(8, 6))

    for (name, r), color in zip(results.items(), palette):
        fpr, tpr, _ = roc_curve(y_test, r['y_prob'])
        ax.plot(fpr, tpr, color=color, lw=2,
                label=f"{name} (AUC = {r['auc']:.3f})")

    ax.plot([0, 1], [0, 1], 'k--', lw=1)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves — All Models', fontweight='bold')
    ax.legend(loc='lower right', fontsize=9)
    ax.set_facecolor('#f8f9fa')
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(y_test, y_pred, model_name: str = "Model") -> None:
    """Plot a labeled confusion matrix heatmap."""
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap='Blues',
                xticklabels=['No CVD', 'CVD'],
                yticklabels=['No CVD', 'CVD'])
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(f'Confusion Matrix — {model_name}', fontweight='bold')
    plt.tight_layout()
    plt.show()


def error_analysis(y_test, y_pred, X_test: pd.DataFrame) -> None:
    """Print a clinical error analysis report."""
    y_test_arr = np.array(y_test)
    fp = ((y_pred == 1) & (y_test_arr == 0)).sum()
    fn = ((y_pred == 0) & (y_test_arr == 1)).sum()
    tp = ((y_pred == 1) & (y_test_arr == 1)).sum()
    tn = ((y_pred == 0) & (y_test_arr == 0)).sum()

    print(f"\n{'='*45}")
    print(f"  CLINICAL ERROR ANALYSIS")
    print(f"{'='*45}")
    print(f"  True Positives  (CVD caught)   : {tp:>5,}")
    print(f"  True Negatives  (healthy OK)   : {tn:>5,}")
    print(f"  False Positives (false alarm)  : {fp:>5,}")
    print(f"  False Negatives (missed CVD)   : {fn:>5,}  ⚠️")
    print(f"{'='*45}")
    fn_rate = fn / y_test_arr.sum()
    fp_rate = fp / (y_test_arr == 0).sum()
    print(f"  False Negative Rate : {fn_rate:.2%}  (missed CVD)")
    print(f"  False Positive Rate : {fp_rate:.2%}  (false alarms)")
    print(f"{'='*45}")
    print(f"\n  ℹ️  In clinical use, minimize FN Rate (missing CVD is dangerous)")
