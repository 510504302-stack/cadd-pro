"""
Machine learning utility functions: training, evaluation, model management.
"""
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, roc_auc_score,
    classification_report, matthews_corrcoef
)
import streamlit as st
from typing import Optional, Dict, Any, Tuple
import warnings
warnings.filterwarnings('ignore')

# Try importing XGBoost (optional)
try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False


MODEL_CONFIGS = {
    "Random Forest": {
        "class": RandomForestClassifier,
        "params": {
            "n_estimators": {"type": "slider", "min": 50, "max": 500, "default": 100, "step": 10},
            "max_depth": {"type": "slider", "min": 1, "max": 50, "default": 10, "step": 1},
            "max_features": {"type": "slider", "min": 0.1, "max": 1.0, "default": 0.5, "step": 0.1},
            "min_samples_split": {"type": "slider", "min": 2, "max": 20, "default": 2, "step": 1},
        }
    },
    "Logistic Regression": {
        "class": LogisticRegression,
        "params": {
            "C": {"type": "slider", "min": 0.01, "max": 10.0, "default": 1.0, "step": 0.01},
            "max_iter": {"type": "slider", "min": 100, "max": 2000, "default": 1000, "step": 100},
        }
    },
    "SVM": {
        "class": SVC,
        "params": {
            "C": {"type": "slider", "min": 0.01, "max": 10.0, "default": 1.0, "step": 0.01},
            "kernel": {"type": "select", "options": ["rbf", "linear", "poly"], "default": "rbf"},
            "gamma": {"type": "select", "options": ["scale", "auto"], "default": "scale"},
        }
    },
}

if HAS_XGBOOST:
    MODEL_CONFIGS["XGBoost"] = {
        "class": XGBClassifier,
        "params": {
            "n_estimators": {"type": "slider", "min": 50, "max": 500, "default": 100, "step": 10},
            "max_depth": {"type": "slider", "min": 1, "max": 20, "default": 6, "step": 1},
            "learning_rate": {"type": "slider", "min": 0.01, "max": 0.5, "default": 0.1, "step": 0.01},
            "subsample": {"type": "slider", "min": 0.5, "max": 1.0, "default": 0.8, "step": 0.1},
        }
    }


def preprocess_data(fp_file: str) -> pd.DataFrame:
    """Preprocess fingerprint data: drop NA, convert to numeric."""
    data = pd.read_csv(fp_file).dropna()
    for col in data.select_dtypes(include=['object']).columns:
        data[col] = pd.to_numeric(data[col], errors='coerce')
    return data.dropna()


def train_model(
    fp_file: str,
    project_dir: str,
    model_name: str,
    params: Dict[str, Any],
    test_size: float = 0.2,
    cv_folds: int = 5
) -> Tuple[Optional[Any], Dict[str, Any]]:
    """Train a model and return model object + metrics dict."""
    data = preprocess_data(fp_file)
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    metrics = {}

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
    except ValueError as e:
        st.error(f"数据划分出错：{e}")
        return None, metrics

    # Create model instance
    if model_name in MODEL_CONFIGS:
        model_cls = MODEL_CONFIGS[model_name]["class"]
        # SVC needs explicit probability=True; RF/XGB/LR support it by default
        if model_cls == SVC:
            params = {**params, "probability": True}
        model = model_cls(**params, random_state=42)
    else:
        st.error(f"未知模型: {model_name}")
        return None, metrics

    # Train
    try:
        model.fit(X_train, y_train)
    except Exception as e:
        st.error(f"模型训练失败：{e}")
        return None, metrics

    # Save model
    model_filename = os.path.join(project_dir, "model.pkl")
    joblib.dump(model, model_filename)

    # Predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None

    # Basic metrics
    metrics['accuracy'] = accuracy_score(y_test, y_pred)
    metrics['precision'] = precision_score(y_test, y_pred, average='binary', zero_division=0)
    metrics['recall'] = recall_score(y_test, y_pred, average='binary', zero_division=0)
    metrics['f1'] = f1_score(y_test, y_pred, average='binary', zero_division=0)
    metrics['mcc'] = matthews_corrcoef(y_test, y_pred)
    metrics['confusion_matrix'] = confusion_matrix(y_test, y_pred)

    # AUC
    if y_proba is not None:
        metrics['roc_auc'] = roc_auc_score(y_test, y_proba)
        metrics['fpr'], metrics['tpr'], _ = roc_curve(y_test, y_proba)
    else:
        metrics['roc_auc'] = None

    # Cross-validation
    try:
        cv = StratifiedKFold(n_splits=min(cv_folds, min(np.bincount(y.astype(int)))), shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
        metrics['cv_mean'] = cv_scores.mean()
        metrics['cv_std'] = cv_scores.std()
    except Exception:
        metrics['cv_mean'] = None
        metrics['cv_std'] = None

    # Feature importance (if available)
    if hasattr(model, 'feature_importances_'):
        metrics['feature_importance'] = model.feature_importances_
    elif hasattr(model, 'coef_'):
        metrics['feature_importance'] = np.abs(model.coef_[0]) if model.coef_.ndim > 1 else np.abs(model.coef_)

    return model, metrics


def generate_evaluation_plots(metrics: Dict[str, Any], project_dir: str, X_columns: list = None):
    """Generate and save all evaluation plots."""
    plot_style()

    # 1. ROC Curve
    if metrics.get('fpr') is not None and metrics.get('tpr') is not None:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(metrics['fpr'], metrics['tpr'], color='#2563eb', lw=2.5,
                label=f"ROC (AUC = {metrics['roc_auc']:.3f})")
        ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5, label='Random')
        ax.fill_between(metrics['fpr'], metrics['tpr'], alpha=0.15, color='#2563eb')
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title('ROC Curve', fontsize=14, fontweight='bold')
        ax.legend(loc='lower right', fontsize=10)
        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.02])
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(project_dir, "roc_curve.png"), dpi=150, bbox_inches='tight')
        plt.close(fig)

    # 2. Confusion Matrix
    cm = metrics.get('confusion_matrix')
    if cm is not None:
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    annot_kws={'size': 14, 'weight': 'bold'},
                    cbar_kws={'shrink': 0.8})
        ax.set_xlabel('Predicted Label', fontsize=12)
        ax.set_ylabel('True Label', fontsize=12)
        ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
        fig.tight_layout()
        fig.savefig(os.path.join(project_dir, "confusion_matrix.png"), dpi=150, bbox_inches='tight')
        plt.close(fig)

    # 3. Feature Importance
    if metrics.get('feature_importance') is not None and X_columns is not None:
        importance = metrics['feature_importance']
        top_n = min(30, len(importance))
        indices = np.argsort(importance)[-top_n:]

        fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.35)))
        colors = plt.cm.Blues_r(np.linspace(0.3, 0.9, top_n))
        ax.barh(range(top_n), importance[indices], color=colors, edgecolor='white', linewidth=0.5)
        ax.set_yticks(range(top_n))
        ax.set_yticklabels([f"Feature {X_columns[i]}" if isinstance(X_columns[i], (int, float))
                           else str(X_columns[i])[:20] for i in indices], fontsize=9)
        ax.set_xlabel('Importance', fontsize=12)
        ax.set_title('Top Feature Importance', fontsize=14, fontweight='bold')
        ax.invert_yaxis()
        fig.tight_layout()
        fig.savefig(os.path.join(project_dir, "feature_importance.png"), dpi=150, bbox_inches='tight')
        plt.close(fig)

    # 4. Metrics Summary Bar Chart
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'MCC']
    metric_values = [
        metrics.get('accuracy', 0) or 0,
        metrics.get('precision', 0) or 0,
        metrics.get('recall', 0) or 0,
        metrics.get('f1', 0) or 0,
        metrics.get('mcc', 0) or 0
    ]
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ['#2563eb', '#7c3aed', '#059669', '#d97706', '#dc2626']
    bars = ax.bar(metric_names, metric_values, color=colors, edgecolor='white', linewidth=1.5)
    for bar, val in zip(bars, metric_values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.set_ylim(0, 1.15)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Model Performance Metrics', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(project_dir, "metrics_summary.png"), dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_style():
    """Set consistent matplotlib style."""
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except Exception:
        try:
            plt.style.use('seaborn-v0_8')
        except Exception:
            plt.style.use('ggplot')
    plt.rcParams.update({
        'font.size': 11,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.dpi': 120,
    })
