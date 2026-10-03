"""Model training, comparison, and threshold analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .data import FEATURES, TARGET, make_demo_dataset


@dataclass
class ModelSuite:
    models: dict[str, Any]
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    probabilities: dict[str, np.ndarray]
    metrics: pd.DataFrame


def _make_models() -> dict[str, Any]:
    models: dict[str, Any] = {
        "Logistic Regression": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                ("model", LogisticRegression(max_iter=1_500, class_weight="balanced", random_state=42)),
            ]
        ),
        "Random Forest": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=220,
                        max_depth=9,
                        min_samples_leaf=8,
                        class_weight="balanced_subsample",
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }

    try:
        from xgboost import XGBClassifier

        models["XGBoost"] = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    XGBClassifier(
                        n_estimators=240,
                        max_depth=3,
                        learning_rate=0.05,
                        subsample=0.85,
                        colsample_bytree=0.9,
                        reg_lambda=1.0,
                        eval_metric="logloss",
                        tree_method="hist",
                        random_state=42,
                        n_jobs=2,
                    ),
                ),
            ]
        )
    except ImportError:
        # The app remains usable without this optional estimator. It is included
        # in requirements.txt to match the full model comparison in the resume.
        pass

    return models


def positive_class_probability(model: Any, rows: pd.DataFrame) -> np.ndarray:
    """Return P(default=1), locating the positive class by its fitted label."""
    class_labels = list(model.classes_)
    positive_index = class_labels.index(1)
    return np.asarray(model.predict_proba(rows))[:, positive_index]


def train_model_suite(dataset: pd.DataFrame | None = None) -> ModelSuite:
    data = dataset if dataset is not None else make_demo_dataset()
    X = data[FEATURES]
    y = data[TARGET].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    fitted: dict[str, Any] = {}
    probabilities: dict[str, np.ndarray] = {}
    rows: list[dict[str, float | str]] = []
    for name, estimator in _make_models().items():
        estimator.fit(X_train, y_train)
        predicted_probability = positive_class_probability(estimator, X_test)
        predicted_label = (predicted_probability >= 0.5).astype(int)
        fitted[name] = estimator
        probabilities[name] = predicted_probability
        rows.append(
            {
                "Model": name,
                "Precision": precision_score(y_test, predicted_label, zero_division=0),
                "Recall": recall_score(y_test, predicted_label, zero_division=0),
                "F1": f1_score(y_test, predicted_label, zero_division=0),
                "ROC AUC": roc_auc_score(y_test, predicted_probability),
                "Brier score": brier_score_loss(y_test, predicted_probability),
            }
        )

    metrics = pd.DataFrame(rows).sort_values("F1", ascending=False).reset_index(drop=True)
    return ModelSuite(fitted, X_train, X_test, y_train, y_test, probabilities, metrics)


def threshold_summary(
    y_true: pd.Series | np.ndarray,
    probabilities: np.ndarray,
    risk_cutoff: float,
) -> dict[str, float]:
    """Describe the demo approval simulation at a chosen maximum risk cutoff."""
    eligible = probabilities < risk_cutoff
    approval_rate = float(eligible.mean()) if len(eligible) else 0.0
    default_rate = float(np.asarray(y_true)[eligible].mean()) if eligible.any() else 0.0
    return {"approval_rate": approval_rate, "default_rate": default_rate, "approved_count": int(eligible.sum())}


def threshold_curve(
    y_true: pd.Series | np.ndarray,
    probabilities: np.ndarray,
    cutoffs: np.ndarray | None = None,
) -> pd.DataFrame:
    cutoffs = cutoffs if cutoffs is not None else np.arange(0.10, 0.71, 0.05)
    points = []
    for cutoff in cutoffs:
        result = threshold_summary(y_true, probabilities, float(cutoff))
        points.append(
            {
                "Risk cutoff": float(cutoff),
                "Approval rate": result["approval_rate"],
                "Default rate among eligible": result["default_rate"],
            }
        )
    return pd.DataFrame(points)
