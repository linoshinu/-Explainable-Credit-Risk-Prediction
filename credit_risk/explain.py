"""SHAP attributions and plain-language labels for a single demo profile."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .data import FEATURES


def _positive_class_values(values: Any, n_features: int) -> np.ndarray:
    if isinstance(values, list):
        values = values[1] if len(values) > 1 else values[0]
    array = np.asarray(values)
    if array.ndim == 3:
        if array.shape[-1] > 1:
            array = array[..., 1]
        elif array.shape[1] > 1:
            array = array[:, 1, :]
    array = np.asarray(array).reshape(-1, n_features)
    return array[0]


def explain_default_risk(model: Any, row: pd.DataFrame, background: pd.DataFrame) -> np.ndarray:
    """Return SHAP contributions for the model's default class output.

    Tree models use Tree SHAP. Logistic regression is explained in standardized
    feature space with Linear SHAP; displayed labels still use original fields.
    """
    import shap

    steps = getattr(model, "named_steps", {})
    if "model" in steps:
        estimator = steps["model"]
    else:
        estimator = model

    if estimator.__class__.__name__ == "LogisticRegression":
        imputer = steps["imputer"]
        scaler = steps["scale"]
        background_scaled = scaler.transform(imputer.transform(background[FEATURES]))
        row_scaled = scaler.transform(imputer.transform(row[FEATURES]))
        explainer = shap.LinearExplainer(estimator, background_scaled)
        values = explainer.shap_values(row_scaled)
    else:
        explainer = shap.TreeExplainer(estimator)
        values = explainer.shap_values(row[FEATURES])

    return _positive_class_values(values, len(FEATURES))
