# Interview walkthrough

## 60-second project story

> I built a small credit-risk decision-support demo to make the modeling trade-offs visible to a non-technical reviewer. It generates a reproducible synthetic borrower dataset, trains Logistic Regression, Random Forest, and XGBoost, and compares precision, recall, F1, ROC AUC, and Brier score on the same held-out split. The app then lets a reviewer change a maximum-risk cutoff and see how the eligible share and observed default rate move together. For one profile, SHAP shows which inputs push the fitted model output up or down. I made the synthetic-data limitation explicit: these results are useful for explaining the workflow, but they are not evidence of real lending performance or fairness.

Adapt this script to what you personally implemented and learned before presenting it.

## A clean demo sequence

1. Start in **Applicant review** and point out the synthetic-data notice.
2. Score the prefilled profile, then change one field such as prior delinquencies.
3. Explain that the SHAP bars describe this model output relative to background profiles, not causes.
4. Open **Model comparison** and describe the fixed 0.50 evaluation cutoff.
5. Open **Threshold lab** and explain why a more inclusive cutoff can raise both eligibility and risk.
6. Finish with **Method & limits** and say what would be required before any real-world use.

## Likely interviewer questions

### Why compare these models?

Logistic Regression is a useful transparent baseline. Random Forest and XGBoost can capture nonlinear patterns and interactions. A model is not selected solely because one metric is higher; the decision depends on the operating objective, calibration, stability, validation population, governance, and interpretability requirements.

### Why do precision and recall matter here?

They summarize different error trade-offs at a specified cutoff. Higher recall means catching more of the synthetic defaults; higher precision means fewer non-default cases among profiles flagged as defaults. The acceptable balance depends on the real decision context and cannot be chosen from this demo alone.

### How did you pick the cutoff?

The app leaves it adjustable and makes the resulting eligibility and default rates visible together. A real cutoff would require a clearly defined business objective, calibrated probabilities, representative validation data, policy constraints, and human review.

### What does SHAP tell you?

It allocates a model output across feature contributions relative to background profiles under the explainer's assumptions. It helps inspect an individual score, but it does not show that changing a feature would cause an outcome to change.

### What would you improve next?

Use an authorized real dataset with a documented sampling process; add temporal validation, leakage and calibration checks, subgroup outcome analysis, monitoring, and a model card; review explainability stability; and set governance and human review before considering any operational use.

## Honest claim boundary

This repository contains a locally runnable Streamlit prototype. It does not claim a production deployment, access to real borrower records, validated lending performance, or a fairness certification.
