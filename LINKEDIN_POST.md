# LinkedIn post draft

I built **Explainable Credit Risk Prediction**, an interactive decision-support demo that brings model performance, cutoff trade-offs, and individual explanations into one place.

The project compares Logistic Regression, Random Forest, and XGBoost; reviews precision, recall, F1, ROC AUC, and Brier score; and uses SHAP to show which features move a sample score up or down. A threshold lab makes the eligibility-versus-risk trade-off visible instead of hiding it behind one default cutoff.

One important boundary: the app uses **synthetic borrower profiles and simulated outcomes**. The metrics are for demonstrating the workflow, not for claiming real lending performance or fairness. It is a prototype for learning and interview discussion, not a tool for assessing real applicants.

I also wrote an interview guide covering the model choices, threshold logic, SHAP interpretation, and limitations.

Project: https://github.com/linoshinu/-Explainable-Credit-Risk-Prediction

#Python #MachineLearning #ExplainableAI #CreditRisk #Streamlit #ResponsibleAI
