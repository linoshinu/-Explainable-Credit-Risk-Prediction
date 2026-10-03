"""Deterministic synthetic data for a credit-risk teaching demo."""

from __future__ import annotations

import numpy as np
import pandas as pd

TARGET = "defaulted"
FEATURES = [
    "annual_income",
    "loan_amount",
    "debt_to_income_ratio",
    "employment_years",
    "credit_history_years",
    "prior_delinquencies",
    "savings_balance",
    "loan_term_months",
]

FEATURE_LABELS = {
    "annual_income": "Annual income",
    "loan_amount": "Requested loan amount",
    "debt_to_income_ratio": "Debt-to-income ratio",
    "employment_years": "Employment history",
    "credit_history_years": "Credit history length",
    "prior_delinquencies": "Prior delinquencies",
    "savings_balance": "Savings balance",
    "loan_term_months": "Loan term",
}


def make_demo_dataset(n_samples: int = 5_000, random_state: int = 42) -> pd.DataFrame:
    """Create simulated borrower profiles and a simulated default outcome.

    No real applicant records are used. The outcome is sampled from a known
    synthetic formula so the full demo is reproducible and self-contained.
    """
    rng = np.random.default_rng(random_state)

    income = np.clip(rng.lognormal(np.log(58_000), 0.48, n_samples), 18_000, 240_000)
    loan_to_income = np.clip(rng.lognormal(np.log(0.34), 0.50, n_samples), 0.06, 1.35)
    loan = income * loan_to_income
    dti = np.clip(rng.beta(2.2, 5.2, n_samples), 0.02, 0.92)
    employment = np.clip(rng.gamma(2.0, 3.3, n_samples), 0, 35)
    credit_history = np.clip(rng.gamma(2.4, 3.2, n_samples), 0, 35)
    delinquencies = rng.poisson(0.28 + 0.55 * (dti > 0.48), n_samples).clip(0, 5)
    savings = np.clip(income * rng.beta(2.0, 4.8, n_samples), 0, 180_000)
    term = rng.choice([12, 24, 36, 48, 60], size=n_samples, p=[0.08, 0.16, 0.38, 0.23, 0.15])

    savings_to_income = savings / np.maximum(income, 1)
    log_odds = (
        -2.35
        + 1.55 * dti
        + 1.05 * loan_to_income
        + 0.72 * delinquencies
        - 0.052 * employment
        - 0.024 * credit_history
        - 0.32 * savings_to_income
        + 0.008 * (term - 36)
    )
    default_probability = 1.0 / (1.0 + np.exp(-log_odds))
    defaulted = rng.binomial(1, default_probability)

    return pd.DataFrame(
        {
            "annual_income": np.round(income, 2),
            "loan_amount": np.round(loan, 2),
            "debt_to_income_ratio": np.round(dti, 4),
            "employment_years": np.round(employment, 1),
            "credit_history_years": np.round(credit_history, 1),
            "prior_delinquencies": delinquencies.astype(int),
            "savings_balance": np.round(savings, 2),
            "loan_term_months": term.astype(int),
            TARGET: defaulted.astype(int),
        }
    )
