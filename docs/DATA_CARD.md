# Synthetic data card

## Purpose

This generated dataset exists only to demonstrate model comparison, threshold analysis, and local feature explanations in a portfolio application. It is not copied from a bank, lender, public credit file, or real applicant record.

## Creation

`credit_risk/data.py` uses NumPy's random generator with seed 42 to simulate 5,000 profiles. It samples income, requested amount, debt-to-income ratio, employment history, credit history length, prior delinquencies, savings, and loan term. A documented logistic formula turns those inputs into a default probability, then samples a binary outcome from that probability.

Because the target is generated from the same inputs given to the classifiers, held-out scores are intentionally illustrative. They do not estimate performance on real populations.

## Fields

| Field | Meaning | Unit |
| --- | --- | --- |
| `annual_income` | Simulated gross annual income | USD |
| `loan_amount` | Simulated requested principal | USD |
| `debt_to_income_ratio` | Simulated monthly debt / monthly gross income | Fraction |
| `employment_years` | Simulated employment history | Years |
| `credit_history_years` | Simulated credit history length | Years |
| `prior_delinquencies` | Simulated prior delinquency count | Count |
| `savings_balance` | Simulated liquid savings | USD |
| `loan_term_months` | Simulated requested loan duration | Months |
| `defaulted` | Sampled synthetic target | 0 or 1 |

## Intended use and limits

- Intended for local learning, demos, and interview discussion of an end-to-end modeling workflow.
- Not for real underwriting, lending decisions, financial advice, or production model validation.
- No personally identifying or protected demographic fields are generated.
- The absence of protected fields does not prove a model is fair; correlated variables and unequal outcomes still require scrutiny.
- No real-world performance, approval, loss, or fairness claim should be inferred from this dataset.
