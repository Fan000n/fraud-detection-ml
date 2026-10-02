# Fraud Detection ML

A beginner-to-intermediate machine learning project that predicts whether a
financial transaction is fraudulent or legitimate.

> **Status:**  Work in progress. Full documentation will be added as the project progresses.

## Dataset

**Credit Card Transactions Fraud Detection Dataset** (Kartik Shenoy, Kaggle).

- ~1.85 M transactions (train + test combined).
- Binary target: `is_fraud` (1 = fraud, 0 = legitimate).
- ~0.5 % fraud → strongly imbalanced.
- Interpretable features: transaction amount, merchant, category, time, cardholder location, etc.
- Full source and setup instructions: [`data/README.md`](data/README.md).