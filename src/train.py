"""
Model builders for the fraud detection project.

Each `build_*` function returns an unfitted scikit-learn estimator, ready to
be .fit() on preprocessed training data.

Design notes:

- Logistic Regression needs scaled input, so it's paired with the "lr"
    preprocessing variant (see src/preprocessing.py).
- Decision Tree and Random Forest don't care about scale, so they use the
    "tree" variant. They'll be added in Step 8.

`class_weight` is left as None by default. Class imbalance is a separate
experiment in Step 9, not something baked into the baseline.
"""

from sklearn.linear_model import LogisticRegression


def build_logistic_regression(random_state: int = 42) -> LogisticRegression:
    """
    Baseline model.

    `max_iter` is bumped from the default 100 because with ~72 features and
    1.1M rows the solver can need more iterations to converge. `solver="lbfgs"`
    is the default and works well for this size.
    """
    return LogisticRegression(
        max_iter=1000,
        random_state=random_state,
    )