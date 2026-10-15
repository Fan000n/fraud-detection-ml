"""
Model builders for the fraud detection project.

Each `build_*` function returns an unfitted scikit-learn estimator, ready to
be .fit() on preprocessed training data.

Design notes:

- Logistic Regression needs scaled input, so it's paired with the "lr"
  preprocessing variant (see src/preprocessing.py).
- Decision Tree and Random Forest don't care about scale, so they use the
  "tree" variant.
- The depth/leaf constraints on DT are baseline regularizers. Without them a
  tree on 1.1M rows grows effectively unbounded, which is both slow and a
  textbook overfitting demo. Proper tuning is Step 10.
- Random Forest uses `n_jobs=-1` to parallelize across cores. On a T480S this
  is the difference between ~5 min and ~30 min for a fit.
- `class_weight` is left as None everywhere by default. Class imbalance is a
  separate experiment in Step 9, not something baked into the baselines.
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier


def build_logistic_regression(random_state: int = 42) -> LogisticRegression:
    """
    Baseline model.

    `max_iter` is bumped from the default 100 because with ~72 features and
    1.1M rows the solver can need more iterations to converge.
    """
    return LogisticRegression(
        max_iter=1000,
        random_state=random_state,
    )


def build_decision_tree(random_state: int = 42) -> DecisionTreeClassifier:
    """
    Single decision tree.

    `max_depth=15` and `min_samples_leaf=10` are baseline regularization.
    Without them the tree grows essentially unbounded on 1.1M rows - slow to
    fit and a classic overfitting case. Both will be revisited in Step 10.
    """
    return DecisionTreeClassifier(
        max_depth=15,
        min_samples_leaf=10,
        random_state=random_state,
    )


def build_random_forest(random_state: int = 42) -> RandomForestClassifier:
    """
    Random Forest with 100 trees.

    `max_depth=15` and `min_samples_leaf=10` keep individual trees in check,
    same reasoning as the single tree. `n_jobs=-1` uses all cores.
    """
    return RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        min_samples_leaf=10,
        n_jobs=-1,
        random_state=random_state,
    )