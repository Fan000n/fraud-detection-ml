"""
Model builders for the fraud detection project.

Each `build_*` function returns an unfitted scikit-learn estimator, ready to
be .fit() on preprocessed training data.

Design notes:

- Logistic Regression needs scaled input, so it's paired with the "lr"
  preprocessing variant (see src/preprocessing.py).
- Decision Tree and Random Forest don't care about scale, so they use the
  "tree" variant.
- The depth/leaf constraints on DT and RF are baseline regularizers. Proper
  tuning is Step 10.
- `class_weight="balanced"` makes the model penalize errors on the minority
  class (fraud) much more heavily than errors on the majority class. This is
  the first of two imbalance strategies we'll test (the other is SMOTE).
"""

from typing import Optional
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier


def build_logistic_regression(
    random_state: int = 42,
    class_weight: Optional[str] = None,
) -> LogisticRegression:
    """
    Baseline model.

    `max_iter` is bumped from the default 100 because with ~72 features and
    1.1M rows the solver can need more iterations to converge.
    """
    return LogisticRegression(
        max_iter=1000,
        class_weight=class_weight,
        random_state=random_state,
    )


def build_decision_tree(
    random_state: int = 42,
    class_weight: Optional[str] = None,
) -> DecisionTreeClassifier:
    """
    Single decision tree.

    `max_depth=15` and `min_samples_leaf=10` are baseline regularization.
    """
    return DecisionTreeClassifier(
        max_depth=15,
        min_samples_leaf=10,
        class_weight=class_weight,
        random_state=random_state,
    )


def build_random_forest(
    random_state: int = 42,
    class_weight: Optional[str] = None,
) -> RandomForestClassifier:
    """
    Random Forest with 100 trees.
    """
    return RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        min_samples_leaf=10,
        n_jobs=-1,
        class_weight=class_weight,
        random_state=random_state,
    )