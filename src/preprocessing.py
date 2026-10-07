"""
Preprocessing pipeline for the fraud detection project.

Two variants, because different model families need different things:

- build_preprocessor("tree")  -> for Decision Tree, Random Forest.
    One-hot encodes categoricals, passes numerics through unchanged.
    Trees split on thresholds, so feature scaling is irrelevant.

- build_preprocessor("lr")    -> for Logistic Regression.
    One-hot encodes categoricals, applies log1p to `amt` (which is heavily
    right-skewed), then standardizes all numerics. LR is sensitive to both
    skew and scale, so both transforms matter here.

Critical rule: fit the preprocessor on TRAINING DATA ONLY.
Fitting on the full dataset (or on test) leaks information and silently
inflates every downstream metric.
"""

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer


# --- Column groups ---

TARGET = "is_fraud"

CATEGORICAL_COLS = ["gender", "category", "state"]

# `amt` is handled separately in the LR pipeline (log1p + scale).
# The rest are scaled as-is.
NUMERIC_COLS = ["amt", "city_pop", "hour", "day_of_week", "month"]
OTHER_NUMERIC_COLS = ["city_pop", "hour", "day_of_week", "month"]


# --- Pipeline builders ---

def _one_hot_encoder() -> OneHotEncoder:
    """One-hot encoder with a sane default for unseen categories."""
    return OneHotEncoder(
        handle_unknown="ignore",   # unseen categories at inference -> all zeros
        sparse_output=False,       # dense output; feature space is small (~72)
    )


def build_preprocessor(kind: str = "tree") -> ColumnTransformer:
    """
    Return a scikit-learn ColumnTransformer for the given model family.

    Parameters
    ----------
    kind : {"tree", "lr"}
        - "tree": numerics pass through, no scaling.
        - "lr":   `amt` gets log1p + scaling; other numerics get scaling.

    Returns
    -------
    ColumnTransformer
        Unfitted. Call .fit(X_train) then .transform(X_train / X_test).
    """
    if kind not in ("tree", "lr"):
        raise ValueError(f"kind must be 'tree' or 'lr', got {kind!r}")

    cat_pipe = _one_hot_encoder()

    if kind == "tree":
        preprocessor = ColumnTransformer(
            transformers=[
                ("cat", cat_pipe, CATEGORICAL_COLS),
                ("num", "passthrough", NUMERIC_COLS),
            ],
            remainder="drop",
            verbose_feature_names_out=False,
        )
    else:  # kind == "lr"
        # `amt` gets log1p (fix skew) then StandardScaler.
        amt_pipe = Pipeline([
            ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
            ("scale", StandardScaler()),
        ])
        # Everything else just gets scaled.
        other_num_pipe = Pipeline([
            ("scale", StandardScaler()),
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ("cat", cat_pipe, CATEGORICAL_COLS),
                ("amt", amt_pipe, ["amt"]),
                ("num", other_num_pipe, OTHER_NUMERIC_COLS),
            ],
            remainder="drop",
            verbose_feature_names_out=False,
        )

    return preprocessor