"""
generate_shap_values.py
========================
Prerequisite step for generate_final_cancellation_probability_data.py.

That script expects a shap_values_test.csv file on disk (SHAP values per
feature for the held-out test set) but nothing in this repo produces it.
This script trains the same RandomForestClassifier on the same train/test
split (identical hyperparameters and random_state to
generate_final_cancellation_probability_data.py), computes SHAP values for
the test set with shap.TreeExplainer, and writes shap_values_test.csv.

OUTPUT
------
  shap_values_test.csv   – SHAP values per feature, indexed by the original
                            row index from final_model_dataset.csv (so it
                            aligns with X_test.index used downstream).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "final_model_dataset.csv"
OUTPUT_PATH = BASE_DIR / "shap_values_test.csv"


def main() -> None:
    model_df = pd.read_csv(MODEL_PATH)

    X = model_df.drop(columns=["cancel_flag"])
    y = model_df["cancel_flag"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    model = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    explainer = shap.TreeExplainer(model)
    raw_shap_values = explainer.shap_values(X_test)

    # Binary classifier: keep SHAP values for the positive (cancel) class.
    if isinstance(raw_shap_values, list):
        shap_pos = raw_shap_values[1]
    elif getattr(raw_shap_values, "ndim", 2) == 3:
        shap_pos = raw_shap_values[:, :, 1]
    else:
        shap_pos = raw_shap_values

    shap_df = pd.DataFrame(shap_pos, columns=X_test.columns, index=X_test.index)
    shap_df.to_csv(OUTPUT_PATH)

    print(f"Saved: {OUTPUT_PATH}")
    print(f"Shape: {shap_df.shape}")


if __name__ == "__main__":
    main()
