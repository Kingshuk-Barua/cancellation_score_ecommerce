from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


RANDOM_STATE = 42
THRESHOLD = 0.5

BASE_DIR = Path(__file__).resolve().parent
CLEANED_PATH = BASE_DIR / "cleaned_orders_dataset.csv"
MODEL_PATH = BASE_DIR / "final_model_dataset.csv"
SHAP_PATH = BASE_DIR / "shap_values_test.csv"
OUTPUT_PATH = BASE_DIR / "final_cancellation_probability_data.csv"

# Mapping from modeled feature names to original cleaned dataset column names.
FEATURE_NAME_MAP: Dict[str, str] = {
    "scaled_approval_time": "approval_time_h",
    "scaled_log_total_price": "total_price",
    "scaled_log_total_freight": "total_freight_value",
    "scaled_freight_to_price_ratio": "freight_to_price_ratio",
    "scaled_payment_installments": "payment_installments",
    "encoded_payment_type": "payment_type",
    "encoded_customer_state": "customer_state",
    "encoded_customer_city": "customer_city",
    "encoded_product_category_name": "product_category_name",
    "encoded_seller_state": "seller_state",
    "encoded_time_of_day": "order_purchase_timestamp",
}


def _classify_order(y_true: int, y_pred: int) -> str:
    if y_true == 1 and y_pred == 1:
        return "TP"
    if y_true == 0 and y_pred == 0:
        return "TN"
    if y_true == 0 and y_pred == 1:
        return "FP"
    return "FN"


def _ship_delay_recommendation(row: pd.Series) -> str:
    purchase_ts = pd.to_datetime(row.get("order_purchase_timestamp"), errors="coerce")
    approved_ts = pd.to_datetime(row.get("order_approved_at"), errors="coerce")

    # Use approval lag as an observable pre-shipment proxy for cancellation window.
    if pd.notna(purchase_ts) and pd.notna(approved_ts):
        lag_hours = (approved_ts - purchase_ts).total_seconds() / 3600.0
        lag_hours = max(lag_hours, 0.0)
    else:
        lag_hours = 24.0

    recommended_hold = int(min(max(np.ceil(lag_hours), 12), 72))
    return (
        f"Delay shipment by about {recommended_hold} hours after approval to reduce revenue leakage "
        "from early post-purchase cancellations."
    )


def _customer_centric_recommendations(row: pd.Series) -> List[str]:
    recs: List[str] = [_ship_delay_recommendation(row)]

    review_score = pd.to_numeric(row.get("review_score"), errors="coerce")
    installments = pd.to_numeric(row.get("payment_installments"), errors="coerce")
    payment_type = str(row.get("payment_type", "Unknown"))

    if pd.notna(review_score) and review_score <= 2:
        recs.append("Create a proactive outreach flow for low-satisfaction customers before dispatch.")
    else:
        recs.append("Send an order confirmation and ETA reassurance message within 2 hours of purchase.")

    if pd.notna(installments) and installments >= 6:
        recs.append("For high-installment orders, validate payment intent with one-click reconfirmation before packing.")
    else:
        recs.append("Enable a lightweight cancellation-reason capture to intervene with targeted support offers.")

    if payment_type in {"voucher", "boleto"}:
        recs.append("For voucher/boleto cohorts, block broad discounts and use targeted credits only after risk checks.")
    else:
        recs.append("Limit aggressive promotions for customers with repeated cancellation behavior.")

    recs.append("Prioritize customer support callbacks for high-risk orders in the first 24 hours.")
    return recs[:5]


def _product_centric_recommendations(row: pd.Series) -> List[str]:
    recs: List[str] = [_ship_delay_recommendation(row)]

    freight = pd.to_numeric(row.get("total_freight_value"), errors="coerce")
    price = pd.to_numeric(row.get("total_price"), errors="coerce")
    weight = pd.to_numeric(row.get("product_weight_g"), errors="coerce")
    photos = pd.to_numeric(row.get("product_photos_qty"), errors="coerce")
    category = str(row.get("product_category_name", "Unknown"))

    if pd.notna(freight) and pd.notna(price) and price > 0 and (freight / price) > 0.3:
        recs.append("Optimize freight-heavy SKUs by adding closer fulfillment stock to reduce shipping burden.")
    else:
        recs.append("Audit product listing promises and align lead times with actual seller handling capacity.")

    if pd.notna(weight) and weight >= 2000:
        recs.append("For heavier products, use sturdier packaging and pre-dispatch checks to reduce cancellation drivers.")
    else:
        recs.append("Bundle lightweight items where possible to improve perceived value and reduce cancel intent.")

    if pd.notna(photos) and photos <= 1:
        recs.append("Improve product content quality with richer images/specs to reduce expectation mismatch.")
    else:
        recs.append("Highlight key product attributes clearly to prevent late-stage buyer hesitation.")

    recs.append(f"Run a category-specific cancellation playbook for '{category}' with weekly defect tracking.")
    return recs[:5]


def _determine_issue_type(shap_row: pd.Series) -> str:
    customer_cols = {
        "encoded_customer_state",
        "encoded_customer_city",
        "encoded_payment_type",
        "scaled_payment_installments",
        "encoded_time_of_day",
    }
    product_cols = {
        "encoded_product_category_name",
        "scaled_log_total_price",
        "scaled_log_total_freight",
        "scaled_freight_to_price_ratio",
    }

    customer_score = float(shap_row.reindex(list(customer_cols), fill_value=0).abs().sum())
    product_score = float(shap_row.reindex(list(product_cols), fill_value=0).abs().sum())

    return "customer" if customer_score >= product_score else "product"


def _top_shap_recommendations(cleaned_row: pd.Series, shap_row: pd.Series) -> str:
    issue_type = _determine_issue_type(shap_row)
    if issue_type == "customer":
        recs = _customer_centric_recommendations(cleaned_row)
    else:
        recs = _product_centric_recommendations(cleaned_row)

    return " | ".join(recs)


def main() -> None:
    cleaned = pd.read_csv(CLEANED_PATH)
    model_df = pd.read_csv(MODEL_PATH)
    shap_df = pd.read_csv(SHAP_PATH, index_col=0)

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

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= THRESHOLD).astype(int)

    test_out = pd.DataFrame(
        {
            "row_index": X_test.index,
            "cancel_flag": y_test.values,
            "pred_cancel_flag": y_pred,
            "cancellation_probability_pct": np.round(y_prob * 100.0, 2),
        }
    )
    test_out["confusion_type"] = [
        _classify_order(int(a), int(b))
        for a, b in zip(test_out["cancel_flag"], test_out["pred_cancel_flag"])
    ]

    # Keep only True Positives and True Negatives as requested.
    test_out = test_out[test_out["confusion_type"].isin(["TP", "TN"])].copy()

    # Merge back with cleaned rows using stable original index.
    cleaned_with_index = cleaned.reset_index().rename(columns={"index": "row_index"})
    out = cleaned_with_index.merge(test_out, on="row_index", how="inner")

    # Join SHAP values using original row index alignment.
    shap_df = shap_df.copy()
    shap_df.index = shap_df.index.astype(int)

    shap_feature_cols = [
        col for col in shap_df.columns if col in X.columns
    ]

    for shap_col in shap_feature_cols:
        target_name = FEATURE_NAME_MAP.get(shap_col, shap_col)
        out[f"shap_{target_name}"] = out["row_index"].map(shap_df[shap_col])

    # Recommendations and SHAP visibility rule:
    # low cancellation probability -> safe, keep SHAP/recommendations null.
    out["risk_status"] = np.where(out["cancellation_probability_pct"] < 50.0, "safe", "at_risk")

    shap_output_cols = [c for c in out.columns if c.startswith("shap_")]
    out["recommendations"] = None

    at_risk_mask = out["risk_status"] == "at_risk"
    if at_risk_mask.any():
        out.loc[at_risk_mask, "recommendations"] = out.loc[at_risk_mask].apply(
            lambda r: _top_shap_recommendations(
                cleaned_row=r,
                shap_row=pd.Series(
                    {col: r.get(f"shap_{FEATURE_NAME_MAP.get(col, col)}", 0.0) for col in shap_feature_cols}
                ),
            ),
            axis=1,
        )

    # Null-out SHAP and recommendations for safe orders.
    out.loc[~at_risk_mask, shap_output_cols] = np.nan
    out.loc[~at_risk_mask, "recommendations"] = pd.NA

    # Final ordering: cleaned dataset columns first, then model outputs and explainability.
    extra_cols = [
        "cancel_flag",
        "pred_cancel_flag",
        "cancellation_probability_pct",
        "confusion_type",
        "risk_status",
    ] + shap_output_cols + ["recommendations"]

    cleaned_cols = list(cleaned.columns)
    final_cols = cleaned_cols + [c for c in extra_cols if c in out.columns and c not in cleaned_cols]

    final_out = out[final_cols].copy()
    final_out.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved: {OUTPUT_PATH}")
    print(f"Rows written: {len(final_out)}")
    print("Confusion split:")
    print(final_out["confusion_type"].value_counts(dropna=False))


if __name__ == "__main__":
    main()
