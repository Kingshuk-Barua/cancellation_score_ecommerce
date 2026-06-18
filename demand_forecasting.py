"""
demand_forecasting.py
=====================
Section 4.1 – Demand Forecasting & Inventory Planning
------------------------------------------------------
Reads the classified monthly time-series produced by time_series_data.ipynb
(product_monthly_sales_timeseries_classified.csv or product_monthly_sales.csv)
and applies class-appropriate forecasting models:

    FX  (Fast-moving, Stable demand)      → SARIMA
    FY  (Fast-moving, Semi-variable)      → Facebook Prophet
    SX  (Slow-moving, Stable demand)      → ARIMA
    SY  (Slow-moving, Semi-variable)      → Facebook Prophet

NX/NY/NZ (Non-moving) and *Z (Highly variable) are excluded from forecasting
because they don't yield reliable predictions.

OUTPUTS
-------
  forecasts/
    fx_sarima_forecasts.csv        – FX class: SARIMA per-product forecasts
    fy_prophet_forecasts.csv       – FY class: Prophet per-product forecasts
    sx_arima_forecasts.csv         – SX class: ARIMA per-product forecasts
    sy_prophet_forecasts.csv       – SY class: Prophet per-product forecasts
    all_forecasts_combined.csv     – All classes merged with metrics
    inventory_suggestions.csv      – Safety stock + EOQ recommendations
  evaluation_summary.csv           – RMSE and MAPE per product

HOW TO RUN (VS Code terminal)
------------------------------
  1.  pip install pandas numpy matplotlib prophet statsmodels scikit-learn tqdm
  2.  Make sure you have already run time_series_data.ipynb (all 4 cells) so
      that product_monthly_sales_timeseries_classified.csv exists in the
      project folder.
  3.  cd into your project folder:
        cd path/to/cancellation_score_ecommerce-main
  4.  Run:
        python demand_forecasting.py
  Optional flags:
        python demand_forecasting.py --forecast_months 6
        python demand_forecasting.py --max_products 50   (limit products for speed)
        python demand_forecasting.py --input your_file.csv

STRUCTURE
---------
  load_data()           – Load & validate the classified monthly time-series
  sarima_forecast()     – ARIMA / SARIMA for FX & SX classes
  prophet_forecast()    – Prophet for FY & SY classes
  compute_metrics()     – RMSE, MAPE on hold-out last 3 months
  compute_safety_stock()– Safety Stock and EOQ per product
  run_class()           – Orchestrate one class group end-to-end
  main()                – Entry point
"""

from __future__ import annotations

import argparse
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

INPUT_CANDIDATES = [
    "product_monthly_sales_timeseries_classified.csv",
    "product_monthly_sales.csv",
    "product_monthly_sales_timeseries.csv",
]

OUTPUT_DIR = Path("forecasts")
EVAL_OUTPUT = Path("evaluation_summary.csv")

# How many future months to forecast
DEFAULT_FORECAST_MONTHS = 3

# Minimum months of history a product must have to be forecasted
MIN_HISTORY_MONTHS = 4

# For safety-stock: z-score for 95% service level
Z_SCORE = 1.65

# Lead time in months assumed for EOQ/Safety-stock
LEAD_TIME_MONTHS = 1

# Holding cost as fraction of item value per month (for EOQ)
HOLDING_COST_RATE = 0.02

# Ordering cost assumed (BRL equivalent per order)
ORDERING_COST = 50.0


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

def load_data(input_file: str | None = None) -> pd.DataFrame:
    """Load the classified monthly time-series from disk."""
    if input_file:
        path = Path(input_file)
    else:
        path = None
        for candidate in INPUT_CANDIDATES:
            if Path(candidate).exists():
                path = Path(candidate)
                break

    if path is None or not path.exists():
        raise FileNotFoundError(
            "Could not find monthly classified time-series. "
            "Run all cells in time_series_data.ipynb first.\n"
            f"Looked for: {INPUT_CANDIDATES}"
        )

    print(f"[load] Reading: {path}")
    df = pd.read_csv(path, parse_dates=["order_month"])
    df["order_month"] = pd.to_datetime(df["order_month"])

    required = {"product_id", "order_month", "quantity_sold"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input file is missing required columns: {missing}")

    # Ensure class column exists (may be absent in base timeseries)
    if "class" not in df.columns:
        print("[warn] 'class' column not found – all products will be treated as FX.")
        df["class"] = "FX"

    df["class"] = df["class"].astype(str).str.upper().str.strip()
    df["quantity_sold"] = pd.to_numeric(df["quantity_sold"], errors="coerce").fillna(0)

    print(f"[load] Shape: {df.shape} | Products: {df['product_id'].nunique()}")
    print(f"[load] Class distribution:\n{df.groupby('product_id')['class'].first().value_counts()}\n")
    return df


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    mask = actual != 0
    if not mask.any():
        return np.nan
    return float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100)


def compute_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    return {"rmse": rmse(actual, predicted), "mape_pct": mape(actual, predicted)}


# ---------------------------------------------------------------------------
# SARIMA / ARIMA Forecasting  (FX and SX classes)
# ---------------------------------------------------------------------------

def sarima_forecast(
    series: pd.Series,
    forecast_months: int,
    seasonal: bool = True,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Fit ARIMA (seasonal=False) or SARIMA (seasonal=True) on a monthly series.
    Returns (in_sample_fitted, future_forecast, metrics_dict).
    """
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from statsmodels.tools.sm_exceptions import ConvergenceWarning

    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

    # Hold out last 3 months for validation
    n_holdout = min(3, len(series) // 3)
    train = series.iloc[:-n_holdout] if n_holdout > 0 else series

    best_aic = np.inf
    best_model = None

    # Grid-search a small parameter space to avoid overfit
    p_values = [0, 1, 2]
    d_values = [0, 1]
    q_values = [0, 1]

    for p in p_values:
        for d in d_values:
            for q in q_values:
                try:
                    if seasonal and len(train) >= 12:
                        # Monthly seasonality with period=12
                        model = SARIMAX(
                            train,
                            order=(p, d, q),
                            seasonal_order=(1, 0, 1, 12),
                            enforce_stationarity=False,
                            enforce_invertibility=False,
                        )
                    else:
                        model = SARIMAX(
                            train,
                            order=(p, d, q),
                            enforce_stationarity=False,
                            enforce_invertibility=False,
                        )
                    result = model.fit(disp=False, maxiter=200)
                    if result.aic < best_aic:
                        best_aic = result.aic
                        best_model = result
                except Exception:
                    continue

    if best_model is None:
        # Fallback: naive forecast (last value)
        naive_val = float(train.mean())
        fitted = np.full(len(train), naive_val)
        forecast = np.full(forecast_months, naive_val)
        metrics = {"rmse": np.nan, "mape_pct": np.nan, "model": "naive_fallback"}
        return fitted, forecast, metrics

    # Refit on full series for final forecast
    try:
        if seasonal and len(series) >= 12:
            final_model = SARIMAX(
                series,
                order=best_model.model.order,
                seasonal_order=best_model.model.seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False,
            ).fit(disp=False, maxiter=200)
        else:
            final_model = SARIMAX(
                series,
                order=best_model.model.order,
                enforce_stationarity=False,
                enforce_invertibility=False,
            ).fit(disp=False, maxiter=200)

        future = final_model.forecast(steps=forecast_months)
        future = np.maximum(0, future.values)
        fitted = np.maximum(0, final_model.fittedvalues.values)
    except Exception:
        naive_val = float(series.mean())
        fitted = np.full(len(series), naive_val)
        future = np.full(forecast_months, naive_val)

    # Evaluate on hold-out
    if n_holdout > 0:
        try:
            holdout_pred = best_model.forecast(steps=n_holdout).values
            holdout_pred = np.maximum(0, holdout_pred)
            actual_holdout = series.iloc[-n_holdout:].values
            metrics = compute_metrics(actual_holdout, holdout_pred)
        except Exception:
            metrics = {"rmse": np.nan, "mape_pct": np.nan}
    else:
        metrics = {"rmse": np.nan, "mape_pct": np.nan}

    order_str = str(best_model.model.order)
    metrics["model"] = f"SARIMA{order_str}" if seasonal else f"ARIMA{order_str}"
    return fitted, future, metrics


# ---------------------------------------------------------------------------
# Prophet Forecasting  (FY and SY classes)
# ---------------------------------------------------------------------------

def prophet_forecast(
    series: pd.Series,
    dates: pd.DatetimeIndex,
    forecast_months: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Fit Facebook Prophet on a monthly series.
    Returns (in_sample_fitted, future_forecast, metrics_dict).
    """
    try:
        from prophet import Prophet
    except ImportError:
        try:
            from fbprophet import Prophet
        except ImportError:
            raise ImportError(
                "Prophet is not installed. Run: pip install prophet"
            )

    # Hold out last 3 months for validation
    n_holdout = min(3, len(series) // 3)

    df_prophet = pd.DataFrame({"ds": dates, "y": series.values})
    train_df = df_prophet.iloc[:-n_holdout] if n_holdout > 0 else df_prophet

    model = Prophet(
        seasonality_mode="multiplicative",
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        changepoint_prior_scale=0.3,
        seasonality_prior_scale=10.0,
    )

    model.fit(train_df)

    # In-sample fitted values
    in_sample_pred = model.predict(df_prophet[["ds"]].iloc[: len(train_df)])
    fitted = np.maximum(0, in_sample_pred["yhat"].values)

    # Hold-out metrics
    if n_holdout > 0:
        holdout_future = df_prophet[["ds"]].iloc[-n_holdout:]
        holdout_pred = model.predict(holdout_future)
        holdout_values = np.maximum(0, holdout_pred["yhat"].values)
        actual_holdout = series.values[-n_holdout:]
        metrics = compute_metrics(actual_holdout, holdout_values)
    else:
        metrics = {"rmse": np.nan, "mape_pct": np.nan}

    # Refit on full data for future forecast
    full_model = Prophet(
        seasonality_mode="multiplicative",
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        changepoint_prior_scale=0.3,
        seasonality_prior_scale=10.0,
    )
    full_model.fit(df_prophet)

    last_date = dates[-1]
    future_dates = pd.DataFrame(
        {
            "ds": pd.date_range(
                start=last_date + pd.DateOffset(months=1),
                periods=forecast_months,
                freq="MS",
            )
        }
    )
    future_pred = full_model.predict(future_dates)
    future_forecast = np.maximum(0, future_pred["yhat"].values)

    metrics["model"] = "Prophet"
    # Extend fitted to cover full series (Prophet fitted only on train above)
    full_fitted_pred = full_model.predict(df_prophet[["ds"]])
    fitted_full = np.maximum(0, full_fitted_pred["yhat"].values)

    return fitted_full, future_forecast, metrics


# ---------------------------------------------------------------------------
# Inventory helpers
# ---------------------------------------------------------------------------

def compute_safety_stock(
    series: pd.Series,
    lead_time: float = LEAD_TIME_MONTHS,
    z: float = Z_SCORE,
) -> float:
    """Safety Stock = Z * std(demand) * sqrt(lead_time)"""
    std_demand = float(series.std(ddof=1)) if len(series) > 1 else 0.0
    return z * std_demand * np.sqrt(lead_time)


def compute_eoq(
    avg_demand_per_month: float,
    ordering_cost: float = ORDERING_COST,
    holding_cost_rate: float = HOLDING_COST_RATE,
    avg_unit_price: float = 1.0,
) -> float:
    """EOQ = sqrt(2 * D * S / H)  where H = holding_cost_rate * unit_price"""
    if avg_demand_per_month <= 0:
        return 0.0
    D = avg_demand_per_month * 12  # annual demand
    H = holding_cost_rate * avg_unit_price
    if H <= 0:
        return D  # order once a year
    return float(np.sqrt(2 * D * ordering_cost / H))


# ---------------------------------------------------------------------------
# Per-class orchestration
# ---------------------------------------------------------------------------

def run_class(
    class_label: str,
    group_df: pd.DataFrame,
    forecast_months: int,
    max_products: int | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run forecasting for one FSN-XYZ class.
    Returns (forecasts_df, metrics_df).
    """
    from tqdm import tqdm

    use_sarima = class_label in ("FX", "SX")
    use_prophet = class_label in ("FY", "SY")
    seasonal = class_label.startswith("F")   # FX gets seasonal SARIMA

    products = group_df["product_id"].unique()
    if max_products:
        products = products[:max_products]

    all_forecasts = []
    all_metrics = []

    print(f"\n[{class_label}] Forecasting {len(products)} products "
          f"({'SARIMA' if use_sarima else 'Prophet'})...")

    for pid in tqdm(products, desc=class_label, unit="product"):
        prod_df = (
            group_df[group_df["product_id"] == pid]
            .sort_values("order_month")
            .reset_index(drop=True)
        )

        if len(prod_df) < MIN_HISTORY_MONTHS:
            continue

        ts = prod_df["quantity_sold"].astype(float)
        dates = pd.DatetimeIndex(prod_df["order_month"])

        avg_price = (
            float(prod_df["avg_unit_price"].mean())
            if "avg_unit_price" in prod_df.columns
            else 1.0
        )

        try:
            if use_sarima:
                _, future_fc, metrics = sarima_forecast(ts, forecast_months, seasonal)
            elif use_prophet:
                _, future_fc, metrics = prophet_forecast(ts, dates, forecast_months)
            else:
                # Fallback: moving average
                window = min(3, len(ts))
                naive_val = float(ts.rolling(window).mean().iloc[-1])
                future_fc = np.full(forecast_months, max(0, naive_val))
                metrics = {"rmse": np.nan, "mape_pct": np.nan, "model": "moving_avg"}
        except Exception as e:
            naive_val = float(ts.mean())
            future_fc = np.full(forecast_months, naive_val)
            metrics = {"rmse": np.nan, "mape_pct": np.nan, "model": f"error:{e}"}

        # Build future date index
        last_date = dates[-1]
        future_dates = pd.date_range(
            start=last_date + pd.DateOffset(months=1),
            periods=forecast_months,
            freq="MS",
        )

        # Safety stock and EOQ
        safety_stock = compute_safety_stock(ts)
        eoq = compute_eoq(float(ts.mean()), avg_unit_price=avg_price)
        reorder_point = float(ts.mean()) * LEAD_TIME_MONTHS + safety_stock

        for i, (fdate, fval) in enumerate(zip(future_dates, future_fc)):
            all_forecasts.append(
                {
                    "product_id": pid,
                    "class": class_label,
                    "forecast_month": fdate,
                    "forecasted_quantity": round(fval, 2),
                    "safety_stock": round(safety_stock, 2),
                    "eoq": round(eoq, 2),
                    "reorder_point": round(reorder_point, 2),
                    "model": metrics.get("model", "unknown"),
                }
            )

        all_metrics.append(
            {
                "product_id": pid,
                "class": class_label,
                "rmse": round(metrics.get("rmse", np.nan), 4),
                "mape_pct": round(metrics.get("mape_pct", np.nan), 4),
                "model": metrics.get("model", "unknown"),
                "history_months": len(ts),
                "avg_monthly_demand": round(float(ts.mean()), 2),
            }
        )

    forecasts_df = pd.DataFrame(all_forecasts)
    metrics_df = pd.DataFrame(all_metrics)
    return forecasts_df, metrics_df


# ---------------------------------------------------------------------------
# Inventory suggestions summary
# ---------------------------------------------------------------------------

def build_inventory_suggestions(combined_forecasts: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate forecasts into a clean inventory planning table.
    Shows the recommended order quantity for the next month per product.
    """
    if combined_forecasts.empty:
        return pd.DataFrame()

    next_month = combined_forecasts["forecast_month"].min()
    next_month_fc = combined_forecasts[combined_forecasts["forecast_month"] == next_month].copy()

    next_month_fc["suggested_order_qty"] = (
        next_month_fc["forecasted_quantity"] + next_month_fc["safety_stock"]
    ).round(0)

    return next_month_fc[
        [
            "product_id",
            "class",
            "forecast_month",
            "forecasted_quantity",
            "safety_stock",
            "eoq",
            "reorder_point",
            "suggested_order_qty",
            "model",
        ]
    ].sort_values(["class", "product_id"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Demand Forecasting – Section 4.1")
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to monthly classified CSV (auto-detected if omitted)",
    )
    parser.add_argument(
        "--forecast_months",
        type=int,
        default=DEFAULT_FORECAST_MONTHS,
        help=f"Number of months to forecast ahead (default: {DEFAULT_FORECAST_MONTHS})",
    )
    parser.add_argument(
        "--max_products",
        type=int,
        default=None,
        help="Limit products per class (useful for quick testing)",
    )
    args = parser.parse_args()

    # --- Load ---
    df = load_data(args.input)

    # --- Filter to forecasting-eligible classes ---
    ELIGIBLE_CLASSES = {"FX", "FY", "SX", "SY"}
    df_eligible = df[df["class"].isin(ELIGIBLE_CLASSES)].copy()
    print(f"[filter] Eligible rows: {len(df_eligible):,} | "
          f"Products: {df_eligible['product_id'].nunique():,}")

    if df_eligible.empty:
        print("[warn] No eligible products found after filtering. "
              "Check that time_series_data.ipynb has been run.")
        return

    # --- Create output directory ---
    OUTPUT_DIR.mkdir(exist_ok=True)

    # --- Run each class ---
    CLASS_MODEL_MAP = {
        "FX": "SARIMA (seasonal)",
        "FY": "Prophet",
        "SX": "ARIMA",
        "SY": "Prophet",
    }

    all_forecasts_list = []
    all_metrics_list = []

    for class_label in ["FX", "FY", "SX", "SY"]:
        class_df = df_eligible[df_eligible["class"] == class_label]
        if class_df.empty:
            print(f"[skip] No products in class {class_label}")
            continue

        fc_df, m_df = run_class(
            class_label=class_label,
            group_df=class_df,
            forecast_months=args.forecast_months,
            max_products=args.max_products,
        )

        if not fc_df.empty:
            out_name = f"{class_label.lower()}_{CLASS_MODEL_MAP[class_label].split()[0].lower()}_forecasts.csv"
            fc_df.to_csv(OUTPUT_DIR / out_name, index=False)
            print(f"[{class_label}] Saved → {OUTPUT_DIR / out_name} "
                  f"({len(fc_df)} rows, {fc_df['product_id'].nunique()} products)")

        all_forecasts_list.append(fc_df)
        all_metrics_list.append(m_df)

    # --- Combine & save ---
    combined_forecasts = pd.concat(all_forecasts_list, ignore_index=True)
    combined_metrics = pd.concat(all_metrics_list, ignore_index=True)

    combined_forecasts.to_csv(OUTPUT_DIR / "all_forecasts_combined.csv", index=False)
    combined_metrics.to_csv(EVAL_OUTPUT, index=False)

    inventory_df = build_inventory_suggestions(combined_forecasts)
    if not inventory_df.empty:
        inventory_df.to_csv(OUTPUT_DIR / "inventory_suggestions.csv", index=False)

    # --- Print summary ---
    print("\n" + "=" * 60)
    print("DEMAND FORECASTING SUMMARY")
    print("=" * 60)
    print(f"Total products forecasted : {combined_forecasts['product_id'].nunique():>6,}")
    print(f"Forecast horizon          : {args.forecast_months} month(s)")
    print()
    print("Results by class:")
    if not combined_metrics.empty:
        summary = (
            combined_metrics.groupby("class")
            .agg(
                products=("product_id", "nunique"),
                avg_rmse=("rmse", "mean"),
                avg_mape_pct=("mape_pct", "mean"),
            )
            .round(3)
        )
        print(summary.to_string())

    print()
    print("Output files:")
    for f in sorted(OUTPUT_DIR.iterdir()):
        print(f"  {f}")
    print(f"  {EVAL_OUTPUT}")
    print("=" * 60)


if __name__ == "__main__":
    main()
