"""
run_pipeline.py
================
Single entry point that runs the full analytics pipeline end-to-end:

  1. Demand Forecasting & Inventory Planning   (demand_forecasting.py)
  2. Cancellation Probability / Risk Scoring   (generate_final_cancellation_probability_data.py)
  3. Seller & Supply Chain Analytics           (seller_analytics.py)

Each module is invoked as its own subprocess so none of its existing logic
is touched -- this script only sequences prerequisite generation and the
three existing scripts in the right order, with the same interpreter that
ran this file.

PREREQUISITES handled automatically if missing
------------------------------------------------
  - product_monthly_sales_timeseries_classified.csv
        Required by demand_forecasting.py. Produced by executing
        time_series_data.ipynb (all 4 cells) via nbconvert.
  - shap_values_test.csv
        Required by generate_final_cancellation_probability_data.py.
        Produced by generate_shap_values.py, which trains the same
        RandomForest model on the same split/seed used downstream and
        computes SHAP values for the held-out test set.

HOW TO RUN
----------
    python run_pipeline.py

Optional flags (forwarded unchanged to the underlying scripts):
    python run_pipeline.py --forecast_months 6 --max_products 50 --clusters 5

Use --skip-notebook / --skip-shap to skip regenerating those prerequisite
files (the pipeline will then fail downstream if they are truly absent),
and --force-notebook / --force-shap to regenerate them even if already
present.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

CLASSIFIED_TIMESERIES = BASE_DIR / "product_monthly_sales_timeseries_classified.csv"
SHAP_VALUES_FILE = BASE_DIR / "shap_values_test.csv"
TIME_SERIES_NOTEBOOK = BASE_DIR / "time_series_data.ipynb"
SHAP_SCRIPT = BASE_DIR / "generate_shap_values.py"
DEMAND_SCRIPT = BASE_DIR / "demand_forecasting.py"
CANCELLATION_SCRIPT = BASE_DIR / "generate_final_cancellation_probability_data.py"
SELLER_SCRIPT = BASE_DIR / "seller_analytics.py"


def _subprocess_env() -> dict:
    """Force UTF-8 stdio so module print statements with unicode (->, emoji)
    don't crash on Windows consoles using a non-UTF8 codepage."""
    import os

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def _run(cmd: list[str], step_name: str) -> None:
    print("\n" + "=" * 70)
    print(f"  {step_name}")
    print("=" * 70)
    start = time.time()
    result = subprocess.run(cmd, cwd=BASE_DIR, env=_subprocess_env())
    elapsed = time.time() - start
    if result.returncode != 0:
        print(f"\n[FAILED] {step_name} exited with code {result.returncode} "
              f"after {elapsed:.1f}s")
        sys.exit(result.returncode)
    print(f"\n[OK] {step_name} completed in {elapsed:.1f}s")


def _jupyter_path_env() -> dict:
    """
    `jupyter <subcommand>` dispatches by looking up a `jupyter-<subcommand>`
    script on PATH, not purely via Python imports. If another Python/conda
    install earlier on PATH ships a broken or binary-incompatible
    jupyter/zmq, nbconvert can crash. Prepend this interpreter's own
    bin/Scripts directories so it resolves to itself first.
    """
    import os

    env = os.environ.copy()
    py_dir = str(Path(sys.executable).resolve().parent)
    scripts_dir = str(Path(sys.executable).resolve().parent / "Scripts")
    library_bin = str(Path(sys.executable).resolve().parent / "Library" / "bin")
    env["PATH"] = os.pathsep.join([py_dir, scripts_dir, library_bin, env.get("PATH", "")])
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def ensure_classified_timeseries(force: bool, skip: bool) -> None:
    if skip:
        print("[skip] Not regenerating product_monthly_sales_timeseries_classified.csv "
              "(--skip-notebook)")
        return
    if CLASSIFIED_TIMESERIES.exists() and not force:
        print(f"[ok] Found existing {CLASSIFIED_TIMESERIES.name}")
        return

    print("\n" + "=" * 70)
    print("  PREREQUISITE: Executing time_series_data.ipynb")
    print("=" * 70)
    start = time.time()
    result = subprocess.run(
        [
            sys.executable, "-m", "jupyter", "nbconvert",
            "--to", "notebook", "--execute", "--inplace",
            str(TIME_SERIES_NOTEBOOK),
        ],
        cwd=BASE_DIR,
        env=_jupyter_path_env(),
    )
    elapsed = time.time() - start
    if result.returncode != 0:
        print(f"\n[FAILED] time_series_data.ipynb execution exited with code "
              f"{result.returncode} after {elapsed:.1f}s")
        sys.exit(result.returncode)
    print(f"\n[OK] time_series_data.ipynb executed in {elapsed:.1f}s")


def ensure_shap_values(force: bool, skip: bool) -> None:
    if skip:
        print("[skip] Not regenerating shap_values_test.csv (--skip-shap)")
        return
    if SHAP_VALUES_FILE.exists() and not force:
        print(f"[ok] Found existing {SHAP_VALUES_FILE.name}")
        return
    _run([sys.executable, str(SHAP_SCRIPT)], "PREREQUISITE: Generating SHAP values")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the full pipeline: demand forecasting -> cancellation "
                     "score -> seller analytics."
    )
    # Prerequisite control
    parser.add_argument("--skip-notebook", action="store_true",
                         help="Don't auto-execute time_series_data.ipynb")
    parser.add_argument("--force-notebook", action="store_true",
                         help="Re-execute time_series_data.ipynb even if its output already exists")
    parser.add_argument("--skip-shap", action="store_true",
                         help="Don't auto-generate shap_values_test.csv")
    parser.add_argument("--force-shap", action="store_true",
                         help="Regenerate shap_values_test.csv even if it already exists")

    # Module selection
    parser.add_argument("--skip-demand", action="store_true", help="Skip demand forecasting")
    parser.add_argument("--skip-cancellation", action="store_true", help="Skip cancellation scoring")
    parser.add_argument("--skip-seller", action="store_true", help="Skip seller analytics")

    # Pass-through options for demand_forecasting.py
    parser.add_argument("--forecast_months", type=int, default=None,
                         help="Forwarded to demand_forecasting.py")
    parser.add_argument("--max_products", type=int, default=None,
                         help="Forwarded to demand_forecasting.py")

    # Pass-through options for seller_analytics.py
    parser.add_argument("--clusters", type=int, default=None,
                         help="Forwarded to seller_analytics.py")
    parser.add_argument("--dispatch_sla", type=float, default=None,
                         help="Forwarded to seller_analytics.py")
    parser.add_argument("--delivery_sla", type=float, default=None,
                         help="Forwarded to seller_analytics.py")

    args = parser.parse_args()

    pipeline_start = time.time()

    print("#" * 70)
    print("#  UNIFIED PIPELINE: Demand Forecasting -> Cancellation Score -> "
          "Seller Analytics")
    print("#" * 70)

    # ---- Step 1: Demand Forecasting -------------------------------------
    if not args.skip_demand:
        ensure_classified_timeseries(force=args.force_notebook, skip=args.skip_notebook)

        demand_cmd = [sys.executable, str(DEMAND_SCRIPT)]
        if args.forecast_months is not None:
            demand_cmd += ["--forecast_months", str(args.forecast_months)]
        if args.max_products is not None:
            demand_cmd += ["--max_products", str(args.max_products)]
        _run(demand_cmd, "MODULE 1/3: Demand Forecasting & Inventory Planning")
    else:
        print("\n[skip] Demand forecasting skipped (--skip-demand)")

    # ---- Step 2: Cancellation Score --------------------------------------
    if not args.skip_cancellation:
        ensure_shap_values(force=args.force_shap, skip=args.skip_shap)
        _run([sys.executable, str(CANCELLATION_SCRIPT)],
             "MODULE 2/3: Cancellation Probability Scoring")
    else:
        print("\n[skip] Cancellation scoring skipped (--skip-cancellation)")

    # ---- Step 3: Seller Analytics ----------------------------------------
    if not args.skip_seller:
        seller_cmd = [sys.executable, str(SELLER_SCRIPT)]
        if args.clusters is not None:
            seller_cmd += ["--clusters", str(args.clusters)]
        if args.dispatch_sla is not None:
            seller_cmd += ["--dispatch_sla", str(args.dispatch_sla)]
        if args.delivery_sla is not None:
            seller_cmd += ["--delivery_sla", str(args.delivery_sla)]
        _run(seller_cmd, "MODULE 3/3: Seller & Supply Chain Analytics")
    else:
        print("\n[skip] Seller analytics skipped (--skip-seller)")

    total_elapsed = time.time() - pipeline_start

    print("\n" + "#" * 70)
    print("#  PIPELINE COMPLETE")
    print("#" * 70)
    print(f"  Total runtime: {total_elapsed/60:.1f} min")
    print("  Output locations:")
    print(f"    Demand forecasting   -> {BASE_DIR / 'forecasts'}/, "
          f"{BASE_DIR / 'evaluation_summary.csv'}")
    print(f"    Cancellation score   -> "
          f"{BASE_DIR / 'final_cancellation_probability_data.csv'}")
    print(f"    Seller analytics     -> {BASE_DIR / 'seller_analytics_output'}/")
    print("#" * 70)


if __name__ == "__main__":
    main()
