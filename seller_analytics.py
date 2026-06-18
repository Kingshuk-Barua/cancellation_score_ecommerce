"""
seller_analytics.py
===================
Section 4.2 – Seller & Supply Chain Analytics
----------------------------------------------
Implements the full analytics pipeline:

  1.  Feature Engineering
        • Order-to-Dispatch time   (approved → carrier pickup)
        • Dispatch-to-Delivery time (carrier pickup → customer receipt)
        • On-time delivery flag    (delivered before estimated date)
        • Cancellation frequency per seller
        • Normalised review scores
        • Revenue / volume metrics

  2.  Weighted Composite Seller Score  (0–100)
        Combines delivery speed, review quality, cancellation rate,
        and on-time reliability into one interpretable KPI.

  3.  PCA – latent composite score (alternative ranking axis)
        Reduces the 6 raw KPI dimensions to a single PC-1 score that
        explains the most variance across sellers.

  4.  K-Means Clustering (seller behaviour groups)
        Groups sellers into behavioural personas
        (e.g. "Elite", "Reliable", "At-Risk", "Problematic").

  5.  ABC–XYZ Supplier Classification
        ABC by cumulative revenue contribution (Pareto).
        XYZ by month-to-month order volume variability (CV).

  6.  Regional (State-level) Bottleneck Detection
        Identifies which seller states have the worst delivery performance,
        highest cancellation rates, and longest lead-times.

  7.  Supply-Chain Bottleneck Flags
        Flags individual orders/sellers breaching SLA thresholds.

OUTPUTS  (all written to  seller_analytics_output/ )
-------
  seller_feature_matrix.csv          – Raw engineered features per seller
  seller_scores.csv                  – Weighted score + PCA score + ABC-XYZ
  seller_clusters.csv                – Cluster labels + persona names
  seller_ranking_dashboard.csv       – Final merged dashboard table
  regional_bottlenecks.csv           – State-level aggregated KPIs
  bottleneck_orders.csv              – Individual orders breaching SLA
  plots/
    seller_score_distribution.html
    cluster_scatter.html
    regional_heatmap.html
    kpi_radar_by_cluster.html
    bottleneck_map.html
    abc_xyz_matrix.html

HOW TO RUN (VS Code terminal)
------------------------------
  1. pip install pandas numpy scikit-learn plotly tqdm
  2. cd path/to/cancellation_score_ecommerce-main
  3. python seller_analytics.py

Optional flags:
  --input   path/to/merged_orders_clean.csv   (auto-detected by default)
  --output  my_output_dir                     (default: seller_analytics_output)
  --clusters  4                               (number of K-Means clusters, default 4)
  --dispatch_sla  3.0                         (days threshold for dispatch SLA breach)
  --delivery_sla  15.0                        (days threshold for delivery SLA breach)
"""

from __future__ import annotations

import argparse
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# Configuration / weight matrix
# ──────────────────────────────────────────────────────────────────────────────

# Weights for the composite seller score (must sum to 1.0)
SCORE_WEIGHTS = {
    "norm_review_score":        0.30,   # Customer satisfaction is king
    "norm_ontime_rate":         0.25,   # Did the seller deliver on time?
    "norm_cancellation_rate":   0.20,   # How often does the seller cancel?
    "norm_dispatch_speed":      0.15,   # How fast does the seller dispatch?
    "norm_delivery_speed":      0.10,   # How fast does the carrier deliver?
}

# SLA thresholds (days)  – orders breaching these are flagged as bottlenecks
DEFAULT_DISPATCH_SLA = 3.0    # seller should dispatch within 3 days of approval
DEFAULT_DELIVERY_SLA = 15.0   # customer should receive within 15 days of purchase

# K-Means
DEFAULT_N_CLUSTERS = 4

# Minimum orders a seller must have to be included in scoring
MIN_ORDERS_FOR_SCORING = 5

# ABC cutoffs (cumulative revenue share)
ABC_A_THRESHOLD = 0.80
ABC_B_THRESHOLD = 0.95

# XYZ CV thresholds (coefficient of variation of monthly order counts)
XYZ_X_THRESHOLD = 0.25   # stable
XYZ_Y_THRESHOLD = 0.50   # moderate

INPUT_CANDIDATES = [
    "merged_orders_clean.csv",
    "cleaned_orders_dataset.csv",
]

# ──────────────────────────────────────────────────────────────────────────────
# 1. Data loading
# ──────────────────────────────────────────────────────────────────────────────

def load_data(input_file: str | None) -> pd.DataFrame:
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
            f"Cannot find input file. Tried: {INPUT_CANDIDATES}\n"
            "Run analysis.ipynb (Cell 1) first to generate merged_orders_clean.csv"
        )

    print(f"[load] Reading: {path}")

    DATE_COLS = [
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ]

    df = pd.read_csv(path, parse_dates=DATE_COLS, low_memory=False)

    # Normalise status
    df["order_status"] = df["order_status"].astype(str).str.strip().str.lower()
    df["cancel_flag"] = df["order_status"].isin(["canceled", "unavailable"]).astype(int)

    print(f"[load] Shape: {df.shape} | Sellers: {df['seller_id'].nunique()} | "
          f"States: {df['seller_state'].nunique()}")
    return df


# ──────────────────────────────────────────────────────────────────────────────
# 2. Feature Engineering
# ──────────────────────────────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add time-difference and derived columns to the order-level dataframe."""
    out = df.copy()

    # ── Time deltas (in days) ──────────────────────────────────────────────
    out["approval_to_dispatch_days"] = (
        (out["order_delivered_carrier_date"] - out["order_approved_at"])
        .dt.total_seconds() / 86_400
    )
    out["dispatch_to_delivery_days"] = (
        (out["order_delivered_customer_date"] - out["order_delivered_carrier_date"])
        .dt.total_seconds() / 86_400
    )
    out["purchase_to_delivery_days"] = (
        (out["order_delivered_customer_date"] - out["order_purchase_timestamp"])
        .dt.total_seconds() / 86_400
    )
    out["estimated_minus_actual_days"] = (
        (out["order_estimated_delivery_date"] - out["order_delivered_customer_date"])
        .dt.total_seconds() / 86_400
    )

    # ── On-time flag (positive = delivered before estimated date) ──────────
    out["delivered_on_time"] = (out["estimated_minus_actual_days"] >= 0).astype(float)
    # Only valid for delivered orders
    delivered_mask = out["order_status"] == "delivered"
    out.loc[~delivered_mask, "delivered_on_time"] = np.nan

    # ── Clip negative time-deltas (data quality) ───────────────────────────
    for col in ["approval_to_dispatch_days", "dispatch_to_delivery_days",
                "purchase_to_delivery_days"]:
        out[col] = out[col].clip(lower=0)

    return out


def aggregate_seller_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate order-level features to seller-level KPIs.
    Returns one row per seller.
    """
    print("[features] Aggregating seller-level KPIs...")

    grp = df.groupby("seller_id")

    seller_stats = grp.agg(
        total_orders            = ("order_id",                   "count"),
        total_revenue           = ("total_price",                "sum"),
        avg_order_value         = ("total_price",                "mean"),
        cancellations           = ("cancel_flag",                "sum"),
        avg_review_score        = ("review_score",               "mean"),
        avg_dispatch_days       = ("approval_to_dispatch_days",  "mean"),
        avg_delivery_days       = ("dispatch_to_delivery_days",  "mean"),
        avg_total_delivery_days = ("purchase_to_delivery_days",  "mean"),
        ontime_deliveries       = ("delivered_on_time",          "sum"),
        total_with_ontime_data  = ("delivered_on_time",          "count"),
        avg_freight_value       = ("total_freight_value",        "mean"),
        seller_state            = ("seller_state",               "first"),
        unique_categories       = ("product_category_name",      "nunique"),
        unique_customers        = ("customer_id",                "nunique"),
    ).reset_index()

    seller_stats["cancellation_rate"] = (
        seller_stats["cancellations"] / seller_stats["total_orders"]
    ).fillna(0)

    seller_stats["ontime_delivery_rate"] = (
        seller_stats["ontime_deliveries"] / seller_stats["total_with_ontime_data"]
    ).fillna(0)

    # Filter to sellers with enough history for reliable scoring
    before = len(seller_stats)
    seller_stats = seller_stats[seller_stats["total_orders"] >= MIN_ORDERS_FOR_SCORING].copy()
    print(f"[features] Kept {len(seller_stats)}/{before} sellers "
          f"(≥{MIN_ORDERS_FOR_SCORING} orders)")

    return seller_stats.reset_index(drop=True)


def add_monthly_order_variability(df: pd.DataFrame, seller_stats: pd.DataFrame) -> pd.DataFrame:
    """
    Compute month-to-month order count variability per seller (used for XYZ).
    Adds 'order_volume_cv' column to seller_stats.
    """
    df2 = df.copy()
    df2["order_month"] = df2["order_purchase_timestamp"].dt.to_period("M")
    monthly = (
        df2.groupby(["seller_id", "order_month"])["order_id"]
        .count()
        .reset_index(name="monthly_orders")
    )
    cv_df = (
        monthly.groupby("seller_id")["monthly_orders"]
        .agg(lambda s: (s.std(ddof=0) / s.mean()) if s.mean() > 0 else 0)
        .reset_index(name="order_volume_cv")
    )
    return seller_stats.merge(cv_df, on="seller_id", how="left")


# ──────────────────────────────────────────────────────────────────────────────
# 3. Normalisation helpers
# ──────────────────────────────────────────────────────────────────────────────

def _norm_0_1(series: pd.Series) -> pd.Series:
    """Min-max normalise to [0, 1]."""
    lo, hi = series.min(), series.max()
    if hi == lo:
        return pd.Series(0.5, index=series.index)
    return (series - lo) / (hi - lo)


def normalise_kpis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create normalised [0,1] columns where 1 = best for the seller.
    For metrics where lower raw = better (dispatch/delivery days, cancellation),
    we invert so that 1 = fastest/lowest.
    """
    out = df.copy()

    # Higher is better → normalise directly
    out["norm_review_score"]      = _norm_0_1(out["avg_review_score"].fillna(0))
    out["norm_ontime_rate"]       = _norm_0_1(out["ontime_delivery_rate"].fillna(0))

    # Lower is better → invert after normalisation
    out["norm_cancellation_rate"] = 1 - _norm_0_1(out["cancellation_rate"].fillna(0))
    out["norm_dispatch_speed"]    = 1 - _norm_0_1(out["avg_dispatch_days"].fillna(
                                        out["avg_dispatch_days"].median()))
    out["norm_delivery_speed"]    = 1 - _norm_0_1(out["avg_delivery_days"].fillna(
                                        out["avg_delivery_days"].median()))
    return out


# ──────────────────────────────────────────────────────────────────────────────
# 4. Weighted Composite Score
# ──────────────────────────────────────────────────────────────────────────────

def compute_weighted_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute a 0–100 composite seller score using pre-defined weights.
    Formula:
        score = 100 × Σ(weight_i × norm_kpi_i)
    """
    out = df.copy()
    score = pd.Series(0.0, index=out.index)
    for col, w in SCORE_WEIGHTS.items():
        score += w * out[col].fillna(0)
    out["composite_score"] = (score * 100).round(2)

    # Letter grade tiers
    bins   = [-1, 40, 60, 75, 88, 101]
    labels = ["E – Critical", "D – Poor", "C – Average", "B – Good", "A – Excellent"]
    out["score_grade"] = pd.cut(out["composite_score"], bins=bins, labels=labels)

    print(f"[score] Composite score range: "
          f"{out['composite_score'].min():.1f} – {out['composite_score'].max():.1f} "
          f"| Mean: {out['composite_score'].mean():.1f}")
    return out


# ──────────────────────────────────────────────────────────────────────────────
# 5. PCA – latent composite score
# ──────────────────────────────────────────────────────────────────────────────

def compute_pca_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reduce the 5 normalised KPI dimensions to a single PC-1 score.
    PC-1 explains the most variance → acts as an unbiased composite ranking.
    """
    feature_cols = list(SCORE_WEIGHTS.keys())
    X = df[feature_cols].fillna(0).values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=len(feature_cols))
    pca.fit(X_scaled)
    explained = pca.explained_variance_ratio_

    pc1_scores = pca.transform(X_scaled)[:, 0]
    # Flip if needed so higher = better seller
    if np.corrcoef(pc1_scores, df["composite_score"].values)[0, 1] < 0:
        pc1_scores = -pc1_scores

    # Re-scale to 0-100 for interpretability
    pc1_min, pc1_max = pc1_scores.min(), pc1_scores.max()
    if pc1_max > pc1_min:
        pc1_scaled = (pc1_scores - pc1_min) / (pc1_max - pc1_min) * 100
    else:
        pc1_scaled = np.full_like(pc1_scores, 50.0)

    out = df.copy()
    out["pca_score"] = pc1_scaled.round(2)

    print(f"[PCA] Variance explained by PC1: {explained[0]*100:.1f}% | "
          f"PC2: {explained[1]*100:.1f}%")
    print(f"[PCA] Feature loadings on PC1:")
    for feat, loading in zip(feature_cols, pca.components_[0]):
        print(f"      {feat:<30} {loading:+.3f}")

    # Store loadings for reference
    out.attrs["pca_loadings"] = dict(zip(feature_cols, pca.components_[0].tolist()))
    out.attrs["pca_variance"] = explained.tolist()
    return out


# ──────────────────────────────────────────────────────────────────────────────
# 6. K-Means Clustering
# ──────────────────────────────────────────────────────────────────────────────

CLUSTER_PERSONAS = {
    # Determined after fitting by matching to score quartile
    # Will be assigned dynamically based on mean composite score per cluster
    0: "Cluster-0",
    1: "Cluster-1",
    2: "Cluster-2",
    3: "Cluster-3",
}

PERSONA_NAMES_BY_RANK = ["⚠️  Problematic", "🔴 At-Risk", "🟡 Reliable", "🏆 Elite"]


def cluster_sellers(df: pd.DataFrame, n_clusters: int) -> pd.DataFrame:
    """
    K-Means cluster on the normalised KPI space.
    Assigns human-readable persona names ranked by composite score.
    """
    feature_cols = list(SCORE_WEIGHTS.keys())
    X = df[feature_cols].fillna(0).values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=20, max_iter=500)
    labels = kmeans.fit_predict(X_scaled)

    out = df.copy()
    out["cluster_id"] = labels

    # Rank clusters by mean composite score → assign personas
    cluster_means = (
        out.groupby("cluster_id")["composite_score"].mean().sort_values()
    )
    rank_map = {cid: i for i, cid in enumerate(cluster_means.index)}

    persona_map: dict[int, str] = {}
    for cid, rank in rank_map.items():
        idx = min(rank, len(PERSONA_NAMES_BY_RANK) - 1)
        persona_map[cid] = PERSONA_NAMES_BY_RANK[idx]

    out["seller_persona"] = out["cluster_id"].map(persona_map)

    # Cluster-level summary
    print(f"\n[cluster] K-Means with {n_clusters} clusters:")
    summary = out.groupby(["cluster_id", "seller_persona"]).agg(
        sellers      = ("seller_id", "count"),
        avg_score    = ("composite_score", "mean"),
        avg_review   = ("avg_review_score", "mean"),
        avg_cancel   = ("cancellation_rate", "mean"),
        avg_dispatch = ("avg_dispatch_days", "mean"),
    ).round(3)
    print(summary.to_string())

    return out


# ──────────────────────────────────────────────────────────────────────────────
# 7. ABC–XYZ Classification
# ──────────────────────────────────────────────────────────────────────────────

def abc_classify(revenue_series: pd.Series) -> pd.Series:
    """Pareto ABC: A = top 80% revenue, B = next 15%, C = remaining 5%."""
    sorted_rev = revenue_series.sort_values(ascending=False)
    total = float(sorted_rev.sum())
    if total <= 0:
        return pd.Series("C", index=revenue_series.index, dtype=str)
    cum_share = sorted_rev.cumsum() / total
    abc = pd.Series("C", index=sorted_rev.index, dtype=str)
    abc[cum_share <= ABC_A_THRESHOLD] = "A"
    abc[(cum_share > ABC_A_THRESHOLD) & (cum_share <= ABC_B_THRESHOLD)] = "B"
    return abc.reindex(revenue_series.index)


def xyz_classify(cv_series: pd.Series) -> pd.Series:
    """XYZ: X = stable (CV ≤ 0.25), Y = moderate, Z = erratic."""
    xyz = pd.Series("Z", index=cv_series.index, dtype=str)
    xyz[cv_series <= XYZ_X_THRESHOLD] = "X"
    xyz[(cv_series > XYZ_X_THRESHOLD) & (cv_series <= XYZ_Y_THRESHOLD)] = "Y"
    return xyz


def classify_abc_xyz(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["abc_class"] = abc_classify(out["total_revenue"])
    out["xyz_class"] = xyz_classify(out["order_volume_cv"].fillna(1.0))
    out["abc_xyz"]   = out["abc_class"] + out["xyz_class"]

    print("\n[ABC-XYZ] Distribution:")
    print(out["abc_xyz"].value_counts().to_string())
    return out


# ──────────────────────────────────────────────────────────────────────────────
# 8. Regional Bottleneck Analysis
# ──────────────────────────────────────────────────────────────────────────────

def regional_analysis(df_orders: pd.DataFrame, seller_scores: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate KPIs at seller_state level.
    Flags states that are in the worst quartile on 2+ dimensions as bottlenecks.
    """
    print("\n[regional] Computing state-level bottleneck metrics...")

    df2 = df_orders.copy()
    df2["order_month"] = df2["order_purchase_timestamp"].dt.to_period("M").astype(str)

    regional = df2.groupby("seller_state").agg(
        total_orders          = ("order_id",                   "count"),
        total_revenue         = ("total_price",                "sum"),
        avg_review_score      = ("review_score",               "mean"),
        cancellation_rate     = ("cancel_flag",                "mean"),
        avg_dispatch_days     = ("approval_to_dispatch_days",  "mean"),
        avg_delivery_days     = ("dispatch_to_delivery_days",  "mean"),
        avg_total_days        = ("purchase_to_delivery_days",  "mean"),
        ontime_rate           = ("delivered_on_time",          "mean"),
        unique_sellers        = ("seller_id",                  "nunique"),
    ).reset_index()

    # Bottleneck flag: worst 25% on ≥ 2 of these dimensions
    dims = {
        "high_cancel":        regional["cancellation_rate"] >= regional["cancellation_rate"].quantile(0.75),
        "slow_dispatch":      regional["avg_dispatch_days"] >= regional["avg_dispatch_days"].quantile(0.75),
        "slow_delivery":      regional["avg_delivery_days"] >= regional["avg_delivery_days"].quantile(0.75),
        "low_review":         regional["avg_review_score"]  <= regional["avg_review_score"].quantile(0.25),
        "low_ontime":         regional["ontime_rate"]        <= regional["ontime_rate"].quantile(0.25),
    }
    bottleneck_score = sum(v.astype(int) for v in dims.values())
    regional["bottleneck_signals"] = bottleneck_score
    regional["is_bottleneck"]      = (bottleneck_score >= 2).astype(int)

    # Severity label
    def severity(n):
        if n >= 4: return "🔴 Critical"
        if n >= 2: return "🟠 At-Risk"
        return "🟢 Healthy"

    regional["region_severity"] = regional["bottleneck_signals"].apply(severity)

    bottleneck_states = regional[regional["is_bottleneck"] == 1]["seller_state"].tolist()
    print(f"[regional] Bottleneck states ({len(bottleneck_states)}): {bottleneck_states}")
    return regional.sort_values("bottleneck_signals", ascending=False).reset_index(drop=True)


# ──────────────────────────────────────────────────────────────────────────────
# 9. Order-level SLA breach detection
# ──────────────────────────────────────────────────────────────────────────────

def detect_bottleneck_orders(
    df: pd.DataFrame,
    dispatch_sla: float,
    delivery_sla: float,
) -> pd.DataFrame:
    """
    Flag individual orders that breach dispatch or delivery SLA thresholds.
    These are the specific transactions pulling seller scores down.
    """
    delivered = df[df["order_status"] == "delivered"].copy()

    delivered["dispatch_breach"]  = (delivered["approval_to_dispatch_days"] > dispatch_sla).astype(int)
    delivered["delivery_breach"]  = (delivered["purchase_to_delivery_days"] > delivery_sla).astype(int)
    delivered["any_breach"]       = ((delivered["dispatch_breach"] == 1) |
                                     (delivered["delivery_breach"] == 1)).astype(int)

    breached = delivered[delivered["any_breach"] == 1][
        [
            "order_id", "seller_id", "seller_state",
            "order_purchase_timestamp",
            "approval_to_dispatch_days",
            "purchase_to_delivery_days",
            "dispatch_breach", "delivery_breach",
            "review_score", "cancel_flag",
            "total_price",
        ]
    ].copy()

    print(f"\n[SLA] Dispatch SLA ({dispatch_sla}d) breaches: "
          f"{(delivered['dispatch_breach']==1).sum():,} "
          f"({(delivered['dispatch_breach']==1).mean()*100:.1f}%)")
    print(f"[SLA] Delivery SLA ({delivery_sla}d) breaches: "
          f"{(delivered['delivery_breach']==1).sum():,} "
          f"({(delivered['delivery_breach']==1).mean()*100:.1f}%)")
    return breached.sort_values("purchase_to_delivery_days", ascending=False).reset_index(drop=True)


# ──────────────────────────────────────────────────────────────────────────────
# 10. Plotly Visualisations
# ──────────────────────────────────────────────────────────────────────────────

def build_plots(
    seller_scores: pd.DataFrame,
    regional: pd.DataFrame,
    bottleneck_orders: pd.DataFrame,
    plots_dir: Path,
) -> None:
    try:
        import plotly.express as px
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        print("[plots] plotly not installed – skipping visualisations. "
              "Run: pip install plotly")
        return

    plots_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[plots] Writing charts to {plots_dir}/")

    # ── 1. Score distribution by grade ───────────────────────────────────
    fig1 = px.histogram(
        seller_scores,
        x="composite_score",
        color="score_grade",
        nbins=40,
        title="Seller Score Distribution",
        labels={"composite_score": "Composite Score (0–100)", "count": "Sellers"},
        color_discrete_sequence=px.colors.qualitative.Safe,
    )
    fig1.update_layout(barmode="overlay", bargap=0.05)
    fig1.write_html(plots_dir / "seller_score_distribution.html")

    # ── 2. Cluster scatter: PCA score vs composite score ─────────────────
    fig2 = px.scatter(
        seller_scores,
        x="composite_score",
        y="pca_score",
        color="seller_persona",
        hover_data=["seller_id", "seller_state", "total_orders",
                    "avg_review_score", "cancellation_rate"],
        title="Seller Clusters: Composite Score vs PCA Score",
        labels={"composite_score": "Weighted Score (0–100)",
                "pca_score": "PCA Score (0–100)"},
        symbol="abc_class",
        opacity=0.75,
    )
    fig2.write_html(plots_dir / "cluster_scatter.html")

    # ── 3. Regional KPI heatmap ───────────────────────────────────────────
    kpi_cols = [
        "cancellation_rate", "avg_dispatch_days",
        "avg_delivery_days", "avg_review_score", "ontime_rate",
    ]
    regional_norm = regional.copy()
    for col in kpi_cols:
        lo = regional_norm[col].min()
        hi = regional_norm[col].max()
        regional_norm[col] = (regional_norm[col] - lo) / (hi - lo + 1e-9)

    heat_data = regional_norm.set_index("seller_state")[kpi_cols].T
    fig3 = go.Figure(
        data=go.Heatmap(
            z=heat_data.values,
            x=heat_data.columns.tolist(),
            y=kpi_cols,
            colorscale="RdYlGn_r",
            colorbar=dict(title="Normalised<br>value"),
        )
    )
    fig3.update_layout(
        title="Regional KPI Heatmap (Red = Worse, Green = Better)",
        xaxis_title="Seller State",
        yaxis_title="KPI",
        height=450,
    )
    fig3.write_html(plots_dir / "regional_heatmap.html")

    # ── 4. Radar chart – KPI profile by cluster ───────────────────────────
    radar_features = [
        "norm_review_score", "norm_ontime_rate",
        "norm_cancellation_rate", "norm_dispatch_speed", "norm_delivery_speed",
    ]
    radar_labels = [
        "Review Score", "On-time Rate",
        "Low Cancellation", "Fast Dispatch", "Fast Delivery",
    ]
    cluster_means = seller_scores.groupby("seller_persona")[radar_features].mean()

    fig4 = go.Figure()
    colors = ["#e74c3c", "#e67e22", "#2ecc71", "#2980b9"]
    for i, (persona, row) in enumerate(cluster_means.iterrows()):
        vals = row.values.tolist()
        vals += [vals[0]]  # close the radar loop
        labs = radar_labels + [radar_labels[0]]
        fig4.add_trace(go.Scatterpolar(
            r=vals, theta=labs, fill="toself",
            name=persona,
            line=dict(color=colors[i % len(colors)]),
        ))
    fig4.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        title="KPI Radar by Seller Persona",
        showlegend=True,
    )
    fig4.write_html(plots_dir / "kpi_radar_by_cluster.html")

    # ── 5. Regional bottleneck bar chart ─────────────────────────────────
    fig5 = px.bar(
        regional.sort_values("bottleneck_signals", ascending=False).head(20),
        x="seller_state",
        y="bottleneck_signals",
        color="region_severity",
        text="bottleneck_signals",
        title="Top States by Bottleneck Signals (higher = worse)",
        labels={"seller_state": "State", "bottleneck_signals": "Bottleneck Signals"},
        color_discrete_map={
            "🔴 Critical": "#e74c3c",
            "🟠 At-Risk":  "#e67e22",
            "🟢 Healthy":  "#2ecc71",
        },
    )
    fig5.update_traces(textposition="outside")
    fig5.write_html(plots_dir / "bottleneck_map.html")

    # ── 6. ABC-XYZ matrix bubble chart ───────────────────────────────────
    abc_xyz_summary = (
        seller_scores.groupby("abc_xyz").agg(
            sellers      = ("seller_id",         "count"),
            avg_score    = ("composite_score",   "mean"),
            total_revenue= ("total_revenue",     "sum"),
        ).reset_index()
    )
    abc_xyz_summary["abc_class"] = abc_xyz_summary["abc_xyz"].str[0]
    abc_xyz_summary["xyz_class"] = abc_xyz_summary["abc_xyz"].str[1]

    fig6 = px.scatter(
        abc_xyz_summary,
        x="xyz_class", y="abc_class",
        size="total_revenue",
        color="avg_score",
        text="abc_xyz",
        title="ABC–XYZ Supplier Matrix (bubble size = revenue)",
        labels={"xyz_class": "XYZ (Demand Variability)",
                "abc_class": "ABC (Revenue Tier)",
                "avg_score": "Avg Seller Score"},
        color_continuous_scale="RdYlGn",
        size_max=80,
    )
    fig6.update_traces(textposition="top center")
    fig6.write_html(plots_dir / "abc_xyz_matrix.html")

    print(f"[plots] ✓ 6 interactive charts saved.")


# ──────────────────────────────────────────────────────────────────────────────
# 11. Final merged dashboard table
# ──────────────────────────────────────────────────────────────────────────────

def build_dashboard_table(seller_scores: pd.DataFrame) -> pd.DataFrame:
    """
    The single table that powers a Seller Ranking Dashboard.
    Includes rank, score, grade, persona, ABC-XYZ, and all KPIs.
    """
    dashboard = seller_scores.copy()
    dashboard["rank"] = dashboard["composite_score"].rank(ascending=False, method="min").astype(int)
    dashboard = dashboard.sort_values("rank")

    cols = [
        "rank", "seller_id", "seller_state", "seller_persona",
        "composite_score", "pca_score", "score_grade",
        "abc_class", "xyz_class", "abc_xyz",
        "total_orders", "total_revenue", "avg_order_value",
        "avg_review_score", "cancellation_rate", "ontime_delivery_rate",
        "avg_dispatch_days", "avg_delivery_days", "avg_total_delivery_days",
        "unique_categories", "unique_customers",
        # normalised KPIs for transparency
        "norm_review_score", "norm_ontime_rate",
        "norm_cancellation_rate", "norm_dispatch_speed", "norm_delivery_speed",
    ]
    available = [c for c in cols if c in dashboard.columns]
    return dashboard[available].reset_index(drop=True)


# ──────────────────────────────────────────────────────────────────────────────
# 12. Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Seller & Supply Chain Analytics – Section 4.2")
    parser.add_argument("--input",        type=str,   default=None,
                        help="Path to merged_orders_clean.csv (auto-detected if omitted)")
    parser.add_argument("--output",       type=str,   default="seller_analytics_output",
                        help="Output directory (default: seller_analytics_output)")
    parser.add_argument("--clusters",     type=int,   default=DEFAULT_N_CLUSTERS,
                        help=f"Number of K-Means clusters (default: {DEFAULT_N_CLUSTERS})")
    parser.add_argument("--dispatch_sla", type=float, default=DEFAULT_DISPATCH_SLA,
                        help=f"Dispatch SLA in days (default: {DEFAULT_DISPATCH_SLA})")
    parser.add_argument("--delivery_sla", type=float, default=DEFAULT_DELIVERY_SLA,
                        help=f"Delivery SLA in days (default: {DEFAULT_DELIVERY_SLA})")
    args = parser.parse_args()

    out_dir   = Path(args.output)
    plots_dir = out_dir / "plots"
    out_dir.mkdir(exist_ok=True)

    print("=" * 65)
    print("  Section 4.2 – Seller & Supply Chain Analytics")
    print("=" * 65)

    # ── Step 1: Load raw data ─────────────────────────────────────────────
    df_raw = load_data(args.input)

    # ── Step 2: Feature engineering ───────────────────────────────────────
    df = engineer_features(df_raw)
    seller_stats = aggregate_seller_features(df)
    seller_stats = add_monthly_order_variability(df, seller_stats)

    # ── Step 3: Normalise KPIs ────────────────────────────────────────────
    seller_stats = normalise_kpis(seller_stats)
    seller_stats.to_csv(out_dir / "seller_feature_matrix.csv", index=False)
    print(f"\n[save] seller_feature_matrix.csv  ({len(seller_stats)} sellers)")

    # ── Step 4: Weighted composite score ──────────────────────────────────
    seller_stats = compute_weighted_score(seller_stats)

    # ── Step 5: PCA score ─────────────────────────────────────────────────
    seller_stats = compute_pca_score(seller_stats)

    # ── Step 6: K-Means clustering ────────────────────────────────────────
    seller_stats = cluster_sellers(seller_stats, n_clusters=args.clusters)

    # ── Step 7: ABC-XYZ classification ───────────────────────────────────
    seller_stats = classify_abc_xyz(seller_stats)
    seller_stats.to_csv(out_dir / "seller_scores.csv", index=False)
    print(f"\n[save] seller_scores.csv")

    # Cluster-only file
    cluster_cols = [
        "seller_id", "seller_state", "cluster_id", "seller_persona",
        "composite_score", "pca_score", "abc_xyz",
        "avg_review_score", "cancellation_rate",
        "avg_dispatch_days", "avg_delivery_days",
    ]
    seller_stats[[c for c in cluster_cols if c in seller_stats.columns]].to_csv(
        out_dir / "seller_clusters.csv", index=False
    )
    print(f"[save] seller_clusters.csv")

    # ── Step 8: Regional bottleneck analysis ─────────────────────────────
    regional = regional_analysis(df, seller_stats)
    regional.to_csv(out_dir / "regional_bottlenecks.csv", index=False)
    print(f"[save] regional_bottlenecks.csv")

    # ── Step 9: SLA breach detection ─────────────────────────────────────
    breached_orders = detect_bottleneck_orders(df, args.dispatch_sla, args.delivery_sla)
    breached_orders.to_csv(out_dir / "bottleneck_orders.csv", index=False)
    print(f"[save] bottleneck_orders.csv  ({len(breached_orders):,} breach events)")

    # ── Step 10: Final ranking dashboard ─────────────────────────────────
    dashboard = build_dashboard_table(seller_stats)
    dashboard.to_csv(out_dir / "seller_ranking_dashboard.csv", index=False)
    print(f"[save] seller_ranking_dashboard.csv")

    # ── Step 11: Visualisations ───────────────────────────────────────────
    build_plots(seller_stats, regional, breached_orders, plots_dir)

    # ── Final summary ─────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  FINAL SUMMARY")
    print("=" * 65)
    print(f"  Sellers scored        : {len(seller_stats):,}")
    print(f"  Bottleneck regions    : {regional['is_bottleneck'].sum()}")

    print(f"\n  Top 10 Sellers (by composite score):")
    top10 = dashboard[["rank","seller_id","seller_state","seller_persona",
                        "composite_score","score_grade"]].head(10)
    print(top10.to_string(index=False))

    print(f"\n  Bottom 10 Sellers (need intervention):")
    bot10 = dashboard[["rank","seller_id","seller_state","seller_persona",
                        "composite_score","score_grade"]].tail(10)
    print(bot10.to_string(index=False))

    print(f"\n  Score grade distribution:")
    print(dashboard["score_grade"].value_counts().sort_index().to_string())

    print(f"\n  Persona distribution:")
    print(seller_stats["seller_persona"].value_counts().to_string())

    print(f"\n  ABC-XYZ distribution:")
    print(seller_stats["abc_xyz"].value_counts().head(12).to_string())

    print(f"\n  Critical bottleneck states:")
    crit = regional[regional["region_severity"] == "🔴 Critical"][
        ["seller_state", "cancellation_rate", "avg_dispatch_days",
         "avg_delivery_days", "ontime_rate", "bottleneck_signals"]
    ]
    if not crit.empty:
        print(crit.to_string(index=False))
    else:
        print("  None detected with current thresholds.")

    print(f"\n  Output files saved to: {out_dir.resolve()}/")
    print("=" * 65)


if __name__ == "__main__":
    main()
