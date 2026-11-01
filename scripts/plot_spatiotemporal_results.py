"""
Generate publication-quality figures for Spatio-Temporal LOSO Cross-Validation (July 1 Split).
Includes:
  1. Time-series comparison across the three folds (July 1 - Dec 31).
  2. Actual vs Predicted scatter plots (Model A vs Model B).
  3. Spatio-temporal vs Spatial bars.
  4. Monthly sample and climatic regimes (sample counts, soil moisture, precipitation, frozen flags).
  5. Comprehensive Three-Experiment Validation Benchmark comparison.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

PRED_CSV = PROCESSED_DIR / "loso_spatiotemporal_predictions.csv"
METRICS_CSV = PROCESSED_DIR / "loso_spatiotemporal_metrics.csv"
COMP_CSV = PROCESSED_DIR / "loso_spatiotemporal_vs_spatial_comparison.csv"
HARMONIZED_CSV = PROCESSED_DIR / "loso_harmonized_dataset_median.csv"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.titlesize": 14,
})


def plot_spatiotemporal_timeseries():
    df = pd.read_csv(PRED_CSV)
    df["date"] = pd.to_datetime(df["date"])

    stations = [
        ("p037", "Fold 1: Fremont Pass, CO (P037, Test N=145, Alt: 3514m)", "Fold_1"),
        ("p043", "Fold 2: Newcastle, WY (P043, Test N=138, Alt: 1304m)", "Fold_2"),
        ("p041", "Fold 3: Marshall, CO (P041, Test N=150, Alt: 1743m)", "Fold_3"),
    ]

    fig, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=True)

    for i, (st, title, fold) in enumerate(stations):
        ax = axes[i]
        sub = df[df["station"] == st].sort_values("date")

        ax.plot(sub["date"], sub["pbo_sm"], label="Actual PBO_H2O Reference", color="black", linewidth=1.8, marker="o", markersize=3.5, alpha=0.9)
        ax.plot(sub["date"], sub["pred_rf_Model_A_weighted"], label="RF Model A (rh + amp)", color="#1f77b4", linewidth=1.5, linestyle="-")
        ax.plot(sub["date"], sub["pred_rf_Model_B_weighted"], label="RF Model B (+ phase)", color="#2ca02c", linewidth=1.3, linestyle="--")
        ax.plot(sub["date"], sub["pred_precip_ols_weighted"], label="Precipitation OLS Baseline (3-feat)", color="#ff7f0e", linewidth=1.2, linestyle=":")
        ax.plot(sub["date"], sub["pred_climatology_weighted"], label="Climatology (Pre-Jul Mean)", color="#7f7f7f", linewidth=1.1, linestyle="-.")

        ax.set_title(title, fontweight="bold", pad=8)
        ax.set_ylabel("Soil Moisture ($m^3/m^3$)")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.set_ylim(-0.02, 0.35)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))

        if i == 0:
            ax.legend(loc="upper right", framealpha=0.9, ncol=2)

    axes[2].set_xlabel("Test Period Date (July 1 - December 31, 2016)")
    plt.tight_layout()
    out_path = FIGURES_DIR / "loso_spatiotemporal_timeseries.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[SAVED] Spatio-temporal time-series figure: {out_path}")


def plot_spatiotemporal_scatters():
    df = pd.read_csv(PRED_CSV)
    
    stations = [
        ("p037", "Fold 1: Fremont Pass (P037)"),
        ("p043", "Fold 2: Newcastle (P043)"),
        ("p041", "Fold 3: Marshall (P041)"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 9.5))

    for col, (st, title) in enumerate(stations):
        sub = df[df["station"] == st]
        act = sub["pbo_sm"]

        # Row 0: Model A (rh + amp)
        ax_a = axes[0, col]
        pred_a = sub["pred_rf_Model_A_weighted"]
        rmse_a = np.sqrt(np.mean((pred_a - act)**2))
        ubrmse_a = np.sqrt(np.mean((pred_a - act - np.mean(pred_a - act))**2))
        r_a = np.corrcoef(act, pred_a)[0, 1]

        ax_a.scatter(act, pred_a, alpha=0.6, color="#1f77b4", edgecolor="none", s=35)
        ax_a.plot([0, 0.35], [0, 0.35], "k--", alpha=0.75, label="1:1 Line")
        z = np.polyfit(act, pred_a, 1)
        p = np.poly1d(z)
        ax_a.plot(np.linspace(0, 0.30, 100), p(np.linspace(0, 0.30, 100)), "r-", alpha=0.8, linewidth=1.5, label="Fit")
        ax_a.set_xlim(-0.01, 0.35)
        ax_a.set_ylim(-0.01, 0.35)
        ax_a.set_title(f"{title}\nModel A (rh + amp)", fontweight="bold")
        ax_a.set_xlabel("Actual $m^3/m^3$")
        ax_a.set_ylabel("Predicted $m^3/m^3$")
        ax_a.grid(True, linestyle="--", alpha=0.5)
        ax_a.text(0.03, 0.92, f"RMSE = {rmse_a:.4f}\nubRMSE = {ubrmse_a:.4f}\n$r$ = {r_a:.3f}",
                  transform=ax_a.transAxes, verticalalignment="top",
                  bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85, edgecolor="#ccc"))

        # Row 1: Model B (rh + amp + phase)
        ax_b = axes[1, col]
        pred_b = sub["pred_rf_Model_B_weighted"]
        rmse_b = np.sqrt(np.mean((pred_b - act)**2))
        ubrmse_b = np.sqrt(np.mean((pred_b - act - np.mean(pred_b - act))**2))
        r_b = np.corrcoef(act, pred_b)[0, 1]

        ax_b.scatter(act, pred_b, alpha=0.6, color="#2ca02c", edgecolor="none", s=35)
        ax_b.plot([0, 0.35], [0, 0.35], "k--", alpha=0.75, label="1:1 Line")
        z = np.polyfit(act, pred_b, 1)
        p = np.poly1d(z)
        ax_b.plot(np.linspace(0, 0.30, 100), p(np.linspace(0, 0.30, 100)), "r-", alpha=0.8, linewidth=1.5, label="Fit")
        ax_b.set_xlim(-0.01, 0.35)
        ax_b.set_ylim(-0.01, 0.35)
        ax_b.set_title(f"{title}\nModel B (+ phase)", fontweight="bold")
        ax_b.set_xlabel("Actual $m^3/m^3$")
        ax_b.set_ylabel("Predicted $m^3/m^3$")
        ax_b.grid(True, linestyle="--", alpha=0.5)
        ax_b.text(0.03, 0.92, f"RMSE = {rmse_b:.4f}\nubRMSE = {ubrmse_b:.4f}\n$r$ = {r_b:.3f}",
                  transform=ax_b.transAxes, verticalalignment="top",
                  bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85, edgecolor="#ccc"))

    plt.tight_layout()
    out_path = FIGURES_DIR / "loso_spatiotemporal_scatters.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[SAVED] Spatio-temporal scatter figure: {out_path}")


def plot_spatiotemporal_vs_spatial_bars():
    df = pd.read_csv(COMP_CSV)
    
    # Filter for Random Forest Model A
    rf_df = df[df["model_family"] == "Random_Forest"].copy()
    
    folds = ["Fold_1", "Fold_2", "Fold_3", "Pooled_Overall"]
    fold_labels = ["Fold 1 (P037)", "Fold 2 (P043)", "Fold 3 (P041)", "Pooled Overall"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    x = np.arange(len(folds))
    w = 0.25

    # RMSE comparison: Full Spatial vs Post-Jul Spatial Control vs Spatio-Temporal
    ax1.bar(x - w, rf_df["spatial_all_dates_rmse"], w, label="Spatial LOSO (Full Year, N=771)", color="#aec7e8", edgecolor="#333")
    ax1.bar(x, rf_df["spatial_post_jul_control_rmse"], w, label="Spatial LOSO (Post-Jul Control, N=433)", color="#1f77b4", edgecolor="#333")
    ax1.bar(x + w, rf_df["spatiotemporal_rmse"], w, label="Spatio-Temporal (Early-Train / Late-Test, N=433)", color="#d62728", edgecolor="#333")

    ax1.set_xticks(x)
    ax1.set_xticklabels(fold_labels, fontweight="bold")
    ax1.set_ylabel("RMSE ($m^3/m^3$)")
    ax1.set_title("Cross-Experiment RMSE Comparison (RF Model A)", fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.5, axis="y")
    ax1.legend(loc="upper left")
    ax1.set_ylim(0, 0.15)

    # Correlation comparison
    ax2.bar(x - w, rf_df["spatial_all_dates_r"], w, label="Spatial LOSO (Full Year, N=771)", color="#aec7e8", edgecolor="#333")
    ax2.bar(x, rf_df["spatial_post_jul_control_r"], w, label="Spatial LOSO (Post-Jul Control, N=433)", color="#1f77b4", edgecolor="#333")
    ax2.bar(x + w, rf_df["spatiotemporal_r"], w, label="Spatio-Temporal (Early-Train / Late-Test, N=433)", color="#d62728", edgecolor="#333")

    ax2.set_xticks(x)
    ax2.set_xticklabels(fold_labels, fontweight="bold")
    ax2.set_ylabel("Pearson Correlation ($r$)")
    ax2.set_title("Cross-Experiment Pearson $r$ Comparison (RF Model A)", fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.5, axis="y")
    ax2.legend(loc="upper left")
    ax2.set_ylim(0, 0.8)

    plt.tight_layout()
    out_path = FIGURES_DIR / "loso_spatiotemporal_vs_spatial_bars.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[SAVED] Comparison bars figure: {out_path}")


def plot_spatiotemporal_monthly_regimes():
    df = pd.read_csv(HARMONIZED_CSV)
    df["date"] = pd.to_datetime(df["date"])
    df["month"] = df["date"].dt.month
    
    # Filter usable paired dates
    usable = df[df["usable_for_loso"]].copy()

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    months = np.arange(1, 13)
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    colors = {"p037": "#2ca02c", "p043": "#ff7f0e", "p041": "#1f77b4"}
    names = {"p037": "P037 (Fremont Pass)", "p043": "P043 (Newcastle)", "p041": "P041 (Marshall)"}

    # 1. Monthly usable sample count
    ax1 = axes[0, 0]
    w = 0.25
    for i, st in enumerate(["p037", "p043", "p041"]):
        sub = usable[usable["station"] == st]
        counts = [len(sub[sub["month"] == m]) for m in months]
        ax1.bar(months + (i - 1) * w, counts, w, label=names[st], color=colors[st], edgecolor="#333", alpha=0.85)
    ax1.axvline(6.5, color="red", linestyle="--", linewidth=1.8, label="Split Boundary (July 1)")
    ax1.set_xticks(months)
    ax1.set_xticklabels(month_names, fontweight="bold")
    ax1.set_ylabel("Usable Paired Days")
    ax1.set_title("A: Monthly Data Availability Across 2016", fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.5, axis="y")
    ax1.legend(loc="upper left", fontsize=8.5)
    ax1.set_ylim(0, 35)

    # 2. Monthly mean soil moisture
    ax2 = axes[0, 1]
    for st in ["p037", "p043", "p041"]:
        sub = usable[usable["station"] == st]
        means = [sub[sub["month"] == m]["pbo_sm"].mean() for m in months]
        stds = [sub[sub["month"] == m]["pbo_sm"].std() for m in months]
        ax2.errorbar(months, means, yerr=stds, label=names[st], color=colors[st], marker="o", linewidth=1.5, capsize=3)
    ax2.axvline(6.5, color="red", linestyle="--", linewidth=1.8, label="Split Boundary (July 1)")
    ax2.set_xticks(months)
    ax2.set_xticklabels(month_names, fontweight="bold")
    ax2.set_ylabel("Mean PBO_H2O Soil Moisture ($m^3/m^3$)")
    ax2.set_title("B: Seasonal Soil Moisture Trajectory (Mean +/- 1 SD)", fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="upper right", fontsize=8.5)
    ax2.set_ylim(0, 0.28)

    # 3. Monthly precipitation totals
    ax3 = axes[1, 0]
    for i, st in enumerate(["p037", "p043", "p041"]):
        sub = df[df["station"] == st]
        precip = [sub[sub["month"] == m]["precip_mm"].sum() for m in months]
        ax3.bar(months + (i - 1) * w, precip, w, label=names[st], color=colors[st], edgecolor="#333", alpha=0.85)
    ax3.axvline(6.5, color="red", linestyle="--", linewidth=1.8)
    ax3.set_xticks(months)
    ax3.set_xticklabels(month_names, fontweight="bold")
    ax3.set_ylabel("Total Monthly Precipitation (mm)")
    ax3.set_title("C: Monthly Precipitation Regime", fontweight="bold")
    ax3.grid(True, linestyle="--", alpha=0.5, axis="y")
    ax3.legend(loc="upper left", fontsize=8.5)

    # 4. Monthly QC Flagged / Frozen Days
    ax4 = axes[1, 1]
    for i, st in enumerate(["p037", "p043", "p041"]):
        sub = df[df["station"] == st]
        frozen = [sub[(sub["month"] == m) & (sub["pbo_sm_flag"].isin(["D02", "D03", "D02,D03"]))].shape[0] for m in months]
        ax4.bar(months + (i - 1) * w, frozen, w, label=names[st], color=colors[st], edgecolor="#333", alpha=0.85)
    ax4.axvline(6.5, color="red", linestyle="--", linewidth=1.8)
    ax4.set_xticks(months)
    ax4.set_xticklabels(month_names, fontweight="bold")
    ax4.set_ylabel("Days with Frozen Soil / Air Flag")
    ax4.set_title("D: Cold-Season Frozen Ground Restrictions (ISMN D02/D03 Flags)", fontweight="bold")
    ax4.grid(True, linestyle="--", alpha=0.5, axis="y")
    ax4.legend(loc="upper right", fontsize=8.5)
    ax4.set_ylim(0, 35)

    plt.tight_layout()
    out_path = FIGURES_DIR / "loso_spatiotemporal_monthly_regimes.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[SAVED] Monthly regimes figure: {out_path}")


def plot_three_experiment_comparison_bars():
    """Publication-quality bar chart comparing all 3 validation experiments."""
    comp_df = pd.read_csv(COMP_CSV)
    rf_df = comp_df[comp_df["model_family"] == "Random_Forest"].copy()

    folds = ["Fold_1", "Fold_2", "Fold_3", "Pooled_Overall"]
    fold_labels = ["Fold 1 (P037)", "Fold 2 (P043)", "Fold 3 (P041)", "Pooled Overall"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    x = np.arange(len(folds))
    w = 0.20

    # 1. RMSE Across Experiments
    ax1.bar(x - 1.5*w, rf_df["spatial_all_dates_rmse"], w, label="Exp 1: All-Dates Spatial (N=771)", color="#4682b4", edgecolor="#333")
    ax1.bar(x - 0.5*w, rf_df["same_time_spatial_rmse"], w, label="Exp 2: Same-Time Spatial (N=648)", color="#2e8b57", edgecolor="#333")
    ax1.bar(x + 0.5*w, rf_df["spatial_post_jul_control_rmse"], w, label="Post-Jul Spatial Control (N=433)", color="#9370db", edgecolor="#333")
    ax1.bar(x + 1.5*w, rf_df["spatiotemporal_rmse"], w, label="Exp 3: Spatio-Temporal (N=433)", color="#d9534f", edgecolor="#333")

    ax1.set_xticks(x)
    ax1.set_xticklabels(fold_labels, fontweight="bold")
    ax1.set_ylabel("RMSE ($m^3/m^3$)")
    ax1.set_title("Cross-Experiment RMSE (Random Forest Model A)", fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.5, axis="y")
    ax1.legend(loc="upper left", fontsize=8.5)
    ax1.set_ylim(0, 0.15)

    # 2. Pearson r Across Experiments
    ax2.bar(x - 1.5*w, rf_df["spatial_all_dates_r"], w, label="Exp 1: All-Dates Spatial (N=771)", color="#4682b4", edgecolor="#333")
    ax2.bar(x - 0.5*w, rf_df["same_time_spatial_r"], w, label="Exp 2: Same-Time Spatial (N=648)", color="#2e8b57", edgecolor="#333")
    ax2.bar(x + 0.5*w, rf_df["spatial_post_jul_control_r"], w, label="Post-Jul Spatial Control (N=433)", color="#9370db", edgecolor="#333")
    ax2.bar(x + 1.5*w, rf_df["spatiotemporal_r"], w, label="Exp 3: Spatio-Temporal (N=433)", color="#d9534f", edgecolor="#333")

    ax2.set_xticks(x)
    ax2.set_xticklabels(fold_labels, fontweight="bold")
    ax2.set_ylabel("Pearson Correlation ($r$)")
    ax2.set_title("Cross-Experiment Pearson $r$ (Random Forest Model A)", fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.5, axis="y")
    ax2.legend(loc="upper left", fontsize=8.5)
    ax2.set_ylim(0, 0.8)

    plt.tight_layout()
    out_path = FIGURES_DIR / "loso_three_experiment_comparison_bars.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[SAVED] Three-experiment comparison bars: {out_path}")


if __name__ == "__main__":
    plot_spatiotemporal_timeseries()
    plot_spatiotemporal_scatters()
    plot_spatiotemporal_vs_spatial_bars()
    plot_spatiotemporal_monthly_regimes()
    plot_three_experiment_comparison_bars()
