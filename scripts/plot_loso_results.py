"""
Generate publication-quality figures for Three-Station LOSO Cross-Validation and Feature Ablation.
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

PRED_CSV = PROCESSED_DIR / "loso_primary_spatial_predictions.csv"
METRICS_CSV = PROCESSED_DIR / "loso_primary_spatial_metrics.csv"
SPATIO_CSV = PROCESSED_DIR / "loso_spatiotemporal_split_audit.csv"

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

def plot_loso_timeseries():
    df = pd.read_csv(PRED_CSV)
    df["date"] = pd.to_datetime(df["date"])

    stations = [
        ("p037", "Fold 1: Fremont Pass, CO (P037, Alt: 3514m, Sand: 58%)", "Fold_1"),
        ("p043", "Fold 2: Newcastle, WY (P043, Alt: 1304m, Sand: 37%)", "Fold_2"),
        ("p041", "Fold 3: Marshall, CO (P041, Alt: 1743m, Sand: 37%)", "Fold_3"),
    ]

    fig, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=False)

    for i, (st, title, fold) in enumerate(stations):
        ax = axes[i]
        sub = df[df["station"] == st].sort_values("date")

        ax.plot(sub["date"], sub["pbo_sm"], label="Actual PBO_H2O Reference", color="black", linewidth=1.8, marker="o", markersize=3, alpha=0.9)
        ax.plot(sub["date"], sub["pred_rf_Model_A_weighted"], label="RF Model A (rh + amp)", color="#1f77b4", linewidth=1.4, linestyle="-")
        ax.plot(sub["date"], sub["pred_rf_Model_B_weighted"], label="RF Model B (+ phase)", color="#2ca02c", linewidth=1.2, linestyle="--")
        ax.plot(sub["date"], sub["pred_precip_ols_weighted"], label="Precipitation OLS Baseline", color="#ff7f0e", linewidth=1.1, linestyle=":")

        ax.set_title(title, fontweight="bold", pad=8)
        ax.set_ylabel("Soil Moisture ($m^3/m^3$)")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.set_ylim(-0.02, 0.40)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))

        if i == 0:
            ax.legend(loc="upper right", framealpha=0.9)

    plt.tight_layout()
    out_path = FIGURES_DIR / "loso_spatial_timeseries_comparison.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[SAVED] Time-series figure: {out_path}")


def plot_loso_scatters():
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

        ax_a.scatter(act, pred_a, alpha=0.55, color="#1f77b4", edgecolor="none", s=30)
        ax_a.plot([0, 0.4], [0, 0.4], "k--", alpha=0.75, label="1:1 Line")
        z = np.polyfit(act, pred_a, 1)
        p = np.poly1d(z)
        ax_a.plot(np.linspace(0, 0.35, 100), p(np.linspace(0, 0.35, 100)), "r-", alpha=0.8, linewidth=1.5, label="Fit")
        ax_a.set_xlim(-0.01, 0.38)
        ax_a.set_ylim(-0.01, 0.38)
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

        ax_b.scatter(act, pred_b, alpha=0.55, color="#2ca02c", edgecolor="none", s=30)
        ax_b.plot([0, 0.4], [0, 0.4], "k--", alpha=0.75, label="1:1 Line")
        z = np.polyfit(act, pred_b, 1)
        p = np.poly1d(z)
        ax_b.plot(np.linspace(0, 0.35, 100), p(np.linspace(0, 0.35, 100)), "r-", alpha=0.8, linewidth=1.5, label="Fit")
        ax_b.set_xlim(-0.01, 0.38)
        ax_b.set_ylim(-0.01, 0.38)
        ax_b.set_title(f"{title}\nModel B (+ phase)", fontweight="bold")
        ax_b.set_xlabel("Actual $m^3/m^3$")
        ax_b.set_ylabel("Predicted $m^3/m^3$")
        ax_b.grid(True, linestyle="--", alpha=0.5)
        ax_b.text(0.03, 0.92, f"RMSE = {rmse_b:.4f}\nubRMSE = {ubrmse_b:.4f}\n$r$ = {r_b:.3f}",
                  transform=ax_b.transAxes, verticalalignment="top",
                  bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85, edgecolor="#ccc"))

    plt.tight_layout()
    out_path = FIGURES_DIR / "loso_scatter_actual_vs_predicted.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[SAVED] Scatter figure: {out_path}")


def plot_feature_ablation_bars():
    df = pd.read_csv(METRICS_CSV)
    
    # Filter for weighted primary evaluation
    df_w = df[df["weighting"] == "weighted"].copy()
    
    # Select folds of interest: Fold_1 (P037), Fold_2 (P043), Fold_3 (P041), Pooled_Overall
    folds = ["Fold_1", "Fold_2", "Fold_3", "Pooled_Overall"]
    fold_labels = ["Fold 1 (P037)", "Fold 2 (P043)", "Fold 3 (P041)", "Pooled (All 3)"]
    
    # Models to compare
    models = [
        ("Climatology", "Baseline", "Climatology", "#7f7f7f"),
        ("Precipitation_OLS", "Baseline", "Precip OLS", "#ff7f0e"),
        ("Random_Forest", "Model_A", "RF: Model A (rh+amp)", "#1f77b4"),
        ("Random_Forest", "Model_B", "RF: Model B (+phase)", "#2ca02c"),
        ("Random_Forest", "Model_C", "RF: Model C (+QC)", "#9467bd"),
        ("XGBoost", "Model_A", "XGB: Model A (rh+amp)", "#aec7e8"),
        ("XGBoost", "Model_B", "XGB: Model B (+phase)", "#98df8a"),
        ("XGBoost", "Model_C", "XGB: Model C (+QC)", "#c5b0d5"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(17, 6))

    metrics = [
        ("rmse", "Root Mean Square Error (RMSE, $m^3/m^3$)", axes[0]),
        ("ubrmse", "Unbiased RMSE (ubRMSE, $m^3/m^3$)", axes[1]),
        ("pearson_r", "Pearson Correlation ($r$)", axes[2]),
    ]

    x = np.arange(len(folds))
    width = 0.10

    for m_key, m_title, ax in metrics:
        for j, (fam, fset, label, color) in enumerate(models):
            vals = []
            for f in folds:
                row = df_w[(df_w["fold"] == f) & (df_w["model_family"] == fam) & (df_w["feature_set"] == fset)]
                if len(row) > 0 and pd.notnull(row[m_key].values[0]):
                    vals.append(row[m_key].values[0])
                else:
                    vals.append(0.0)
            offset = (j - len(models) / 2 + 0.5) * width
            ax.bar(x + offset, vals, width, label=label if m_key == "rmse" else "", color=color, edgecolor="#333", linewidth=0.5)

        ax.set_title(m_title, fontweight="bold", pad=10)
        ax.set_xticks(x)
        ax.set_xticklabels(fold_labels, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.5, axis="y")
        if m_key == "pearson_r":
            ax.set_ylim(-0.4, 0.7)
            ax.axhline(0, color="k", linewidth=0.8, linestyle="-")
        else:
            ax.set_ylim(0, 0.14)

    axes[0].legend(loc="upper left", bbox_to_anchor=(0.0, 1.25), ncol=4, framealpha=0.9, fontsize=8.5)
    plt.tight_layout()
    out_path = FIGURES_DIR / "loso_feature_ablation_metrics_bar.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] Ablation bars figure: {out_path}")


def plot_spatiotemporal_audit():
    df = pd.read_csv(SPATIO_CSV)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # Bar chart of Train vs Test N
    x = np.arange(len(df))
    w = 0.35
    ax1.bar(x - w/2, df["Train_Total_N"], w, label="Training Samples (Jan 1 - Jun 30)", color="#1f77b4", edgecolor="#333")
    ax1.bar(x + w/2, df["Test_N"], w, label="Testing Samples (Jul 1 - Dec 31)", color="#2ca02c", edgecolor="#333")
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{r['Fold']}\nHeld-Out: {r['Held_Out_Test_Station']}" for _, r in df.iterrows()], fontweight="bold")
    ax1.set_ylabel("Number of Daily Samples")
    ax1.set_title("Sample Counts: Early-Train vs Late-Test (July 1 Split)", fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.5, axis="y")
    ax1.legend(loc="upper right")
    for i, r in df.iterrows():
        ax1.text(i - w/2, r["Train_Total_N"] + 3, str(int(r["Train_Total_N"])), ha="center", fontsize=9, fontweight="bold")
        ax1.text(i + w/2, r["Test_N"] + 3, str(int(r["Test_N"])), ha="center", fontsize=9, fontweight="bold")
    ax1.set_ylim(0, 280)

    # Monthly distribution of test samples
    months = ["Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    colors = ["#4e79a7", "#f28e2c", "#e15759"]
    
    x_m = np.arange(len(months))
    w_m = 0.25
    for i, r in df.iterrows():
        m_counts = [r[f"Test_{m}_N"] for m in months]
        offset = (i - 1) * w_m
        ax2.bar(x_m + offset, m_counts, w_m, label=f"{r['Fold']} ({r['Held_Out_Test_Station']})", color=colors[i], edgecolor="#333")

    ax2.set_xticks(x_m)
    ax2.set_xticklabels(months, fontweight="bold")
    ax2.set_ylabel("Held-Out Test Sample Count")
    ax2.set_title("Monthly Test Sample Availability (Jul - Dec 2016)", fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.5, axis="y")
    ax2.legend(loc="upper right")
    ax2.set_ylim(0, 36)

    plt.tight_layout()
    out_path = FIGURES_DIR / "loso_spatiotemporal_split_overview.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[SAVED] Spatio-temporal audit figure: {out_path}")


if __name__ == "__main__":
    plot_loso_timeseries()
    plot_loso_scatters()
    plot_feature_ablation_bars()
    plot_spatiotemporal_audit()
