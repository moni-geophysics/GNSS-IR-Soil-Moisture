"""
Generate publication-quality figures for Same-Time Spatial LOSO Cross-Validation.
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

PRED_CSV = PROCESSED_DIR / "loso_same_time_predictions.csv"
METRICS_CSV = PROCESSED_DIR / "loso_same_time_metrics.csv"
COMP_CSV = PROCESSED_DIR / "loso_same_time_vs_all_dates_comparison.csv"
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

def plot_same_time_timeseries():
    df = pd.read_csv(PRED_CSV)
    df["date"] = pd.to_datetime(df["date"])

    stations = [
        ("p037", "Fold 1: Fremont Pass, CO (P037, Common Dates N=216, Alt: 3514m)", "Fold_1"),
        ("p043", "Fold 2: Newcastle, WY (P043, Common Dates N=216, Alt: 1304m)", "Fold_2"),
        ("p041", "Fold 3: Marshall, CO (P041, Common Dates N=216, Alt: 1743m)", "Fold_3"),
    ]

    fig, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=True)

    for i, (st, title, fold) in enumerate(stations):
        ax = axes[i]
        sub = df[df["station"] == st].sort_values("date")

        ax.plot(sub["date"], sub["pbo_sm"], label="Actual PBO_H2O Reference", color="black", linewidth=1.8, marker="o", markersize=3, alpha=0.9)
        ax.plot(sub["date"], sub["pred_rf_Model_A_weighted"], label="RF Model A (rh + amp)", color="#1f77b4", linewidth=1.4, linestyle="-")
        ax.plot(sub["date"], sub["pred_rf_Model_B_weighted"], label="RF Model B (+ phase)", color="#2ca02c", linewidth=1.2, linestyle="--")
        ax.plot(sub["date"], sub["pred_precip_ols_weighted"], label="Precipitation OLS Baseline", color="#ff7f0e", linewidth=1.1, linestyle=":")
        ax.plot(sub["date"], sub["pred_climatology_weighted"], label="Climatology Baseline", color="#7f7f7f", linewidth=1.0, linestyle="-.")

        ax.set_title(title, fontweight="bold", pad=8)
        ax.set_ylabel("Soil Moisture ($m^3/m^3$)")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.set_ylim(-0.02, 0.38)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))

        if i == 0:
            ax.legend(loc="upper right", framealpha=0.9, ncol=2)

    axes[2].set_xlabel("Date (Synchronized Period: Feb 18 - Nov 22, 2016)")
    plt.tight_layout()
    out_path = FIGURES_DIR / "loso_same_time_timeseries.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[SAVED] Same-time time-series figure: {out_path}")


def plot_same_time_scatters():
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

        ax_a.scatter(act, pred_a, alpha=0.6, color="#1f77b4", edgecolor="none", s=32)
        ax_a.plot([0, 0.38], [0, 0.38], "k--", alpha=0.75, label="1:1 Line")
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

        ax_b.scatter(act, pred_b, alpha=0.6, color="#2ca02c", edgecolor="none", s=32)
        ax_b.plot([0, 0.38], [0, 0.38], "k--", alpha=0.75, label="1:1 Line")
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
    out_path = FIGURES_DIR / "loso_same_time_scatters.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[SAVED] Same-time scatter figure: {out_path}")


def plot_same_time_vs_all_dates_bars():
    df = pd.read_csv(COMP_CSV)
    
    # Filter for Random Forest Model A
    rf_df = df[df["model_family"] == "Random_Forest"].copy()
    
    folds = ["Fold_1", "Fold_2", "Fold_3", "Pooled_Overall"]
    fold_labels = ["Fold 1 (P037)", "Fold 2 (P043)", "Fold 3 (P041)", "Pooled Overall"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    x = np.arange(len(folds))
    w = 0.25

    # RMSE comparison: All Available Dates vs Common Subset Control vs Same-Time Spatial
    ax1.bar(x - w, rf_df["all_dates_spatial_rmse"], w, label="All-Dates Spatial LOSO (N=771)", color="#aec7e8", edgecolor="#333")
    ax1.bar(x, rf_df["all_dates_model_on_common_subset_rmse"], w, label="All-Dates Model on Common Subset (N=648 Control)", color="#1f77b4", edgecolor="#333")
    ax1.bar(x + w, rf_df["same_time_spatial_rmse"], w, label="Same-Time Spatial LOSO (N=648)", color="#2ca02c", edgecolor="#333")

    ax1.set_xticks(x)
    ax1.set_xticklabels(fold_labels, fontweight="bold")
    ax1.set_ylabel("RMSE ($m^3/m^3$)")
    ax1.set_title("Cross-Experiment RMSE: Same-Time vs All-Dates (RF Model A)", fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.5, axis="y")
    ax1.legend(loc="upper left")
    ax1.set_ylim(0, 0.13)

    # Correlation comparison
    ax2.bar(x - w, rf_df["all_dates_spatial_r"], w, label="All-Dates Spatial LOSO (N=771)", color="#aec7e8", edgecolor="#333")
    ax2.bar(x, rf_df["all_dates_model_on_common_subset_r"], w, label="All-Dates Model on Common Subset (N=648 Control)", color="#1f77b4", edgecolor="#333")
    ax2.bar(x + w, rf_df["same_time_spatial_r"], w, label="Same-Time Spatial LOSO (N=648)", color="#2ca02c", edgecolor="#333")

    ax2.set_xticks(x)
    ax2.set_xticklabels(fold_labels, fontweight="bold")
    ax2.set_ylabel("Pearson Correlation ($r$)")
    ax2.set_title("Cross-Experiment Pearson $r$: Same-Time vs All-Dates (RF Model A)", fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.5, axis="y")
    ax2.legend(loc="upper left")
    ax2.set_ylim(0, 0.7)

    plt.tight_layout()
    out_path = FIGURES_DIR / "loso_same_time_vs_all_dates_bars.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[SAVED] Comparison bars figure: {out_path}")


def plot_same_time_coverage_calendar():
    df = pd.read_csv(HARMONIZED_CSV)
    df["date"] = pd.to_datetime(df["date"])
    
    # 216 common dates
    p041_d = set(df[(df["usable_for_loso"]) & (df["station"] == "p041")]["date"])
    p037_d = set(df[(df["usable_for_loso"]) & (df["station"] == "p037")]["date"])
    p043_d = set(df[(df["usable_for_loso"]) & (df["station"] == "p043")]["date"])
    common_dates = set(p041_d.intersection(p037_d).intersection(p043_d))

    all_dates_2016 = pd.date_range("2016-01-01", "2016-12-31", freq="D")
    
    fig, ax = plt.subplots(figsize=(15, 4.5))

    y_pos = [3, 2, 1, 0]
    st_names = ["P041 (Marshall)", "P037 (Fremont Pass)", "P043 (Newcastle)", "Synchronized Common Dates (N=216)"]
    
    # Plot individual usable dots
    p041_usable = all_dates_2016.isin(p041_d)
    p037_usable = all_dates_2016.isin(p037_d)
    p043_usable = all_dates_2016.isin(p043_d)
    common_usable = all_dates_2016.isin(common_dates)

    ax.scatter(all_dates_2016[p041_usable], np.full(p041_usable.sum(), 3), color="#1f77b4", marker="|", s=150, label="P041 Usable (N=247)")
    ax.scatter(all_dates_2016[p037_usable], np.full(p037_usable.sum(), 2), color="#2ca02c", marker="|", s=150, label="P037 Usable (N=276)")
    ax.scatter(all_dates_2016[p043_usable], np.full(p043_usable.sum(), 1), color="#ff7f0e", marker="|", s=150, label="P043 Usable (N=248)")
    ax.scatter(all_dates_2016[common_usable], np.full(common_usable.sum(), 0), color="#d62728", marker="|", s=180, label="Strictly Common Dates (N=216)")

    # Shade the common coverage range
    ax.axvspan(pd.to_datetime("2016-02-18"), pd.to_datetime("2016-11-22"), alpha=0.15, color="red", label="Synchronized Window: Feb 18 - Nov 22 (279 calendar days)")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(st_names, fontweight="bold")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax.set_xlim(pd.to_datetime("2016-01-01"), pd.to_datetime("2016-12-31"))
    ax.set_ylim(-0.8, 3.8)
    ax.set_title("Calendar Timeline of Station Usability and Strictly Common Dates across 2016", fontweight="bold", pad=12)
    ax.grid(True, linestyle="--", alpha=0.5, axis="x")
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 1.25), ncol=3, fontsize=9)

    plt.tight_layout()
    out_path = FIGURES_DIR / "loso_same_time_coverage_calendar.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] Coverage calendar figure: {out_path}")


if __name__ == "__main__":
    plot_same_time_timeseries()
    plot_same_time_scatters()
    plot_same_time_vs_all_dates_bars()
    plot_same_time_coverage_calendar()
