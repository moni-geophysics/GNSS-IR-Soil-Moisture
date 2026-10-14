"""
Leakage-Safe Temporal Cross-Validation for Model A Random Forest Tuning
Target Station: P041 (Marshall, Colorado), Year 2016.

This script executes:
1. 5-fold expanding-window temporal cross-validation strictly within the 252 training days.
2. Grid search across 24 RF hyperparameter combinations:
   - min_samples_leaf: [2, 3, 5, 10]
   - max_features: [0.5, 1.0]
   - max_depth: [None, 5, 10]
   - n_estimators: 300 (fixed)
   - random_state: 42 (fixed)
3. Evaluates both Mean and Median daily feature aggregations on identical folds.
4. Computes fold-level metrics: RMSE, Mean Bias, ubRMSE, Pearson r, Day-to-Day Delta r.
5. Computes summary metrics: Mean & Std across folds, and Pooled Out-of-Fold (OOF) metrics.
6. Strictly preserves zero access to the 63 held-out test dates.
7. Exports results to data/processed/p041/ (with fallback support for scratch/).
"""

import itertools
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit

# Suppress joblib / scikit-learn parallel worker warnings
warnings.filterwarnings("ignore", category=UserWarning)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "p041"
SCRATCH_DIR = PROJECT_ROOT / "scratch"


def resolve_input_path(primary: Path, fallback: Path) -> Path:
    if primary.exists():
        return primary
    if fallback.exists():
        return fallback
    return primary


MEAN_DATASET_CSV = resolve_input_path(
    PROCESSED_DIR / "p041_2016_daily_mean_dataset.csv",
    SCRATCH_DIR / "model_a_daily_mean_dataset.csv",
)
MEDIAN_DATASET_CSV = resolve_input_path(
    PROCESSED_DIR / "p041_2016_daily_median_dataset.csv",
    SCRATCH_DIR / "model_a_daily_median_dataset.csv",
)

OUT_FOLD_METRICS_CSV = PROCESSED_DIR / "p041_2016_rf_temporal_cv_fold_metrics.csv"
OUT_SUMMARY_CSV = PROCESSED_DIR / "p041_2016_rf_temporal_cv_summary.csv"

TARGET_COL = "target_cosmos_sm_median"
FEATURES_MEAN = ["rh_mean", "amp_mean", "phase_cos", "phase_sin"]
FEATURES_MEDIAN = ["rh_median", "amp_median", "phase_cos", "phase_sin"]

# Baseline configuration
BASELINE_CONFIG = {
    "min_samples_leaf": 3,
    "max_features": 1.0,
    "max_depth": None,
    "n_estimators": 300,
    "random_state": 42,
}

# Hyperparameter search space
SEARCH_SPACE = {
    "min_samples_leaf": [2, 3, 5, 10],
    "max_features": [0.5, 1.0],
    "max_depth": [None, 5, 10],
}


def compute_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    """
    Compute standard hydrological evaluation metrics:
    - RMSE
    - Mean Bias = mean(pred - actual)
    - ubRMSE = sqrt(mean((error - mean(error))**2))
    - Pearson r
    - Delta r = correlation of daily first-differences
    """
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    n = len(actual)

    error = predicted - actual
    rmse = float(np.sqrt(np.mean(error ** 2)))
    bias = float(np.mean(error))
    ubrmse = float(np.sqrt(np.mean((error - bias) ** 2)))

    std_act = float(np.std(actual))
    std_pred = float(np.std(predicted))
    if std_act > 1e-9 and std_pred > 1e-9 and n > 2:
        r = float(np.corrcoef(actual, predicted)[0, 1])
    else:
        r = np.nan

    if n > 3:
        d_act = np.diff(actual)
        d_pred = np.diff(predicted)
        if np.std(d_act) > 1e-9 and np.std(d_pred) > 1e-9:
            delta_r = float(np.corrcoef(d_act, d_pred)[0, 1])
        else:
            delta_r = np.nan
    else:
        delta_r = np.nan

    return {
        "n_samples": n,
        "rmse": rmse,
        "bias": bias,
        "ubrmse": ubrmse,
        "pearson_r": r,
        "delta_r": delta_r,
    }


def run_temporal_cv():
    print("=" * 80)
    print("LEAKAGE-SAFE TEMPORAL CROSS-VALIDATION FOR MODEL A (RANDOM FOREST)")
    print("=" * 80)

    # 1. Load Datasets
    print(f"Loading mean dataset:   {MEAN_DATASET_CSV}")
    print(f"Loading median dataset: {MEDIAN_DATASET_CSV}")
    df_mean = pd.read_csv(MEAN_DATASET_CSV)
    df_median = pd.read_csv(MEDIAN_DATASET_CSV)

    train_mean = df_mean[df_mean["split"] == "train"].copy().reset_index(drop=True)
    train_median = df_median[df_median["split"] == "train"].copy().reset_index(drop=True)

    n_train = len(train_mean)
    assert n_train == 252, f"Expected 252 training samples, got {n_train}"
    assert len(train_median) == 252, f"Expected 252 training samples, got {len(train_median)}"

    print(f"\nTraining Partition: {n_train} dates ({train_mean['date'].iloc[0]} to {train_mean['date'].iloc[-1]})")

    # 2. Setup 5-Fold Expanding-Window TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=5, gap=0)

    print("\n[TEMPORAL CROSS-VALIDATION FOLDS (5 EXPANDING BLOCKS)]")
    fold_info = []
    for fold_idx, (tr_idx, val_idx) in enumerate(tscv.split(train_mean)):
        tr_start, tr_end = train_mean.loc[tr_idx[0], "date"], train_mean.loc[tr_idx[-1], "date"]
        val_start, val_end = train_mean.loc[val_idx[0], "date"], train_mean.loc[val_idx[-1], "date"]
        val_mean_sm = train_mean.loc[val_idx, TARGET_COL].mean()
        val_std_sm = train_mean.loc[val_idx, TARGET_COL].std()
        val_min_sm = train_mean.loc[val_idx, TARGET_COL].min()
        val_max_sm = train_mean.loc[val_idx, TARGET_COL].max()
        fold_info.append({
            "fold": fold_idx + 1,
            "n_train": len(tr_idx),
            "tr_start": tr_start,
            "tr_end": tr_end,
            "n_val": len(val_idx),
            "val_start": val_start,
            "val_end": val_end,
            "val_mean_sm": val_mean_sm,
            "val_std_sm": val_std_sm,
            "val_range": f"[{val_min_sm:.3f}, {val_max_sm:.3f}]",
        })
        print(f"  Fold {fold_idx + 1}: Train N={len(tr_idx):3d} ({tr_start} to {tr_end}) | Val N={len(val_idx):2d} ({val_start} to {val_end}) | Val SM Mean={val_mean_sm:.4f}, Range=[{val_min_sm:.3f}, {val_max_sm:.3f}]")

    # 3. Build Hyperparameter Combinations
    param_keys = list(SEARCH_SPACE.keys())
    param_combos = [dict(zip(param_keys, p)) for p in itertools.product(*SEARCH_SPACE.values())]
    print(f"\nTotal hyperparameter configurations per aggregation: {len(param_combos)}")

    fold_records = []
    summary_records = []

    aggregations = [
        ("mean", train_mean, FEATURES_MEAN),
        ("median", train_median, FEATURES_MEDIAN),
    ]

    for agg_name, train_df, feats in aggregations:
        print(f"\n" + "-" * 70)
        print(f"RUNNING EXPANDING-WINDOW CV FOR: {agg_name.upper()} AGGREGATION")
        print("-" * 70)

        for cfg_idx, cfg in enumerate(param_combos):
            min_leaf = cfg["min_samples_leaf"]
            max_feat = cfg["max_features"]
            max_depth = cfg["max_depth"]

            is_base = (
                min_leaf == BASELINE_CONFIG["min_samples_leaf"] and
                max_feat == BASELINE_CONFIG["max_features"] and
                max_depth == BASELINE_CONFIG["max_depth"]
            )

            depth_str = str(max_depth) if max_depth is not None else "None"

            fold_rmses = []
            fold_biases = []
            fold_ubrmses = []
            fold_rs = []
            fold_delta_rs = []

            oof_preds = []
            oof_actual = []

            for fold_idx, (tr_idx, val_idx) in enumerate(tscv.split(train_df)):
                X_tr = train_df.loc[tr_idx, feats].to_numpy()
                y_tr = train_df.loc[tr_idx, TARGET_COL].to_numpy()
                X_val = train_df.loc[val_idx, feats].to_numpy()
                y_val = train_df.loc[val_idx, TARGET_COL].to_numpy()

                rf = RandomForestRegressor(
                    n_estimators=300,
                    min_samples_leaf=min_leaf,
                    max_features=max_feat,
                    max_depth=max_depth,
                    random_state=42,
                    n_jobs=-1,
                )
                rf.fit(X_tr, y_tr)
                val_preds = rf.predict(X_val)

                f_metrics = compute_metrics(y_val, val_preds)

                fold_rmses.append(f_metrics["rmse"])
                fold_biases.append(f_metrics["bias"])
                fold_ubrmses.append(f_metrics["ubrmse"])
                fold_rs.append(f_metrics["pearson_r"])
                fold_delta_rs.append(f_metrics["delta_r"])

                oof_preds.extend(val_preds)
                oof_actual.extend(y_val)

                f_info = fold_info[fold_idx]
                fold_records.append({
                    "aggregation": agg_name,
                    "min_samples_leaf": min_leaf,
                    "max_features": max_feat,
                    "max_depth": depth_str,
                    "is_baseline": is_base,
                    "fold": fold_idx + 1,
                    "n_train": f_info["n_train"],
                    "train_start_date": f_info["tr_start"],
                    "train_end_date": f_info["tr_end"],
                    "n_val": f_info["n_val"],
                    "val_start_date": f_info["val_start"],
                    "val_end_date": f_info["val_end"],
                    "rmse": f_metrics["rmse"],
                    "bias": f_metrics["bias"],
                    "ubrmse": f_metrics["ubrmse"],
                    "pearson_r": f_metrics["pearson_r"],
                    "delta_r": f_metrics["delta_r"],
                })

            oof_metrics = compute_metrics(np.array(oof_actual), np.array(oof_preds))

            valid_rs = [r for r in fold_rs if np.isfinite(r)]
            mean_fold_r = float(np.mean(valid_rs)) if valid_rs else np.nan
            std_fold_r = float(np.std(valid_rs)) if len(valid_rs) > 1 else np.nan

            valid_delta_rs = [dr for dr in fold_delta_rs if np.isfinite(dr)]
            mean_fold_delta_r = float(np.mean(valid_delta_rs)) if valid_delta_rs else np.nan

            summary_records.append({
                "aggregation": agg_name,
                "min_samples_leaf": min_leaf,
                "max_features": max_feat,
                "max_depth": depth_str,
                "is_baseline": is_base,
                "mean_fold_rmse": float(np.mean(fold_rmses)),
                "std_fold_rmse": float(np.std(fold_rmses)),
                "mean_fold_bias": float(np.mean(fold_biases)),
                "std_fold_bias": float(np.std(fold_biases)),
                "mean_fold_ubrmse": float(np.mean(fold_ubrmses)),
                "std_fold_ubrmse": float(np.std(fold_ubrmses)),
                "mean_fold_pearson_r": mean_fold_r,
                "std_fold_pearson_r": std_fold_r,
                "mean_fold_delta_r": mean_fold_delta_r,
                "pooled_oof_rmse": oof_metrics["rmse"],
                "pooled_oof_bias": oof_metrics["bias"],
                "pooled_oof_ubrmse": oof_metrics["ubrmse"],
                "pooled_oof_pearson_r": oof_metrics["pearson_r"],
                "pooled_oof_delta_r": oof_metrics["delta_r"],
            })

    df_fold_metrics = pd.DataFrame(fold_records)
    df_summary = pd.DataFrame(summary_records)

    OUT_FOLD_METRICS_CSV.parent.mkdir(parents=True, exist_ok=True)
    df_fold_metrics.to_csv(OUT_FOLD_METRICS_CSV, index=False)
    df_summary.to_csv(OUT_SUMMARY_CSV, index=False)

    print("\n" + "=" * 80)
    print("SAVED TEMPORAL CV ARTIFACTS:")
    print(f"  - Fold Metrics: {OUT_FOLD_METRICS_CSV} ({len(df_fold_metrics)} rows)")
    print(f"  - Summary:      {OUT_SUMMARY_CSV} ({len(df_summary)} rows)")
    print("=" * 80)


if __name__ == "__main__":
    run_temporal_cv()
