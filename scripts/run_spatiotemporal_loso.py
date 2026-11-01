"""
Spatio-Temporal Leave-One-Station-Out (LOSO) Cross-Validation and Evaluation.

Boundary: July 1, 2016 (DOY 183)
  - Training Period: 2016-01-01 to 2016-06-30 (DOY 1 - 182) on 2 training stations.
  - Testing Period:  2016-07-01 to 2016-12-31 (DOY 183 - 366) on held-out station.

Folds:
  - Fold 1: Train on P041 + P043 (Pre-Jul, N=207) -> Test on P037 (Post-Jul, N=145)
  - Fold 2: Train on P041 + P037 (Pre-Jul, N=228) -> Test on P043 (Post-Jul, N=138)
  - Fold 3: Train on P037 + P043 (Pre-Jul, N=241) -> Test on P041 (Post-Jul, N=150)
  - Pooled: N_test = 433 daily observations across all 3 stations.

Methods Evaluated:
  1. Mean Climatology Baseline (Weighted & Unweighted)
  2. Precipitation OLS Baseline (Weighted & Unweighted) using daily, 3-day, and 7-day precipitation
  3. Random Forest (RF): Model A (Primary), Model B, Model C (Secondary Ablations)
  4. XGBoost: Model A (Primary), Model B, Model C (Secondary Ablations)

Safeguards & Leakage Controls:
  - Reference Target: PBO_H2O surface soil moisture (0-5 cm, Quality Flag 'G').
  - Training sample weights: w(s, i) = (N_train / 2) / N_{s, train} (50% / 50% loss per station).
  - Preprocessing and model fitting strictly on pre-July training data of non-held-out stations.
  - Zero target data from held-out station or future post-July period used in training.
  - Strictly backward-looking precipitation rolling windows (daily, 3d, 7d) without zero replacement.
  - Calendar Gap-Respecting Moving Block Bootstrap (block length = 7 calendar days, B = 1000) for 95% CIs.
  - Cross-experiment comparison with All-Available Spatial LOSO and Same-Time Spatial LOSO.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import xgboost as xgb

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
HARMONIZED_CSV = PROCESSED_DIR / "loso_harmonized_dataset_median.csv"
SPATIAL_METRICS_CSV = PROCESSED_DIR / "loso_primary_spatial_metrics.csv"
SPATIAL_PREDS_CSV = PROCESSED_DIR / "loso_primary_spatial_predictions.csv"
SAME_TIME_METRICS_CSV = PROCESSED_DIR / "loso_same_time_metrics.csv"

OUT_PRED_CSV = PROCESSED_DIR / "loso_spatiotemporal_predictions.csv"
OUT_METRICS_CSV = PROCESSED_DIR / "loso_spatiotemporal_metrics.csv"
OUT_ABLATION_CSV = PROCESSED_DIR / "loso_spatiotemporal_feature_ablation.csv"
OUT_WEIGHTING_CSV = PROCESSED_DIR / "loso_spatiotemporal_weighting_sensitivity.csv"
OUT_COMPARISON_CSV = PROCESSED_DIR / "loso_spatiotemporal_vs_spatial_comparison.csv"

SPLIT_DATE = pd.to_datetime("2016-07-01")

# Hyperparameter locks (matching primary LOSO and P041 benchmarks)
RF_PARAMS = {
    "n_estimators": 300,
    "min_samples_leaf": 3,
    "max_features": 1.0,
    "max_depth": None,
    "random_state": 42,
    "n_jobs": -1,
}

XGB_PARAMS = {
    "learning_rate": 0.05,
    "max_depth": 2,
    "n_estimators": 150,
    "min_child_weight": 1,
    "subsample": 1.0,
    "colsample_bytree": 1.0,
    "random_state": 42,
    "objective": "reg:squarederror",
    "n_jobs": -1,
    "tree_method": "hist",
}

FEATURE_SETS = {
    "Model_A": ["rh", "amp"],
    "Model_B": ["rh", "amp", "phase_cos", "phase_sin"],
    "Model_C": ["rh", "amp", "phase_cos", "phase_sin", "phase_resultant_length", "clean_arc_count", "qc_pass_rate"],
}


def make_calendar_gap_blocks(
    dates: pd.Series | pd.DatetimeIndex,
    block_days: int = 7,
) -> List[np.ndarray]:
    """
    Generate candidate block index slices based on calendar days,
    respecting actual non-consecutive chronological gaps.
    """
    dates_dt = pd.to_datetime(dates).to_numpy()
    n = len(dates_dt)
    blocks = []
    one_day_ns = np.timedelta64(1, 'D')
    block_span_ns = np.timedelta64(block_days, 'D')

    for i in range(n):
        t_start = dates_dt[i]
        t_end = t_start + block_span_ns
        block_indices = [i]
        for j in range(i + 1, n):
            t_curr = dates_dt[j]
            t_prev = dates_dt[j - 1]
            if t_curr >= t_end:
                break
            # If gap between consecutive active dates >= block_days, truncate
            if (t_curr - t_prev) >= block_span_ns:
                break
            block_indices.append(j)
        blocks.append(np.array(block_indices, dtype=int))
    return blocks


def calendar_block_bootstrap_ci(
    dates: pd.Series | pd.DatetimeIndex,
    actual: np.ndarray,
    predicted: np.ndarray,
    block_days: int = 7,
    n_bootstraps: int = 1000,
    seed: int = 42,
) -> Dict[str, Tuple[float, float]]:
    """
    Compute 95% Confidence Intervals using Moving Block Bootstrap
    respecting calendar time gaps to prevent artificial joining across gaps.
    """
    rng = np.random.RandomState(seed)
    n = len(actual)
    if n < block_days * 2:
        return {}

    candidate_blocks = make_calendar_gap_blocks(dates, block_days=block_days)
    n_cand = len(candidate_blocks)
    if n_cand == 0:
        return {}

    rmse_boot, mae_boot, bias_boot, ubrmse_boot, r_boot = [], [], [], [], []

    for _ in range(n_bootstraps):
        sampled_indices = []
        while len(sampled_indices) < n:
            b_idx = rng.randint(0, n_cand)
            sampled_indices.extend(candidate_blocks[b_idx])
        sample_indices = np.array(sampled_indices[:n], dtype=int)

        act_b = actual[sample_indices]
        pred_b = predicted[sample_indices]

        err = pred_b - act_b
        rmse_val = np.sqrt(np.mean(err**2))
        mae_val = np.mean(np.abs(err))
        bias_val = np.mean(err)
        ubrmse_val = np.sqrt(np.mean((err - bias_val)**2))

        std_act = np.std(act_b)
        std_pred = np.std(pred_b)
        if std_act > 1e-9 and std_pred > 1e-9:
            r_val = np.corrcoef(act_b, pred_b)[0, 1]
        else:
            r_val = np.nan

        rmse_boot.append(rmse_val)
        mae_boot.append(mae_val)
        bias_boot.append(bias_val)
        ubrmse_boot.append(ubrmse_val)
        r_boot.append(r_val)

    def get_ci(vals: List[float]) -> Tuple[float, float]:
        v = np.array(vals)
        v = v[np.isfinite(v)]
        if len(v) == 0:
            return np.nan, np.nan
        return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))

    return {
        "rmse_ci": get_ci(rmse_boot),
        "mae_ci": get_ci(mae_boot),
        "bias_ci": get_ci(bias_boot),
        "ubrmse_ci": get_ci(ubrmse_boot),
        "r_ci": get_ci(r_boot),
    }


def compute_metrics(
    dates: pd.Series | pd.DatetimeIndex,
    actual: np.ndarray,
    predicted: np.ndarray,
    block_days: int = 7,
    n_bootstraps: int = 1000,
) -> Dict[str, Any]:
    """Compute hydrological metrics and gap-respecting bootstrap 95% confidence intervals."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    n = len(actual)

    error = predicted - actual
    rmse = float(np.sqrt(np.mean(error**2)))
    mae = float(np.mean(np.abs(error)))
    bias = float(np.mean(error))
    ubrmse = float(np.sqrt(np.mean((error - bias)**2)))

    std_act = float(np.std(actual))
    std_pred = float(np.std(predicted))
    if std_act > 1e-9 and std_pred > 1e-9 and n > 2:
        r = float(np.corrcoef(actual, predicted)[0, 1])
    else:
        r = np.nan

    ci_dict = calendar_block_bootstrap_ci(
        dates, actual, predicted, block_days=block_days, n_bootstraps=n_bootstraps
    )

    return {
        "n": int(n),
        "rmse": rmse,
        "rmse_ci_lower": ci_dict.get("rmse_ci", (np.nan, np.nan))[0],
        "rmse_ci_upper": ci_dict.get("rmse_ci", (np.nan, np.nan))[1],
        "mae": mae,
        "mae_ci_lower": ci_dict.get("mae_ci", (np.nan, np.nan))[0],
        "mae_ci_upper": ci_dict.get("mae_ci", (np.nan, np.nan))[1],
        "bias": bias,
        "bias_ci_lower": ci_dict.get("bias_ci", (np.nan, np.nan))[0],
        "bias_ci_upper": ci_dict.get("bias_ci", (np.nan, np.nan))[1],
        "ubrmse": ubrmse,
        "ubrmse_ci_lower": ci_dict.get("ubrmse_ci", (np.nan, np.nan))[0],
        "ubrmse_ci_upper": ci_dict.get("ubrmse_ci", (np.nan, np.nan))[1],
        "pearson_r": r,
        "r_ci_lower": ci_dict.get("r_ci", (np.nan, np.nan))[0],
        "r_ci_upper": ci_dict.get("r_ci", (np.nan, np.nan))[1],
    }


def execute_spatiotemporal_loso():
    print("=" * 85)
    print("EXECUTING CONDITIONAL SPATIO-TEMPORAL LOSO EXPERIMENT (JULY 1 BOUNDARY)")
    print("=" * 85)

    df_full = pd.read_csv(HARMONIZED_CSV)
    df_full["date"] = pd.to_datetime(df_full["date"])
    df_full = df_full.sort_values(["station", "date"]).reset_index(drop=True)

    # 1. Strictly backward-looking precipitation rolling windows per station
    # Computed on full continuous calendar BEFORE filtering for usable dates
    # min_periods = window ensures no silent zero-filling or imputation
    precip_frames = []
    for st, grp in df_full.groupby("station"):
        g = grp.copy().sort_values("date").reset_index(drop=True)
        g["p_daily"] = g["precip_mm"]
        g["p_3d"] = g["precip_mm"].rolling(window=3, min_periods=3).sum()
        g["p_7d"] = g["precip_mm"].rolling(window=7, min_periods=7).sum()
        precip_frames.append(g)
    df_enriched = pd.concat(precip_frames).sort_values(["station", "date"]).reset_index(drop=True)

    # 2. Filter for usable paired observations
    df = df_enriched[df_enriched["usable_for_loso"]].copy().reset_index(drop=True)
    print(f"Loaded harmonized dataset: {len(df)} total paired station-days.")

    folds = [
        {"fold_id": "Fold_1", "test_station": "p037", "train_stations": ["p041", "p043"]},
        {"fold_id": "Fold_2", "test_station": "p043", "train_stations": ["p041", "p037"]},
        {"fold_id": "Fold_3", "test_station": "p041", "train_stations": ["p037", "p043"]},
    ]

    all_pred_dfs = []
    all_metrics = []

    for fold in folds:
        fold_id = fold["fold_id"]
        test_st = fold["test_station"]
        train_sts = fold["train_stations"]

        print(f"\n" + "-" * 75)
        print(f"RUNNING {fold_id.upper()}: Test on {test_st.upper()} (Late: Jul 1 - Dec 31) | Train on {[s.upper() for s in train_sts]} (Early: Jan 1 - Jun 30)")
        print("-" * 75)

        # Training set: Pre-July 1 from the other two stations only
        train_df = df[
            (df["station"].isin(train_sts)) & (df["date"] < SPLIT_DATE)
        ].copy().reset_index(drop=True)

        # Testing set: Post-July 1 from the held-out station only
        test_df = df[
            (df["station"] == test_st) & (df["date"] >= SPLIT_DATE)
        ].copy().reset_index(drop=True)

        n_train = len(train_df)
        n_test = len(test_df)
        print(f"  Training set (Pre-July): {n_train} obs | Test set (Post-July): {n_test} obs")

        # 1. Compute Equal Station Sample Weights for the 2 training stations
        st1, st2 = train_sts[0], train_sts[1]
        n_st1 = len(train_df[train_df["station"] == st1])
        n_st2 = len(train_df[train_df["station"] == st2])
        w_st1 = (n_train / 2.0) / n_st1
        w_st2 = (n_train / 2.0) / n_st2

        train_df["sample_weight_equal"] = np.where(
            train_df["station"] == st1, w_st1, w_st2
        )
        train_df["sample_weight_unweighted"] = 1.0

        print(f"  Training Station N: {st1.upper()}={n_st1} (w={w_st1:.4f}), {st2.upper()}={n_st2} (w={w_st2:.4f})")
        print(f"  Loss contribution: {st1.upper()}={w_st1*n_st1/n_train*100:.1f}%, {st2.upper()}={w_st2*n_st2/n_train*100:.1f}%")

        y_train = train_df["pbo_sm"].to_numpy()
        y_test = test_df["pbo_sm"].to_numpy()
        dates_test = test_df["date"]

        # Initialize prediction record for this test fold
        pred_record = test_df[["station", "name", "date", "year", "doy", "pbo_sm", "precip_mm", "p_daily", "p_3d", "p_7d"]].copy()
        pred_record["fold"] = fold_id
        pred_record["n_train"] = n_train

        # -------------------------------------------------------------
        # 1. Climatology Baseline (Pre-July Training Mean)
        # -------------------------------------------------------------
        for weight_type, w_col in [("weighted", "sample_weight_equal"), ("unweighted", "sample_weight_unweighted")]:
            weights = train_df[w_col].to_numpy()
            clim_pred_val = float(np.sum(weights * y_train) / np.sum(weights))
            pred_col = f"pred_climatology_{weight_type}"
            pred_record[pred_col] = clim_pred_val

            m = compute_metrics(dates_test, y_test, np.full(n_test, clim_pred_val))
            m.update({
                "fold": fold_id,
                "test_station": test_st.upper(),
                "model_family": "Climatology",
                "feature_set": "Baseline",
                "weighting": weight_type,
                "n_train": n_train,
                "n_test": n_test,
                "split_boundary": "2016-07-01",
            })
            all_metrics.append(m)

        # -------------------------------------------------------------
        # 2. Precipitation OLS Baseline (Pre-July Training Fit)
        # Predictors: [p_daily, p_3d, p_7d]
        # Missing precipitation is preserved as NaN (not zero-filled)
        # -------------------------------------------------------------
        p_cols = ["p_daily", "p_3d", "p_7d"]
        X_p_train = train_df[p_cols].to_numpy()
        assert not np.isnan(X_p_train).any(), f"NaN in training precipitation for {fold_id}"

        # In test set, keep NaNs as NaNs (do not impute or zero-fill)
        valid_p_test_mask = test_df[p_cols].notnull().all(axis=1)
        n_p_valid = int(valid_p_test_mask.sum())
        n_p_excluded = n_test - n_p_valid
        print(f"  Precipitation OLS test coverage: {n_p_valid}/{n_test} valid dates ({n_p_excluded} excluded due to NaN)")

        for weight_type, w_col in [("weighted", "sample_weight_equal"), ("unweighted", "sample_weight_unweighted")]:
            weights = train_df[w_col].to_numpy()
            ols = LinearRegression()
            ols.fit(X_p_train, y_train, sample_weight=weights)

            # Predict on full test set where valid; NaN where incomplete
            p_preds_full = np.full(n_test, np.nan)
            if n_p_valid > 0:
                p_preds_valid = ols.predict(test_df.loc[valid_p_test_mask, p_cols].to_numpy())
                p_preds_full[valid_p_test_mask] = p_preds_valid

            pred_col = f"pred_precip_ols_{weight_type}"
            pred_record[pred_col] = p_preds_full

            # Compute metrics on valid predictions
            y_test_valid = y_test[valid_p_test_mask]
            dates_test_valid = dates_test[valid_p_test_mask]
            p_preds_valid_eval = p_preds_full[valid_p_test_mask]

            m = compute_metrics(dates_test_valid, y_test_valid, p_preds_valid_eval)
            m.update({
                "fold": fold_id,
                "test_station": test_st.upper(),
                "model_family": "Precipitation_OLS",
                "feature_set": "Baseline",
                "weighting": weight_type,
                "n_train": n_train,
                "n_test": n_p_valid,
                "split_boundary": "2016-07-01",
            })
            all_metrics.append(m)

        # -------------------------------------------------------------
        # 3. Machine Learning: Random Forest & XGBoost across Ablations
        # -------------------------------------------------------------
        for model_code, f_cols in FEATURE_SETS.items():
            X_tr = train_df[f_cols].to_numpy()
            X_te = test_df[f_cols].to_numpy()

            assert not np.isnan(X_tr).any(), f"NaN in X_tr for {model_code}"
            assert not np.isnan(X_te).any(), f"NaN in X_te for {model_code}"

            for weight_type, w_col in [("weighted", "sample_weight_equal"), ("unweighted", "sample_weight_unweighted")]:
                weights = train_df[w_col].to_numpy()

                # a) Random Forest
                rf = RandomForestRegressor(**RF_PARAMS)
                rf.fit(X_tr, y_train, sample_weight=weights)
                rf_preds = rf.predict(X_te)
                pred_col_rf = f"pred_rf_{model_code}_{weight_type}"
                pred_record[pred_col_rf] = rf_preds

                m_rf = compute_metrics(dates_test, y_test, rf_preds)
                m_rf.update({
                    "fold": fold_id,
                    "test_station": test_st.upper(),
                    "model_family": "Random_Forest",
                    "feature_set": model_code,
                    "weighting": weight_type,
                    "n_train": n_train,
                    "n_test": n_test,
                    "split_boundary": "2016-07-01",
                })
                all_metrics.append(m_rf)

                # b) XGBoost
                xgb_model = xgb.XGBRegressor(**XGB_PARAMS)
                xgb_model.fit(X_tr, y_train, sample_weight=weights)
                xgb_preds = xgb_model.predict(X_te)
                pred_col_xgb = f"pred_xgboost_{model_code}_{weight_type}"
                pred_record[pred_col_xgb] = xgb_preds

                m_xgb = compute_metrics(dates_test, y_test, xgb_preds)
                m_xgb.update({
                    "fold": fold_id,
                    "test_station": test_st.upper(),
                    "model_family": "XGBoost",
                    "feature_set": model_code,
                    "weighting": weight_type,
                    "n_train": n_train,
                    "n_test": n_test,
                    "split_boundary": "2016-07-01",
                })
                all_metrics.append(m_xgb)

        all_pred_dfs.append(pred_record)

    # =========================================================================
    # POOLED EVALUATION ACROSS ALL THREE FOLDS COMBINED (N = 433)
    # =========================================================================
    full_pred_df = pd.concat(all_pred_dfs, ignore_index=True).sort_values(["station", "date"]).reset_index(drop=True)
    full_pred_df.to_csv(OUT_PRED_CSV, index=False)
    print(f"\n[SAVED] Spatio-Temporal Predictions (N={len(full_pred_df)}): {OUT_PRED_CSV.relative_to(PROJECT_ROOT)}")

    print("\n" + "=" * 75)
    print("COMPUTING POOLED METRICS ACROSS ALL THREE FOLDS COMBINED (N = 433)")
    print("=" * 75)

    pred_cols_to_evaluate = [c for c in full_pred_df.columns if c.startswith("pred_")]
    y_pooled = full_pred_df["pbo_sm"].to_numpy()
    dates_pooled = full_pred_df["date"]

    for p_col in pred_cols_to_evaluate:
        # Determine model family, feature set, and weighting from column name
        parts = p_col.split("_")
        if "climatology" in p_col:
            fam = "Climatology"
            f_set = "Baseline"
            w_type = parts[-1]
        elif "precip" in p_col:
            fam = "Precipitation_OLS"
            f_set = "Baseline"
            w_type = parts[-1]
        elif "rf" in p_col:
            fam = "Random_Forest"
            f_set = f"{parts[2]}_{parts[3]}"
            w_type = parts[-1]
        elif "xgboost" in p_col:
            fam = "XGBoost"
            f_set = f"{parts[2]}_{parts[3]}"
            w_type = parts[-1]
        else:
            continue

        p_vals = full_pred_df[p_col].to_numpy()

        # Handle missing precipitation in pooled evaluation
        valid_mask = np.isfinite(p_vals)
        y_eval = y_pooled[valid_mask]
        dates_eval = dates_pooled[valid_mask]
        p_eval = p_vals[valid_mask]

        m_pool = compute_metrics(dates_eval, y_eval, p_eval)
        m_pool.update({
            "fold": "Pooled_Overall",
            "test_station": "ALL_3_STATIONS",
            "model_family": fam,
            "feature_set": f_set,
            "weighting": w_type,
            "n_train": 771 - len(full_pred_df),  # 338 total pre-July training samples
            "n_test": int(valid_mask.sum()),
            "split_boundary": "2016-07-01",
        })
        all_metrics.append(m_pool)

    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(OUT_METRICS_CSV, index=False)
    print(f"[SAVED] Spatio-Temporal Metrics: {OUT_METRICS_CSV.relative_to(PROJECT_ROOT)}")

    # -----------------------------------------------------------------
    # Save Feature Ablation Summary Table (Model A vs B vs C, Weighted)
    # -----------------------------------------------------------------
    ablation_df = metrics_df[
        (metrics_df["weighting"] == "weighted")
        & (metrics_df["model_family"].isin(["Random_Forest", "XGBoost"]))
    ][["fold", "test_station", "model_family", "feature_set", "n_test", "rmse", "mae", "bias", "ubrmse", "pearson_r"]].sort_values(["model_family", "fold", "feature_set"])
    ablation_df.to_csv(OUT_ABLATION_CSV, index=False)
    print(f"[SAVED] Spatio-Temporal Feature Ablation: {OUT_ABLATION_CSV.relative_to(PROJECT_ROOT)}")

    # -----------------------------------------------------------------
    # Save Weighting Sensitivity Table (Equal Weighted vs Unweighted)
    # -----------------------------------------------------------------
    weighting_df = metrics_df[
        (metrics_df["feature_set"] == "Model_A")
        | (metrics_df["model_family"].isin(["Climatology", "Precipitation_OLS"]))
    ][["fold", "test_station", "model_family", "feature_set", "weighting", "n_train", "n_test", "rmse", "mae", "bias", "ubrmse", "pearson_r"]].sort_values(["model_family", "fold", "weighting"])
    weighting_df.to_csv(OUT_WEIGHTING_CSV, index=False)
    print(f"[SAVED] Spatio-Temporal Weighting Sensitivity: {OUT_WEIGHTING_CSV.relative_to(PROJECT_ROOT)}")

    # -----------------------------------------------------------------
    # Multi-Experiment Comparison Table:
    # 1. Full Spatial LOSO (All-Available Dates, N=771)
    # 2. Same-Time Spatial LOSO (Synchronized Common Dates, N=648)
    # 3. Spatial LOSO Post-July Subset Control (N=433)
    # 4. Spatio-Temporal LOSO (Early-Train / Late-Test, N=433)
    # -----------------------------------------------------------------
    if SPATIAL_METRICS_CSV.exists() and SPATIAL_PREDS_CSV.exists():
        spat_metrics = pd.read_csv(SPATIAL_METRICS_CSV)
        spat_preds = pd.read_csv(SPATIAL_PREDS_CSV)
        spat_preds["date"] = pd.to_datetime(spat_preds["date"])

        same_time_metrics = pd.read_csv(SAME_TIME_METRICS_CSV) if SAME_TIME_METRICS_CSV.exists() else None

        comp_records = []
        for fold in ["Fold_1", "Fold_2", "Fold_3", "Pooled_Overall"]:
            for fam in ["Climatology", "Precipitation_OLS", "Random_Forest", "XGBoost"]:
                fset = "Baseline" if "Baseline" in fam or fam in ["Climatology", "Precipitation_OLS"] else "Model_A"
                
                # 1. Full Spatial LOSO metric (trained on all 2016, tested on all 2016)
                m_spat = spat_metrics[
                    (spat_metrics["fold"] == fold)
                    & (spat_metrics["model_family"] == fam)
                    & (spat_metrics["feature_set"] == fset)
                    & (spat_metrics["weighting"] == "weighted")
                ]
                spat_rmse = m_spat["rmse"].values[0] if len(m_spat) > 0 else np.nan
                spat_ubrmse = m_spat["ubrmse"].values[0] if len(m_spat) > 0 else np.nan
                spat_r = m_spat["pearson_r"].values[0] if len(m_spat) > 0 else np.nan
                spat_n = m_spat["n"].values[0] if len(m_spat) > 0 else np.nan

                # 2. Same-Time Spatial LOSO metric (trained and tested on 216 common dates)
                same_time_rmse, same_time_ubrmse, same_time_r, same_time_n = np.nan, np.nan, np.nan, np.nan
                if same_time_metrics is not None:
                    m_st = same_time_metrics[
                        (same_time_metrics["fold"] == fold)
                        & (same_time_metrics["model_family"] == fam)
                        & (same_time_metrics["feature_set"] == fset)
                        & (same_time_metrics["weighting"] == "weighted")
                    ]
                    if len(m_st) > 0:
                        same_time_rmse = m_st["rmse"].values[0]
                        same_time_ubrmse = m_st["ubrmse"].values[0]
                        same_time_r = m_st["pearson_r"].values[0]
                        same_time_n = m_st["n"].values[0]

                # 3. Spatial LOSO Post-July Subset Control (Full model evaluated on post-July subset)
                if fold == "Pooled_Overall":
                    spat_sub = spat_preds[spat_preds["date"] >= SPLIT_DATE].copy()
                else:
                    spat_sub = spat_preds[(spat_preds["fold"] == fold) & (spat_preds["date"] >= SPLIT_DATE)].copy()

                p_col_map = {
                    "Climatology": "pred_climatology_weighted",
                    "Precipitation_OLS": "pred_precip_ols_weighted",
                    "Random_Forest": "pred_rf_Model_A_weighted",
                    "XGBoost": "pred_xgboost_Model_A_weighted",
                }
                
                # Exclude NaNs if any (e.g. Precipitation OLS)
                eval_mask = spat_sub[p_col_map[fam]].notnull()
                y_sub = spat_sub.loc[eval_mask, "pbo_sm"].to_numpy()
                p_sub = spat_sub.loc[eval_mask, p_col_map[fam]].to_numpy()
                d_sub = spat_sub.loc[eval_mask, "date"]

                m_spat_post_jul = compute_metrics(d_sub, y_sub, p_sub)

                # 4. Spatio-Temporal LOSO metric (Early-Train / Late-Test)
                m_st_loso = metrics_df[
                    (metrics_df["fold"] == fold)
                    & (metrics_df["model_family"] == fam)
                    & (metrics_df["feature_set"] == fset)
                    & (metrics_df["weighting"] == "weighted")
                ]
                st_rmse = m_st_loso["rmse"].values[0] if len(m_st_loso) > 0 else np.nan
                st_ubrmse = m_st_loso["ubrmse"].values[0] if len(m_st_loso) > 0 else np.nan
                st_r = m_st_loso["pearson_r"].values[0] if len(m_st_loso) > 0 else np.nan
                st_n = m_st_loso["n_test"].values[0] if len(m_st_loso) > 0 else np.nan

                comp_records.append({
                    "fold": fold,
                    "model_family": fam,
                    "feature_set": fset,
                    "spatial_all_dates_rmse": spat_rmse,
                    "spatial_all_dates_ubrmse": spat_ubrmse,
                    "spatial_all_dates_r": spat_r,
                    "spatial_all_dates_n": spat_n,
                    "same_time_spatial_rmse": same_time_rmse,
                    "same_time_spatial_ubrmse": same_time_ubrmse,
                    "same_time_spatial_r": same_time_r,
                    "same_time_spatial_n": same_time_n,
                    "spatial_post_jul_control_rmse": m_spat_post_jul["rmse"],
                    "spatial_post_jul_control_ubrmse": m_spat_post_jul["ubrmse"],
                    "spatial_post_jul_control_r": m_spat_post_jul["pearson_r"],
                    "spatial_post_jul_control_n": m_spat_post_jul["n"],
                    "spatiotemporal_rmse": st_rmse,
                    "spatiotemporal_ubrmse": st_ubrmse,
                    "spatiotemporal_r": st_r,
                    "spatiotemporal_n": st_n,
                    "delta_rmse_spatiotemporal_vs_control": st_rmse - m_spat_post_jul["rmse"],
                    "delta_rmse_spatiotemporal_vs_all_dates": st_rmse - spat_rmse,
                })

        comp_df = pd.DataFrame(comp_records)
        comp_df.to_csv(OUT_COMPARISON_CSV, index=False)
        print(f"[SAVED] Cross-Experiment Comparison: {OUT_COMPARISON_CSV.relative_to(PROJECT_ROOT)}")

    print("\n" + "=" * 85)
    print("SPATIO-TEMPORAL LOSO EXPERIMENT COMPLETE.")
    print("=" * 85)


if __name__ == "__main__":
    execute_spatiotemporal_loso()
