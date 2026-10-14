import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Ensure scripts directory is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from run_climatology_baseline import (
    compute_metrics,
    run_climatology_baseline,
    split_chronological_holdout,
)


def test_split_chronological_holdout_ratio():
    """Verify that split_chronological_holdout respects ordering and 80/20 ratio."""
    dates = pd.date_range("2016-01-01", periods=10, freq="D")
    df = pd.DataFrame({
        "date": dates.strftime("%Y-%m-%d"),
        "cosmos_sm_median": [0.1, 0.2, np.nan, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    })

    train_df, test_df = split_chronological_holdout(
        df, date_col="date", target_col="cosmos_sm_median", train_fraction=0.8
    )

    # 9 valid observations -> int(9 * 0.8) = 7 train, 2 test
    assert len(train_df) == 7
    assert len(test_df) == 2
    # Ensure chronological order is maintained and no overlap
    assert train_df["date"].max() < test_df["date"].min()
    assert list(train_df["cosmos_sm_median"]) == [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    assert list(test_df["cosmos_sm_median"]) == [0.8, 0.9]


def test_no_target_leakage_in_climatology_prediction():
    """Verify that the test prediction is strictly the training mean and does not use test values."""
    train_vals = [0.10, 0.12, 0.14, 0.16]
    test_vals = [0.50, 0.60]  # substantially different values

    expected_train_mean = float(np.mean(train_vals))

    df = pd.DataFrame({
        "date": [f"2016-01-0{i+1}" for i in range(len(train_vals) + len(test_vals))],
        "cosmos_sm_median": train_vals + test_vals,
    })

    train_df, test_df = split_chronological_holdout(
        df, date_col="date", target_col="cosmos_sm_median", train_fraction=len(train_vals) / len(df)
    )

    pred = float(train_df["cosmos_sm_median"].mean())
    assert pred == pytest.approx(expected_train_mean)
    # Ensure test target is completely independent of the prediction
    assert pred != pytest.approx(float(test_df["cosmos_sm_median"].mean()))


def test_compute_metrics_exact_values():
    """Verify RMSE, Bias, and ubRMSE definitions."""
    actual = np.array([0.10, 0.20, 0.30])
    # Constant prediction
    predicted = np.array([0.25, 0.25, 0.25])

    # errors: [0.15, 0.05, -0.05]
    # mean error (bias) = (0.15 + 0.05 - 0.05) / 3 = 0.15 / 3 = 0.05
    # squared errors: [0.0225, 0.0025, 0.0025] -> mean = 0.0275 / 3 = 0.0091666667
    # rmse = sqrt(0.0091666667) = 0.09574271
    # ubrmse = sqrt(rmse^2 - bias^2) = sqrt(0.0091666667 - 0.0025) = sqrt(0.0066666667) = 0.08164966
    metrics = compute_metrics(actual, predicted)

    assert metrics["bias"] == pytest.approx(0.05, rel=1e-5)
    assert metrics["rmse"] == pytest.approx(np.sqrt(0.0275 / 3), rel=1e-5)
    assert metrics["ubrmse"] == pytest.approx(np.sqrt(0.0275 / 3 - 0.05**2), rel=1e-5)


def test_climatology_baseline_integration(tmp_path):
    """Run baseline end-to-end on a synthetic dataset and verify export structure."""
    test_csv = tmp_path / "daily_test.csv"
    out_csv = tmp_path / "climatology_out.csv"

    dates = pd.date_range("2016-01-01", periods=10, freq="D")
    df = pd.DataFrame({
        "date": dates.strftime("%Y-%m-%d"),
        "year": [2016] * 10,
        "doy": list(range(1, 11)),
        "cosmos_sm_median": [0.10, 0.12, 0.14, 0.16, 0.18, 0.20, 0.22, 0.24, 0.26, 0.28],
    })
    df.to_csv(test_csv, index=False)

    summary = run_climatology_baseline(
        input_path=test_csv,
        output_path=out_csv,
        target_col="cosmos_sm_median",
        train_fraction=0.8,
    )

    assert summary["train_count"] == 8
    assert summary["test_count"] == 2
    assert summary["train_mean"] == pytest.approx(0.17)
    assert summary["test_mean"] == pytest.approx(0.27)

    # Check exported CSV
    exported = pd.read_csv(out_csv)
    assert len(exported) == 2
    assert list(exported.columns) == [
        "date",
        "year",
        "doy",
        "actual_cosmos_sm",
        "climatology_prediction",
        "prediction_error",
    ]
    assert np.allclose(exported["climatology_prediction"], 0.17)
    assert np.allclose(exported["prediction_error"], 0.17 - exported["actual_cosmos_sm"])
