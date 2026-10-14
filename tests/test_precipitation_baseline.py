import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from run_precipitation_baseline import (
    compute_metrics,
    fit_ols_regression,
    predict_ols_regression,
    run_precipitation_baseline,
)


def test_metrics_computation():
    """Verify RMSE, Bias, and ubRMSE definitions."""
    y_true = np.array([0.1, 0.2, 0.3])
    y_pred = np.array([0.15, 0.25, 0.35])  # constant bias of +0.05

    metrics = compute_metrics(y_true, y_pred)
    assert metrics["rmse"] == pytest.approx(0.05)
    assert metrics["bias"] == pytest.approx(0.05)
    assert metrics["ubrmse"] == pytest.approx(0.0, abs=1e-10)


def test_fit_and_predict_ols():
    """Verify OLS parameter estimation without scikit-learn."""
    # y = 0.5 + 2.0 * x1 - 1.0 * x2
    X = np.array([
        [1.0, 2.0],
        [2.0, 1.0],
        [3.0, 0.5],
        [0.0, 3.0],
        [1.5, 1.5],
    ])
    y = 0.5 + 2.0 * X[:, 0] - 1.0 * X[:, 1]

    intercept, coeffs = fit_ols_regression(X, y)
    assert intercept == pytest.approx(0.5, abs=1e-7)
    assert coeffs[0] == pytest.approx(2.0, abs=1e-7)
    assert coeffs[1] == pytest.approx(-1.0, abs=1e-7)

    y_pred = predict_ols_regression(X, intercept, coeffs)
    np.testing.assert_allclose(y_pred, y, atol=1e-7)


def test_no_target_leakage():
    """Verify that OLS coefficients are trained strictly on training data."""
    X_train = np.array([[1.0], [2.0], [3.0]])
    y_train = np.array([0.1, 0.2, 0.3])

    X_test = np.array([[10.0], [20.0]])
    y_test = np.array([0.9, 0.95])  # test values should have zero influence

    intercept, coeffs = fit_ols_regression(X_train, y_train)

    # Intercept ~ 0.0, slope ~ 0.1
    assert intercept == pytest.approx(0.0, abs=1e-7)
    assert coeffs[0] == pytest.approx(0.1, abs=1e-7)

    # Prediction on test
    preds = predict_ols_regression(X_test, intercept, coeffs)
    np.testing.assert_allclose(preds, np.array([1.0, 2.0]), atol=1e-7)


def test_precipitation_baseline_outputs():
    """Verify end-to-end baseline runs and reproduces exact contract outputs."""
    precip_zip_path = Path("E:/MTP Data/P041_MARSHALL2_ISMN_SM_P_2016_2017.zip")
    cosmos_daily_path = PROJECT_ROOT / "data" / "processed" / "p041" / "p041_2016_daily_gnss_cosmos_validation.csv"
    output_predictions_path = PROJECT_ROOT / "data" / "processed" / "p041" / "p041_2016_precipitation_predictions.csv"

    results = run_precipitation_baseline(
        precip_zip_path=precip_zip_path,
        cosmos_daily_path=cosmos_daily_path,
        output_predictions_path=output_predictions_path,
    )

    # Check partition counts
    assert results["test_total_dates"] == 63
    assert results["train_used_dates"] == 242
    assert results["train_excluded_dates"] == 10

    # Check metrics are within expected physical/numerical ranges
    assert results["rmse"] == pytest.approx(0.051139, abs=1e-5)
    assert results["bias"] == pytest.approx(0.047658, abs=1e-5)
    assert results["ubrmse"] == pytest.approx(0.018545, abs=1e-5)

    # Check output file exists and has correct columns and row count
    assert output_predictions_path.exists()

    df_out = pd.read_csv(output_predictions_path)
    assert len(df_out) == 63
    expected_cols = [
        "date",
        "year",
        "doy",
        "actual_cosmos_sm",
        "precip_prediction",
        "prediction_error",
        "p_daily",
        "p_3d",
        "p_7d",
    ]
    assert list(df_out.columns) == expected_cols

    # Ensure no NaN predictions in the test set
    assert df_out["precip_prediction"].notna().all()


def test_exact_test_dates_match_climatology():
    """Verify that the 63 test dates match p041_2016_climatology_predictions.csv exactly."""
    clim_path = PROJECT_ROOT / "data" / "processed" / "p041" / "p041_2016_climatology_predictions.csv"
    precip_path = PROJECT_ROOT / "data" / "processed" / "p041" / "p041_2016_precipitation_predictions.csv"

    if clim_path.exists() and precip_path.exists():
        df_clim = pd.read_csv(clim_path)
        df_precip = pd.read_csv(precip_path)

        assert list(df_precip["date"]) == list(df_clim["date"])
        assert list(df_precip["actual_cosmos_sm"]) == pytest.approx(list(df_clim["actual_cosmos_sm"]))
