"""
Mean-Climatology Baseline for P041 2016 COSMOS Soil Moisture.

This script:
1. Loads the daily validation dataset from 'data/processed/p041/p041_2016_daily_gnss_cosmos_validation.csv'.
2. Identifies valid daily COSMOS soil moisture observations.
3. Sorts valid observations chronologically and splits into:
   - Training period: First 80% of valid observations.
   - Test period: Final 20% of valid observations.
4. Computes the training-period climatological mean using training targets only.
5. Predicts this constant mean for every test-period date.
6. Evaluates test-period performance metrics:
   - Root Mean Square Error (RMSE)
   - Mean Bias Error (Bias = prediction - observation)
   - Unbiased RMSE (ubRMSE = sqrt(RMSE^2 - Bias^2))
7. Saves the test-period predictions to 'data/processed/p041/p041_2016_climatology_predictions.csv'.
"""

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd


def compute_metrics(actual: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
    """
    Compute RMSE, Bias (prediction - observation), and ubRMSE.

    Parameters
    ----------
    actual : np.ndarray
        Observed ground-truth target values.
    predicted : np.ndarray
        Predicted values from the model.

    Returns
    -------
    dict
        Dictionary containing 'rmse', 'bias', and 'ubrmse'.
    """
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    error = predicted - actual
    rmse = float(np.sqrt(np.mean(error ** 2)))
    bias = float(np.mean(error))
    ubrmse = float(np.sqrt(np.maximum(0.0, rmse ** 2 - bias ** 2)))

    return {
        "rmse": rmse,
        "bias": bias,
        "ubrmse": ubrmse,
    }


def split_chronological_holdout(
    df: pd.DataFrame,
    date_col: str = "date",
    target_col: str = "cosmos_sm_median",
    train_fraction: float = 0.8,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Filter valid target observations, sort chronologically, and partition into
    train (first fraction) and test (remaining fraction).

    Parameters
    ----------
    df : pd.DataFrame
        Input daily validation DataFrame.
    date_col : str
        Column name for date.
    target_col : str
        Column name for the soil-moisture target.
    train_fraction : float
        Fraction of observations for the training set (default 0.8).

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        (train_df, test_df)
    """
    valid = df[df[target_col].notnull()].copy()
    valid = valid.sort_values(date_col).reset_index(drop=True)

    n_total = len(valid)
    if n_total == 0:
        raise ValueError(f"No valid observations found for target column '{target_col}'.")

    n_train = int(n_total * train_fraction)
    if n_train == 0 or n_train >= n_total:
        raise ValueError(
            f"Invalid split: n_total={n_total}, n_train={n_train} with fraction={train_fraction}."
        )

    train_df = valid.iloc[:n_train].copy().reset_index(drop=True)
    test_df = valid.iloc[n_train:].copy().reset_index(drop=True)

    return train_df, test_df


def run_climatology_baseline(
    input_path: Path,
    output_path: Path,
    target_col: str = "cosmos_sm_median",
    date_col: str = "date",
    train_fraction: float = 0.8,
) -> Dict[str, object]:
    """
    Execute the climatology baseline workflow and export predictions.
    """
    df = pd.read_csv(input_path)

    train_df, test_df = split_chronological_holdout(
        df,
        date_col=date_col,
        target_col=target_col,
        train_fraction=train_fraction,
    )

    # 1. Fit: calculate training-period mean target only (strictly no test data)
    climatology_mean = float(train_df[target_col].mean())

    # 2. Predict: assign training mean to all test instances
    test_df["climatology_prediction"] = climatology_mean
    test_df["prediction_error"] = test_df["climatology_prediction"] - test_df[target_col]

    # 3. Calculate evaluation metrics on the test period
    metrics = compute_metrics(
        actual=test_df[target_col].to_numpy(),
        predicted=test_df["climatology_prediction"].to_numpy(),
    )

    # 4. Prepare export table
    export_cols = [
        date_col,
        "year",
        "doy",
        target_col,
        "climatology_prediction",
        "prediction_error",
    ]
    # Ensure export columns exist in test_df
    available_cols = [c for c in export_cols if c in test_df.columns]
    predictions_df = test_df[available_cols].rename(
        columns={target_col: "actual_cosmos_sm"}
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_df.to_csv(output_path, index=False)

    summary = {
        "target_col": target_col,
        "train_count": len(train_df),
        "test_count": len(test_df),
        "train_start_date": str(train_df[date_col].iloc[0]),
        "train_end_date": str(train_df[date_col].iloc[-1]),
        "test_start_date": str(test_df[date_col].iloc[0]),
        "test_end_date": str(test_df[date_col].iloc[-1]),
        "train_mean": climatology_mean,
        "test_mean": float(test_df[target_col].mean()),
        "rmse": metrics["rmse"],
        "bias": metrics["bias"],
        "ubrmse": metrics["ubrmse"],
        "output_path": str(output_path),
    }

    return summary


def main():
    project_root = Path(__file__).resolve().parent.parent
    input_path = project_root / "data" / "processed" / "p041" / "p041_2016_daily_gnss_cosmos_validation.csv"
    output_path = project_root / "data" / "processed" / "p041" / "p041_2016_climatology_predictions.csv"

    print("=" * 80)
    print("RUNNING CLIMATOLOGY BASELINE: P041 2016 COSMOS SOIL MOISTURE")
    print("=" * 80)
    print(f"Input dataset:  {input_path}")
    print(f"Output dataset: {output_path}")

    # Primary target: cosmos_sm_median
    summary_median = run_climatology_baseline(
        input_path=input_path,
        output_path=output_path,
        target_col="cosmos_sm_median",
    )

    print("\n[Primary Target: cosmos_sm_median (m^3/m^3)]")
    print(f"  Training sample count: {summary_median['train_count']} days ({summary_median['train_start_date']} to {summary_median['train_end_date']})")
    print(f"  Testing sample count:  {summary_median['test_count']} days ({summary_median['test_start_date']} to {summary_median['test_end_date']})")
    print(f"  Training-period mean:  {summary_median['train_mean']:.6f} m^3/m^3")
    print(f"  Testing-period mean:   {summary_median['test_mean']:.6f} m^3/m^3")
    print(f"  Test RMSE:             {summary_median['rmse']:.6f} m^3/m^3")
    print(f"  Test Bias (Pred - Obs):{summary_median['bias']:+.6f} m^3/m^3")
    print(f"  Test ubRMSE:           {summary_median['ubrmse']:.6f} m^3/m^3")
    print(f"  Saved predictions to:  {summary_median['output_path']}")

    # Secondary sensitivity check: cosmos_sm_mean
    summary_mean = run_climatology_baseline(
        input_path=input_path,
        output_path=project_root / "data" / "processed" / "p041" / "p041_2016_climatology_predictions_mean_target.csv",
        target_col="cosmos_sm_mean",
    )
    print("\n[Sensitivity Target: cosmos_sm_mean (m^3/m^3)]")
    print(f"  Training-period mean:  {summary_mean['train_mean']:.6f} m^3/m^3")
    print(f"  Testing-period mean:   {summary_mean['test_mean']:.6f} m^3/m^3")
    print(f"  Test RMSE:             {summary_mean['rmse']:.6f} m^3/m^3")
    print(f"  Test Bias:             {summary_mean['bias']:+.6f} m^3/m^3")
    print(f"  Test ubRMSE:           {summary_mean['ubrmse']:.6f} m^3/m^3")
    print("=" * 80)


if __name__ == "__main__":
    main()
