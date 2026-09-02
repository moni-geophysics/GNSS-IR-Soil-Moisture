import pandas as pd


def check_time_series_quality(data: pd.DataFrame) -> dict:
    """
    Perform basic quality checks on a time-series DataFrame.
    """

    report = {
        "number_of_rows": len(data),
        "number_of_columns": len(data.columns),
        "missing_values": data.isna().sum().to_dict(),
        "duplicate_rows": int(data["datetime"].duplicated().sum()),
    }

    return report