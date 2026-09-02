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


def check_time_continuity(
    data: pd.DataFrame,
    time_column: str = "datetime",
    expected_frequency: str = "1h"
) -> dict:
    """
    Check whether a time series has missing time intervals.

    Parameters
    ----------
    data : pd.DataFrame
        Time-series data.
    time_column : str
        Name of the datetime column.
    expected_frequency : str
        Expected sampling interval, e.g. "1h", "30min".

    Returns
    -------
    dict
        Information about time-series continuity.
    """

    times = pd.to_datetime(data[time_column]).sort_values()

    expected_delta = pd.Timedelta(expected_frequency)

    time_differences = times.diff().dropna()

    gaps = time_differences[time_differences > expected_delta]

    return {
        "number_of_gaps": int(len(gaps)),
        "gap_locations": list(times.loc[gaps.index]),
    }


def check_datetime_validity(
    data: pd.DataFrame,
    time_column: str = "datetime"
) -> dict:
    """
    Check whether all values in the datetime column are valid timestamps.
    """

    converted_times = pd.to_datetime(
        data[time_column],
        errors="coerce"
    )

    invalid_mask = converted_times.isna()

    return {
        "number_of_invalid_datetimes": int(invalid_mask.sum()),
        "invalid_rows": list(data.index[invalid_mask]),
    }


def check_duplicate_timestamps(
    data: pd.DataFrame,
    time_column: str = "datetime"
) -> dict:
    """
    Check whether the time-series contains duplicate timestamps.
    """

    timestamps = pd.to_datetime(
        data[time_column],
        errors="coerce"
    )

    duplicate_mask = timestamps.duplicated(keep=False)

    return {
        "number_of_duplicate_timestamps": int(
            timestamps.duplicated().sum()
        ),
        "duplicate_rows": list(data.index[duplicate_mask]),
    }


def check_soil_moisture_values(
    data: pd.DataFrame,
    value_column: str = "value"
) -> dict:
    """
    Check soil-moisture observations for missing and negative values.
    """

    values = pd.to_numeric(data[value_column], errors="coerce")

    invalid_mask = values.isna() | (values < 0)

    return {
        "number_of_invalid_values": int(invalid_mask.sum()),
        "invalid_rows": list(data.index[invalid_mask]),
    }


def count_quality_flags(
    data: pd.DataFrame,
    flag_column: str = "ismn_flag"
) -> dict:
    """
    Count the occurrences of each quality flag in a time series.

    Parameters
    ----------
    data : pandas.DataFrame
        Time-series data containing quality flags.

    flag_column : str
        Name of the column containing quality flags.

    Returns
    -------
    dict
        Dictionary containing the count of each quality flag.
    """

    flag_counts = data[flag_column].value_counts(dropna=False)

    return {
        str(flag): int(count)
        for flag, count in flag_counts.items()
    }


def count_individual_quality_flags(
    data: pd.DataFrame,
    flag_column: str = "ismn_flag"
) -> dict:
    """
    Count individual ISMN quality flags, including flags
    that occur in combinations such as 'D05,D03'.
    """

    flag_counts = {}

    for value in data[flag_column].dropna():
        flags = str(value).split(",")

        for flag in flags:
            flag = flag.strip()
            flag_counts[flag] = flag_counts.get(flag, 0) + 1

    return flag_counts