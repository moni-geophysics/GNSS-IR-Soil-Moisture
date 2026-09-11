"""
Utilities for reading GNSS SNR observation files.
"""

from pathlib import Path

import pandas as pd


SNR_COLUMNS = [
    "satellite",
    "elevation_deg",
    "azimuth_deg",
    "seconds_of_day",
    "elevation_rate",
    "S6",
    "S1",
    "S2",
    "S5",
    "S7",
    "S8",
]


def read_snr_file(file_path):
    """
    Read a gnssrefl-style SNR file into a DataFrame.
    """

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"SNR file not found: {file_path}")

    return pd.read_csv(
        file_path,
        sep=r"\s+",
        header=None,
        names=SNR_COLUMNS,
    )
