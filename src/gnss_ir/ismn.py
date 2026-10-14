"""
Functions for reading ISMN soil-moisture data.
"""

from pathlib import Path
import pandas as pd


import io
import zipfile
from typing import Union


def read_ismn_stm(file_or_stream):
    """
    Read an ISMN .stm soil-moisture file from a path or file-like stream.

    Parameters
    ----------
    file_or_stream : str, Path, or file-like object
        Path to the ISMN .stm file, or an open file-like stream.

    Returns
    -------
    header : dict
        Station and sensor information from the file header.

    data : pandas.DataFrame
        Time series of soil-moisture measurements and quality flags.
    """

    if hasattr(file_or_stream, "readlines"):
        lines = file_or_stream.readlines()
        if lines and isinstance(lines[0], bytes):
            lines = [l.decode("utf-8", errors="replace") for l in lines]
    else:
        file_path = Path(file_or_stream)
        if not file_path.exists():
            raise FileNotFoundError(
                f"ISMN file not found: {file_path}"
            )
        with open(file_path, "r", encoding="utf-8") as file:
            lines = file.readlines()

    # ---------------------------------------------------------
    # 1. Read the header
    # ---------------------------------------------------------

    header_line = lines[0].strip()

    header_parts = header_line.split()

    header = {
        "network": header_parts[0],
        "station": header_parts[2],
        "latitude": float(header_parts[3]),
        "longitude": float(header_parts[4]),
        "elevation_m": float(header_parts[5]),
        "depth_from_m": float(header_parts[6]),
        "depth_to_m": float(header_parts[7]),
        "sensor": " ".join(header_parts[8:]).strip("'"),
    }

    # ---------------------------------------------------------
    # 2. Read the measurement records
    # ---------------------------------------------------------

    records = []

    for line in lines[1:]:
        line = line.strip()

        if not line:
            continue

        parts = line.split()

        date = parts[0]
        time = parts[1]
        soil_moisture = parts[2]
        ismn_flag = parts[3]
        provider_flag = parts[4] if len(parts) > 4 else None

        records.append({
            "datetime": f"{date} {time}",
            "soil_moisture": soil_moisture,
            "ismn_flag": ismn_flag,
            "provider_flag": provider_flag,
        })

    data = pd.DataFrame(records)

    # Convert columns to appropriate data types
    data["datetime"] = pd.to_datetime(
        data["datetime"],
        format="%Y/%m/%d %H:%M"
    )

    data["soil_moisture"] = pd.to_numeric(
        data["soil_moisture"],
        errors="coerce"
    )

    return header, data