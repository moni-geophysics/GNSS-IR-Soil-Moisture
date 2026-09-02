"""
Functions for reading ISMN soil-moisture data.
"""

from pathlib import Path
import pandas as pd


def read_ismn_stm(file_path):
    """
    Read an ISMN .stm soil-moisture file.

    Parameters
    ----------
    file_path : str or Path
        Path to the ISMN .stm file.

    Returns
    -------
    header : dict
        Station and sensor information from the file header.

    data : pandas.DataFrame
        Time series of soil-moisture measurements and quality flags.
    """

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"ISMN file not found: {file_path}"
        )

    # Read all lines from the file
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