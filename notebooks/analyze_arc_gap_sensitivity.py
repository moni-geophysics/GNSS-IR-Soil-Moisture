from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SNR_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "gnss"
    / "2016"
    / "snr"
    / "p041"
    / "p0410010.16.snr66"
)

COLUMNS = [
    "satellite", "elevation", "azimuth", "seconds_of_day",
    "elevation_rate", "S6", "S1", "S2", "S5", "S7", "S8"
]

data = pd.read_csv(
    SNR_FILE,
    sep=r"\s+",
    header=None,
    names=COLUMNS
)

data = data[
    (data["elevation"] >= 5.0) &
    (data["elevation"] <= 25.0)
].copy()

results = []

for satellite, satellite_data in data.groupby("satellite"):

    satellite_data = satellite_data.sort_values(
        "seconds_of_day"
    ).copy()

    satellite_data["elevation_rate"] = (
        satellite_data["elevation"].diff()
    )

    satellite_data["arc_type"] = np.where(
        satellite_data["elevation_rate"] > 0,
        "rising",
        np.where(
            satellite_data["elevation_rate"] < 0,
            "setting",
            "unknown"
        )
    )

    satellite_data["time_difference"] = (
        satellite_data["seconds_of_day"].diff()
    )

    satellite_data["direction_change"] = (
        satellite_data["arc_type"]
        != satellite_data["arc_type"].shift(1)
    )

    satellite_data.iloc[0, satellite_data.columns.get_loc(
        "direction_change"
    )] = True

    for threshold in [30.0, 60.0, 120.0, 300.0]:

        new_arc = (
            (satellite_data["time_difference"] > threshold)
            | satellite_data["direction_change"]
        )

        satellite_data["arc_number_test"] = (
            new_arc.cumsum() - 1
        )

        for arc_number, arc in satellite_data.groupby(
            "arc_number_test"
        ):

            if len(arc) < 20:
                continue

            results.append({
                "threshold_s": threshold,
                "satellite": int(satellite),
                "arc_number": int(arc_number),
                "arc_type": arc["arc_type"].iloc[0],
                "start_seconds": arc["seconds_of_day"].iloc[0],
                "end_seconds": arc["seconds_of_day"].iloc[-1],
                "observations": len(arc),
                "max_internal_gap_s": arc["seconds_of_day"].diff().max()
            })

result = pd.DataFrame(results)

print("=" * 70)
print("EXACT ARC-SELECTION GAP SENSITIVITY")
print("=" * 70)

for threshold in [30.0, 60.0, 120.0, 300.0]:

    subset = result[result["threshold_s"] == threshold]

    print()
    print(f"THRESHOLD: {threshold:.0f} s")
    print(f"Total arcs: {len(subset)}")
    print(
        f"Rising: {(subset['arc_type'] == 'rising').sum()}"
    )
    print(
        f"Setting: {(subset['arc_type'] == 'setting').sum()}"
    )

    prn1 = subset[subset["satellite"] == 1]

    print("\nPRN 1:")
    print(
        prn1[
            [
                "satellite",
                "arc_number",
                "arc_type",
                "start_seconds",
                "end_seconds",
                "observations",
                "max_internal_gap_s"
            ]
        ].to_string(index=False)
    )

print()
print("=" * 70)
