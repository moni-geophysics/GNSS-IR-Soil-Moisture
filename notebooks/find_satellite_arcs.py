from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent

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


columns = [
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


data = pd.read_csv(
    SNR_FILE,
    sep=r"\s+",
    header=None,
    names=columns,
)


# ---------------------------------------------------------
# Select one GPS satellite
# ---------------------------------------------------------

satellite_number = 1

sat = data[
    data["satellite"] == satellite_number
].copy()

sat = sat.sort_values("seconds_of_day").reset_index(drop=True)


# Convert seconds to hours
sat["hour"] = sat["seconds_of_day"] / 3600


# ---------------------------------------------------------
# Keep useful GNSS-IR elevation range
# ---------------------------------------------------------

sat = sat[
    (sat["elevation_deg"] >= 5.0)
    & (sat["elevation_deg"] <= 29.5)
].copy()


# ---------------------------------------------------------
# Determine rising / setting
# ---------------------------------------------------------

sat["arc_type"] = "unknown"

sat.loc[
    sat["elevation_rate"] > 0,
    "arc_type"
] = "rising"

sat.loc[
    sat["elevation_rate"] < 0,
    "arc_type"
] = "setting"


# ---------------------------------------------------------
# Print basic information
# ---------------------------------------------------------

print("GPS PRN:", satellite_number)

print("\nNumber of observations in 5–29.5°:")
print(len(sat))


print("\nRising observations:")
print(
    (sat["arc_type"] == "rising").sum()
)


print("\nSetting observations:")
print(
    (sat["arc_type"] == "setting").sum()
)


# ---------------------------------------------------------
# Identify gaps between separate satellite passes
# ---------------------------------------------------------

sat["time_difference"] = (
    sat["seconds_of_day"].diff()
)

# A gap greater than 5 minutes means a new arc
sat["new_arc"] = (
    sat["time_difference"] > 300
)

sat["arc_number"] = sat["new_arc"].cumsum()


# ---------------------------------------------------------
# Summarize arcs
# ---------------------------------------------------------

print("\n\nCANDIDATE SATELLITE ARCS")
print("========================")

for arc_number, arc in sat.groupby("arc_number"):

    if len(arc) < 20:
        continue

    print(
        f"\nArc {arc_number}"
    )

    print(
        "Type:",
        arc["arc_type"].mode().iloc[0]
    )

    print(
        "Start time:",
        f"{arc['hour'].min():.2f} h"
    )

    print(
        "End time:",
        f"{arc['hour'].max():.2f} h"
    )

    print(
        "Minimum elevation:",
        f"{arc['elevation_deg'].min():.2f}°"
    )

    print(
        "Maximum elevation:",
        f"{arc['elevation_deg'].max():.2f}°"
    )

    print(
        "Number of observations:",
        len(arc)
    )