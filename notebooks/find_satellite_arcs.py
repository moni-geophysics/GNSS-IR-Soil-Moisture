from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# PROJECT PATHS
# ============================================================

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

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "p041"
    / "p041_doy001_arc_inventory.csv"
)


# ============================================================
# SETTINGS
# ============================================================

MIN_ELEVATION = 5.0
MAX_ELEVATION = 25.0

# If the gap between consecutive observations is larger
# than this value, start a new arc.
MAX_TIME_GAP = 300.0  # seconds = 5 minutes

MIN_OBSERVATIONS = 20

# Current GNSS-IR workflow uses GPS L1.
# GPS PRNs are 1–32.
GPS_PRN_MIN = 1
GPS_PRN_MAX = 32


# ============================================================
# READ SNR DATA
# ============================================================

print("=" * 70)
print("P041 DOY 001 - SATELLITE ARC INVENTORY")
print("=" * 70)

print(f"\nReading:")
print(SNR_FILE)

columns = [
    "satellite",
    "elevation",
    "azimuth",
    "seconds_of_day",
    "elevation_rate",
    "snr_s6",
    "snr_s1",
    "snr_s2",
    "snr_s5",
    "snr_s7",
    "snr_s8"
]

data = pd.read_csv(
    SNR_FILE,
    sep=r"\s+",
    header=None,
    names=columns
)

print(f"Total observations in file: {len(data)}")


# ============================================================
# GPS ONLY
# ============================================================

data = data[
    data["satellite"].between(GPS_PRN_MIN, GPS_PRN_MAX)
].copy()

print(
    f"GPS observations (PRNs {GPS_PRN_MIN}-{GPS_PRN_MAX}): "
    f"{len(data)}"
)


# ============================================================
# ELEVATION FILTER
# ============================================================

data = data[
    (data["elevation"] >= MIN_ELEVATION)
    & (data["elevation"] <= MAX_ELEVATION)
].copy()

print(
    f"Observations in elevation range "
    f"{MIN_ELEVATION}-{MAX_ELEVATION}°: {len(data)}"
)


# ============================================================
# ARC IDENTIFICATION
# ============================================================

arc_records = []

for satellite in sorted(data["satellite"].unique()):

    # --------------------------------------------------------
    # Select one satellite
    # --------------------------------------------------------

    satellite_data = data[
        data["satellite"] == satellite
    ].copy()

    # --------------------------------------------------------
    # Sort chronologically
    # --------------------------------------------------------

    satellite_data = satellite_data.sort_values(
        "seconds_of_day"
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # Determine rising / setting
    # --------------------------------------------------------

    elevation_difference = satellite_data["elevation"].diff()

    satellite_data["elevation_rate"] = (
        elevation_difference
        / satellite_data["seconds_of_day"].diff()
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

    # --------------------------------------------------------
    # Time difference between observations
    # --------------------------------------------------------

    satellite_data["time_difference"] = (
        satellite_data["seconds_of_day"].diff()
    )

    # --------------------------------------------------------
    # Detect direction changes
    # --------------------------------------------------------

    satellite_data["direction_change"] = (
        satellite_data["arc_type"]
        != satellite_data["arc_type"].shift(1)
    )

    # First observation must start an arc
    satellite_data.loc[0, "direction_change"] = True

    # --------------------------------------------------------
    # New arc conditions
    # --------------------------------------------------------

    new_arc = (
        (satellite_data["time_difference"] > MAX_TIME_GAP)
        | satellite_data["direction_change"]
    )

    # --------------------------------------------------------
    # Assign arc number
    # --------------------------------------------------------

    satellite_data["arc_number"] = (
        new_arc.cumsum() - 1
    )

    # --------------------------------------------------------
    # Build inventory
    # --------------------------------------------------------

    for arc_number, arc in satellite_data.groupby(
        "arc_number"
    ):

        if len(arc) < MIN_OBSERVATIONS:
            continue

        # ----------------------------------------------------
        # IMPORTANT:
        # Calculate time gaps INSIDE this arc only.
        #
        # This prevents the large gap between two different
        # arcs from being reported as an internal arc gap.
        # ----------------------------------------------------

        within_arc_time_difference = (
            arc["seconds_of_day"].diff()
        )

        valid_time_gaps = (
            within_arc_time_difference
            .dropna()
        )

        if len(valid_time_gaps) > 0:
            max_time_gap = valid_time_gaps.max()
        else:
            max_time_gap = 0.0

        # ----------------------------------------------------
        # Elevation consistency
        # ----------------------------------------------------

        expected_elevation_change = (
            arc["elevation_rate"].shift(1)
            * within_arc_time_difference
        )

        actual_elevation_change = (
            arc["elevation"].diff()
        )

        consistency_error = (
            actual_elevation_change
            - expected_elevation_change
        )

        valid_consistency = (
            consistency_error.dropna()
        )

        if len(valid_consistency) > 0:
            mean_consistency_error = (
                valid_consistency.abs().mean()
            )
        else:
            mean_consistency_error = np.nan

        # ----------------------------------------------------
        # Arc information
        # ----------------------------------------------------

        start_time = arc["seconds_of_day"].min()
        end_time = arc["seconds_of_day"].max()

        duration = end_time - start_time

        min_elevation = arc["elevation"].min()
        max_elevation = arc["elevation"].max()

        elevation_span = (
            max_elevation - min_elevation
        )

        observation_count = len(arc)

        arc_type = arc["arc_type"].iloc[0]

        # ----------------------------------------------------
        # Store record
        # ----------------------------------------------------

        arc_records.append({
            "satellite": int(satellite),
            "arc_number": int(arc_number),
            "arc_type": arc_type,

            "start_seconds": start_time,
            "end_seconds": end_time,

            "start_hour": start_time / 3600.0,
            "end_hour": end_time / 3600.0,

            "duration_sec": duration,
            "duration_min": duration / 60.0,

            "min_elevation_deg": min_elevation,
            "max_elevation_deg": max_elevation,
            "elevation_span_deg": elevation_span,

            "observations": observation_count,

            "max_time_gap_sec": max_time_gap,

            "mean_consistency_error_deg": (
                mean_consistency_error
            )
        })


# ============================================================
# CREATE ARC INVENTORY
# ============================================================

arc_inventory = pd.DataFrame(arc_records)

if len(arc_inventory) > 0:

    arc_inventory = arc_inventory.sort_values(
        ["satellite", "start_seconds"]
    ).reset_index(drop=True)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("ARC INVENTORY SUMMARY")
print("=" * 70)

print(f"\nNumber of candidate arcs: {len(arc_inventory)}")

if len(arc_inventory) > 0:

    print(
        f"Satellites with arcs: "
        f"{arc_inventory['satellite'].nunique()}"
    )

    print(
        f"Rising arcs: "
        f"{(arc_inventory['arc_type'] == 'rising').sum()}"
    )

    print(
        f"Setting arcs: "
        f"{(arc_inventory['arc_type'] == 'setting').sum()}"
    )

    print(
        f"\nMaximum within-arc time gap: "
        f"{arc_inventory['max_time_gap_sec'].max():.1f} s"
    )

    print(
        f"Maximum elevation span: "
        f"{arc_inventory['elevation_span_deg'].max():.2f}°"
    )

    print("\nFirst 20 arcs:")
    print(
        arc_inventory.head(20).to_string(index=False)
    )


# ============================================================
# SAVE
# ============================================================

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

arc_inventory.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n" + "=" * 70)
print("SAVED")
print("=" * 70)

print(OUTPUT_FILE)