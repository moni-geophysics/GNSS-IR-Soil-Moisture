from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


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
# Select GPS PRN 1
# ---------------------------------------------------------

satellite_number = 1

sat = data[
    data["satellite"] == satellite_number
].copy()


# ---------------------------------------------------------
# Select candidate rising arc
# ---------------------------------------------------------

sat = sat[
    (sat["elevation_deg"] >= 5.0)
    & (sat["elevation_deg"] <= 29.5)
    & (sat["elevation_rate"] > 0)
    & (sat["seconds_of_day"] >= 4.6 * 3600)
    & (sat["seconds_of_day"] <= 6.0 * 3600)
].copy()


sat = sat.sort_values("seconds_of_day")


# Convert time to hours
sat["hour"] = sat["seconds_of_day"] / 3600


# Convert elevation to radians
sat["elevation_rad"] = np.deg2rad(
    sat["elevation_deg"]
)


# Calculate sine of elevation
sat["sin_elevation"] = np.sin(
    sat["elevation_rad"]
)


print("GNSS-IR CANDIDATE ARC")
print("=====================")

print("Satellite:", satellite_number)
print("Direction: rising")

print(
    "Time:",
    f"{sat['hour'].min():.3f}",
    "to",
    f"{sat['hour'].max():.3f}",
    "UTC"
)

print(
    "Elevation:",
    f"{sat['elevation_deg'].min():.2f}°",
    "to",
    f"{sat['elevation_deg'].max():.2f}°"
)

print("Observations:", len(sat))

print(
    "Azimuth:",
    f"{sat['azimuth_deg'].min():.2f}°",
    "to",
    f"{sat['azimuth_deg'].max():.2f}°"
)


# ---------------------------------------------------------
# Plot 1: SNR vs elevation
# ---------------------------------------------------------

plt.figure(figsize=(10, 6))

plt.plot(
    sat["elevation_deg"],
    sat["S1"],
    ".",
    markersize=3,
)

plt.xlabel("Elevation angle (degrees)")
plt.ylabel("S1 SNR (dB-Hz)")
plt.title(
    "GPS PRN 1 — Rising Arc — S1 SNR vs Elevation"
)

plt.grid(True)
plt.tight_layout()
plt.show()


# ---------------------------------------------------------
# Plot 2: SNR vs sin(elevation)
# ---------------------------------------------------------

plt.figure(figsize=(10, 6))

plt.plot(
    sat["sin_elevation"],
    sat["S1"],
    ".",
    markersize=3,
)

plt.xlabel("sin(Elevation)")
plt.ylabel("S1 SNR (dB-Hz)")
plt.title(
    "GPS PRN 1 — Rising Arc — S1 SNR vs sin(Elevation)"
)

plt.grid(True)
plt.tight_layout()
plt.show()