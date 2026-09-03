from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
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
# Select GPS PRN 1
# ---------------------------------------------------------

satellite_number = 1

sat = data[
    data["satellite"] == satellite_number
].copy()


# ---------------------------------------------------------
# Select the same rising arc
# ---------------------------------------------------------

sat = sat[
    (sat["elevation_deg"] >= 5.0)
    & (sat["elevation_deg"] <= 29.5)
    & (sat["elevation_rate"] > 0)
    & (sat["seconds_of_day"] >= 4.6 * 3600)
    & (sat["seconds_of_day"] <= 6.0 * 3600)
].copy()


sat = sat.sort_values("seconds_of_day").reset_index(drop=True)


# ---------------------------------------------------------
# Convert elevation to sin(elevation)
# ---------------------------------------------------------

elevation_rad = np.deg2rad(
    sat["elevation_deg"]
)

sat["sin_elevation"] = np.sin(
    elevation_rad
)


# ---------------------------------------------------------
# Sort by sin(elevation)
# ---------------------------------------------------------

sat = sat.sort_values(
    "sin_elevation"
).reset_index(drop=True)


x = sat["sin_elevation"].to_numpy()
y = sat["S1"].to_numpy()


# ---------------------------------------------------------
# Fit a polynomial background trend
# ---------------------------------------------------------

polynomial_degree = 2

coefficients = np.polyfit(
    x,
    y,
    polynomial_degree
)

trend = np.polyval(
    coefficients,
    x
)


# ---------------------------------------------------------
# Calculate detrended SNR
# ---------------------------------------------------------

sat["trend"] = trend

sat["detrended_S1"] = (
    sat["S1"] - sat["trend"]
)


# ---------------------------------------------------------
# Print information
# ---------------------------------------------------------

print("GNSS-IR DETRENDING")
print("==================")

print("Satellite:", satellite_number)

print(
    "Number of observations:",
    len(sat)
)

print(
    "Elevation range:",
    f"{sat['elevation_deg'].min():.2f}°",
    "to",
    f"{sat['elevation_deg'].max():.2f}°"
)

print(
    "Detrended S1 range:",
    f"{sat['detrended_S1'].min():.3f}",
    "to",
    f"{sat['detrended_S1'].max():.3f}",
    "dB-Hz"
)


# ---------------------------------------------------------
# Plot raw SNR + fitted trend
# ---------------------------------------------------------

plt.figure(figsize=(10, 6))

plt.plot(
    x,
    y,
    ".",
    markersize=3,
    label="Observed S1"
)

plt.plot(
    x,
    trend,
    linewidth=2,
    label="Polynomial trend"
)

plt.xlabel("sin(Elevation)")
plt.ylabel("S1 SNR (dB-Hz)")
plt.title(
    "GPS PRN 1 — SNR and Background Trend"
)

plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# ---------------------------------------------------------
# Plot detrended SNR
# ---------------------------------------------------------

plt.figure(figsize=(10, 6))

plt.plot(
    x,
    sat["detrended_S1"],
    ".",
    markersize=3
)

plt.axhline(
    0,
    linewidth=1
)

plt.xlabel("sin(Elevation)")
plt.ylabel("Detrended S1 SNR (dB-Hz)")
plt.title(
    "GPS PRN 1 — Detrended S1 SNR"
)

plt.grid(True)
plt.tight_layout()
plt.show()