from pathlib import Path

import matplotlib.pyplot as plt
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


# Select GPS PRN 1
satellite_number = 1

satellite_data = data[
    data["satellite"] == satellite_number
].copy()

satellite_data = satellite_data.sort_values(
    "seconds_of_day"
)


# Convert seconds of day to hours
satellite_data["hour"] = (
    satellite_data["seconds_of_day"] / 3600
)


# ---------------------------------------------------------
# Plot 1: satellite elevation through the day
# ---------------------------------------------------------

plt.figure(figsize=(10, 5))

plt.scatter(
    satellite_data["hour"],
    satellite_data["elevation_deg"],
    s=5,
)

plt.xlabel("Time (hours UTC)")
plt.ylabel("Elevation angle (degrees)")
plt.title(
    f"GPS PRN {satellite_number} — Elevation Through the Day"
)

plt.grid(True)
plt.tight_layout()
plt.show()


# ---------------------------------------------------------
# Plot 2: S1 SNR through the day
# ---------------------------------------------------------

plt.figure(figsize=(10, 5))

plt.scatter(
    satellite_data["hour"],
    satellite_data["S1"],
    s=5,
)

plt.xlabel("Time (hours UTC)")
plt.ylabel("S1 SNR (dB-Hz)")
plt.title(
    f"GPS PRN {satellite_number} — S1 SNR Through the Day"
)

plt.grid(True)
plt.tight_layout()
plt.show()