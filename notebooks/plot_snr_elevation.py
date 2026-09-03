import sys
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


# ---------------------------------------------------------
# Select one GPS satellite
# ---------------------------------------------------------

satellite_number = 1

satellite_data = data[
    data["satellite"] == satellite_number
].copy()

satellite_data = satellite_data.sort_values(
    "elevation_deg"
)


# ---------------------------------------------------------
# Plot S1 SNR against elevation
# ---------------------------------------------------------

plt.figure(figsize=(10, 6))

plt.scatter(
    satellite_data["elevation_deg"],
    satellite_data["S1"],
    s=5,
)

plt.xlabel("Elevation angle (degrees)")
plt.ylabel("S1 SNR (dB-Hz)")
plt.title(
    f"GNSS SNR vs Elevation — GPS PRN {satellite_number}"
)

plt.grid(True)
plt.tight_layout()
plt.show()