import sys
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

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

if not SNR_FILE.exists():
    raise FileNotFoundError(f"SNR file not found: {SNR_FILE}")

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

print("SNR FILE")
print("--------")
print(SNR_FILE)

print("\nNumber of observations:", len(data))
print("Number of columns:", len(data.columns))

print("\nFirst 10 observations:")
print(data.head(10).to_string(index=False))

print("\nSatellite numbers:")
print(sorted(data["satellite"].unique()))

print("\nElevation range:")
print(
    data["elevation_deg"].min(),
    "to",
    data["elevation_deg"].max(),
    "degrees",
)

print("\nAzimuth range:")
print(
    data["azimuth_deg"].min(),
    "to",
    data["azimuth_deg"].max(),
    "degrees",
)

print("\nS1 observations:")
print((data["S1"] > 0).sum())

print("\nS2 observations:")
print((data["S2"] > 0).sum())

print("\nZero S1 observations:")
print((data["S1"] == 0).sum())

print("\nZero S2 observations:")
print((data["S2"] == 0).sum())