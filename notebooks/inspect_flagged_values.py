import sys
from pathlib import Path

import pandas as pd


# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Add src to Python's module search path
sys.path.insert(0, str(PROJECT_ROOT / "src"))


from gnss_ir.ismn import read_ismn_stm


# ISMN data folder
DATA_DIR = PROJECT_ROOT / "data" / "raw"


# Find the ISMN .stm file
stm_files = list(DATA_DIR.glob("*.stm"))

if not stm_files:
    raise FileNotFoundError("No .stm file found in data/raw")


ismn_file = stm_files[0]


# Read the ISMN data
header, data = read_ismn_stm(ismn_file)


# Display basic statistics for each individual flag
flags_to_inspect = ["D03", "D05", "C03", "C02"]


print("\nFLAGGED OBSERVATION SUMMARY")
print("============================")


for flag in flags_to_inspect:

    # Find observations containing this flag
    mask = data["ismn_flag"].fillna("").apply(
        lambda x: flag in str(x).split(",")
    )

    flagged = data[mask]

    print(f"\n{flag}")
    print("-" * 30)
    print("Number of observations:", len(flagged))

    if len(flagged) > 0:
        print(
            "Minimum soil moisture:",
            flagged["soil_moisture"].min()
        )

        print(
            "Maximum soil moisture:",
            flagged["soil_moisture"].max()
        )

        print(
            "Mean soil moisture:",
            flagged["soil_moisture"].mean()
        )

        print(
            "Median soil moisture:",
            flagged["soil_moisture"].median()
        )


# Show examples of flagged observations
print("\n\nEXAMPLE FLAGGED OBSERVATIONS")
print("============================")

print(
    data[
        data["ismn_flag"].fillna("").ne("G")
    ][
        ["datetime", "soil_moisture", "ismn_flag"]
    ].head(20).to_string(index=False)
)