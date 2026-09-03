import sys
from pathlib import Path


# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Add src to Python's module search path
sys.path.insert(0, str(PROJECT_ROOT / "src"))


# Import project functions
from gnss_ir.ismn import read_ismn_stm
from data_quality import (
    count_quality_flags,
    count_individual_quality_flags,
)


# ISMN data folder
DATA_DIR = PROJECT_ROOT / "data" / "raw"


# Find the ISMN .stm file
stm_files = list(DATA_DIR.glob("*.stm"))

if not stm_files:
    raise FileNotFoundError("No .stm file found in data/raw")


ismn_file = stm_files[0]


# Read the ISMN data
print("Reading ISMN file:")
print(ismn_file)

header, data = read_ismn_stm(ismn_file)


# Count complete quality-flag combinations
flag_counts = count_quality_flags(data)


# Count individual flags, including flags inside combinations
individual_flag_counts = count_individual_quality_flags(data)


# Print complete flag combinations
print("\nISMN QUALITY FLAG COMBINATIONS")
print("------------------------------")

for flag, count in flag_counts.items():
    print(f"{flag}: {count}")


# Print individual flags
print("\nINDIVIDUAL ISMN QUALITY FLAGS")
print("-----------------------------")

for flag, count in individual_flag_counts.items():
    print(f"{flag}: {count}")


# Print total number of observations
print("\nTotal observations:", len(data))