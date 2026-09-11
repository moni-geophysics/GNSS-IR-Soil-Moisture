from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.signal import lombscargle


# =========================================================
# PROJECT PATH
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from gnss_ir.processing import process_gnss_ir_arc_from_files


# =========================================================
# INPUT FILES
# =========================================================

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

ARC_INVENTORY_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "p041"
    / "p041_doy001_arc_inventory.csv"
)


# =========================================================
# SNR FILE COLUMNS
# =========================================================

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


# =========================================================
# READ RAW SNR DATA
# =========================================================

data = pd.read_csv(
    SNR_FILE,
    sep=r"\s+",
    header=None,
    names=columns,
)


# =========================================================
# READ ARC INVENTORY
# =========================================================

arc_inventory = pd.read_csv(
    ARC_INVENTORY_FILE
)


# =========================================================
# SELECT ONE ARC
# =========================================================
#
# PRN 1, rising arc 1
#
# =========================================================

satellite_number = 1
arc_type = "rising"
arc_number = 1


selected_arc = arc_inventory[
    (arc_inventory["satellite"] == satellite_number)
    & (arc_inventory["arc_type"] == arc_type)
    & (arc_inventory["arc_number"] == arc_number)
].copy()


if selected_arc.empty:
    raise ValueError(
        "Requested arc was not found in the arc inventory."
    )


arc = selected_arc.iloc[0]



# =========================================================

manual_sat = data[
    data["satellite"] == satellite_number
].copy()


manual_sat = manual_sat[
    (manual_sat["elevation_deg"] >= 5.0)
    & (manual_sat["elevation_deg"] <= 25.0)
    & (
        manual_sat["elevation_rate"] > 0
        if arc_type == "rising"
        else manual_sat["elevation_rate"] < 0
    )
    & (manual_sat["seconds_of_day"] >= arc["start_seconds"])
    & (manual_sat["seconds_of_day"] <= arc["end_seconds"])
].copy()

manual_sat = manual_sat.sort_values(
    "seconds_of_day"
).reset_index(drop=True)


manual_sat = manual_sat[
    np.isfinite(manual_sat["S1"])
    & (manual_sat["S1"] > 0)
].copy()


manual_sat = manual_sat.reset_index(drop=True)


inventory_sat = data[
    data["satellite"] == satellite_number
].copy()

inventory_start = float(arc["start_seconds"])
inventory_end = float(arc["end_seconds"])

inventory_sat = inventory_sat[
    (inventory_sat["seconds_of_day"] >= inventory_start)
    & (inventory_sat["seconds_of_day"] <= inventory_end)
].copy()

if arc_type == "rising":
    inventory_sat = inventory_sat[
        inventory_sat["elevation_rate"] > 0
    ].copy()

elif arc_type == "setting":
    inventory_sat = inventory_sat[
        inventory_sat["elevation_rate"] < 0
    ].copy()

inventory_sat = inventory_sat[
    np.isfinite(inventory_sat["S1"])
    & (inventory_sat["S1"] > 0)
].copy()

inventory_sat = inventory_sat.sort_values(
    "seconds_of_day"
).reset_index(drop=True)


# =========================================================
# PRINT COMPARISON
# =========================================================

print()
print("REPRODUCIBILITY DIAGNOSTIC")
print("==========================")

print(
    "Manual selection observations:",
    len(manual_sat)
)

print(
    "Inventory selection observations:",
    len(inventory_sat)
)


# ---------------------------------------------------------
# Compare observation counts
# ---------------------------------------------------------

if len(manual_sat) == len(inventory_sat):

    print(
        "Observation count: MATCH"
    )

else:

    print(
        "Observation count: DIFFERENT"
    )


# ---------------------------------------------------------
# Compare timestamps
# ---------------------------------------------------------

if len(manual_sat) == len(inventory_sat):

    same_times = np.array_equal(
        manual_sat["seconds_of_day"].to_numpy(),
        inventory_sat["seconds_of_day"].to_numpy()
    )

    print(
        "Observation timestamps identical:",
        same_times
    )

else:

    same_times = False

    print(
        "Observation timestamps identical: NOT CHECKED"
    )


# ---------------------------------------------------------
# Compare S1 values
# ---------------------------------------------------------

if len(manual_sat) == len(inventory_sat):

    same_s1 = np.allclose(
        manual_sat["S1"].to_numpy(),
        inventory_sat["S1"].to_numpy(),
        rtol=0.0,
        atol=0.0
    )

    print(
        "S1 values identical:",
        same_s1
    )

else:

    same_s1 = False

    print(
        "S1 values identical: NOT CHECKED"
    )


# ---------------------------------------------------------
# Show boundary information
# ---------------------------------------------------------

print()
print("MANUAL SELECTION")
print("----------------")

if not manual_sat.empty:

    print(
        "Start:",
        manual_sat["seconds_of_day"].iloc[0],
        "s"
    )

    print(
        "End:",
        manual_sat["seconds_of_day"].iloc[-1],
        "s"
    )


print()
print("INVENTORY SELECTION")
print("-------------------")

if not inventory_sat.empty:

    print(
        "Start:",
        inventory_sat["seconds_of_day"].iloc[0],
        "s"
    )

    print(
        "End:",
        inventory_sat["seconds_of_day"].iloc[-1],
        "s"
    )


# ---------------------------------------------------------
# Find first difference if present
# ---------------------------------------------------------

if not same_times and len(manual_sat) == len(inventory_sat):

    time_difference = (
        manual_sat["seconds_of_day"].to_numpy()
        - inventory_sat["seconds_of_day"].to_numpy()
    )

    different_indices = np.where(
        time_difference != 0
    )[0]


    print()
    print("FIRST TIMESTAMP DIFFERENCE")
    print("--------------------------")

    if len(different_indices) > 0:

        i = different_indices[0]

        print(
            "Index:",
            i
        )

        print(
            "Manual time:",
            manual_sat["seconds_of_day"].iloc[i]
        )

        print(
            "Inventory time:",
            inventory_sat["seconds_of_day"].iloc[i]
        )


if not same_s1 and len(manual_sat) == len(inventory_sat):

    s1_difference = (
        manual_sat["S1"].to_numpy()
        - inventory_sat["S1"].to_numpy()
    )

    different_s1_indices = np.where(
        s1_difference != 0
    )[0]


    print()
    print("FIRST S1 DIFFERENCE")
    print("-------------------")

    if len(different_s1_indices) > 0:

        i = different_s1_indices[0]

        print(
            "Index:",
            i
        )

        print(
            "Manual S1:",
            manual_sat["S1"].iloc[i]
        )

        print(
            "Inventory S1:",
            inventory_sat["S1"].iloc[i]
        )


# ---------------------------------------------------------
# Final diagnostic conclusion
# ---------------------------------------------------------

print()
print("DIAGNOSTIC CONCLUSION")
print("---------------------")

if (
    len(manual_sat) == len(inventory_sat)
    and same_times
    and same_s1
):

    print(
        "The manual and inventory selections are IDENTICAL."
    )

else:

    print(
        "The manual and inventory selections are DIFFERENT."
    )


# =========================================================
# EXTRACT EXACT ARC FROM RAW SNR DATA
# =========================================================

sat = data[
    data["satellite"] == satellite_number
].copy()


sat = sat[
    (sat["seconds_of_day"] >= arc["start_seconds"])
    & (sat["seconds_of_day"] <= arc["end_seconds"])
].copy()


sat = sat.sort_values(
    "seconds_of_day"
).reset_index(drop=True)


if sat.empty:
    raise ValueError(
        "No observations were found for the selected arc."
    )


# =========================================================
# CHECK RISING / SETTING DIRECTION
# =========================================================

if arc_type == "rising":

    sat = sat[
        sat["elevation_rate"] > 0
    ].copy()

elif arc_type == "setting":

    sat = sat[
        sat["elevation_rate"] < 0
    ].copy()


# =========================================================
# REMOVE INVALID S1 VALUES
# =========================================================

sat = sat[
    np.isfinite(sat["S1"])
    & (sat["S1"] > 0)
].copy()


sat = sat.reset_index(drop=True)


if sat.empty:
    raise ValueError(
        "No valid S1 observations remain."
    )


# =========================================================
# CONVERT ELEVATION TO sin(E)
# =========================================================

elevation_rad = np.deg2rad(
    sat["elevation_deg"]
)

sat["sin_elevation"] = np.sin(
    elevation_rad
)


# =========================================================
# SORT BY sin(E)
# =========================================================

sat = sat.sort_values(
    "sin_elevation"
).reset_index(drop=True)


x = sat["sin_elevation"].to_numpy()


# =========================================================
# RAW SNR
# =========================================================

snr_db = sat["S1"].to_numpy()


# =========================================================
# CONVERT SNR FROM dB-Hz TO LINEAR AMPLITUDE
# =========================================================

snr_linear = 10 ** (
    snr_db / 20.0
)


# =========================================================
# SECOND-ORDER POLYNOMIAL DETRENDING
# =========================================================

polynomial_degree = 2


coefficients = np.polyfit(
    x,
    snr_linear,
    polynomial_degree
)


trend_linear = np.polyval(
    coefficients,
    x
)


# =========================================================
# DETRENDED SNR
# =========================================================

detrended_linear = (
    snr_linear - trend_linear
)


# =========================================================
# REMOVE MEAN
# =========================================================

detrended_linear = (
    detrended_linear
    - np.mean(detrended_linear)
)


# =========================================================
# GPS L1 WAVELENGTH
# =========================================================

wavelength = 0.19029367  # metres


# =========================================================
# REFLECTOR-HEIGHT SEARCH RANGE
# =========================================================

minimum_height = 0.1
maximum_height = 5.0

number_of_heights = 5000


# =========================================================
# LOMB-SCARGLE
# =========================================================

reflector_heights = np.linspace(
    minimum_height,
    maximum_height,
    number_of_heights
)


frequencies = (
    2.0
    * reflector_heights
    / wavelength
)


angular_frequencies = (
    2.0
    * np.pi
    * frequencies
)


power = lombscargle(
    x,
    detrended_linear,
    angular_frequencies,
    normalize=True
)


# =========================================================
# LSP PEAK
# =========================================================

peak_index = np.argmax(
    power
)


lsp_height = reflector_heights[
    peak_index
]


lsp_frequency = frequencies[
    peak_index
]


lsp_power = power[
    peak_index
]


# =========================================================
# FIXED-HEIGHT LEAST-SQUARES FUNCTION
# =========================================================
#
# GNSS-IR model:
#
# dSNR =
#
# A cos(
#     4*pi*H/lambda * sin(E)
#     + phi
# )
#
# Define:
#
# theta = 4*pi*H/lambda * sin(E)
#
# Then:
#
# A cos(theta + phi)
#
# = C cos(theta) + S sin(theta)
#
# C = A cos(phi)
# S = -A sin(phi)
#
# For a fixed H, C and S are solved
# directly using linear least squares.
#
# =========================================================

def fit_fixed_height(
    height,
    x,
    observed
):
    """
    Fit amplitude and phase for one fixed
    reflector height.
    """

    frequency = (
        2.0
        * height
        / wavelength
    )


    argument = (
        2.0
        * np.pi
        * frequency
        * x
    )


    cosine_component = np.cos(
        argument
    )


    sine_component = np.sin(
        argument
    )


    design_matrix = np.column_stack(
        (
            cosine_component,
            sine_component,
        )
    )


    # -----------------------------------------------------
    # LINEAR LEAST-SQUARES SOLUTION
    # -----------------------------------------------------

    coefficients, _, _, _ = np.linalg.lstsq(
        design_matrix,
        observed,
        rcond=None
    )


    c = coefficients[0]
    s = coefficients[1]


    # -----------------------------------------------------
    # FITTED SIGNAL
    # -----------------------------------------------------

    fitted = (
        c * cosine_component
        + s * sine_component
    )


    # -----------------------------------------------------
    # RESIDUAL
    # -----------------------------------------------------

    residual = (
        observed - fitted
    )


    # -----------------------------------------------------
    # SSE
    # -----------------------------------------------------

    sse = np.sum(
        residual ** 2
    )


    # -----------------------------------------------------
    # RMSE
    # -----------------------------------------------------

    rmse = np.sqrt(
        np.mean(
            residual ** 2
        )
    )


    # -----------------------------------------------------
    # R-SQUARED
    # -----------------------------------------------------

    ss_tot = np.sum(
        (
            observed
            - np.mean(observed)
        ) ** 2
    )


    if ss_tot > 0:

        r_squared = (
            1.0
            - sse / ss_tot
        )

    else:

        r_squared = np.nan


    # -----------------------------------------------------
    # AMPLITUDE
    # -----------------------------------------------------

    amplitude = np.sqrt(
        c**2 + s**2
    )


    # -----------------------------------------------------
    # PHASE
    # -----------------------------------------------------

    phase = np.arctan2(
        -s,
        c
    )


    return {
        "height": height,
        "frequency": frequency,
        "c": c,
        "s": s,
        "amplitude": amplitude,
        "phase": phase,
        "fitted": fitted,
        "residual": residual,
        "sse": sse,
        "rmse": rmse,
        "r_squared": r_squared,
    }


# =========================================================
# LS OBJECTIVE FUNCTION
# =========================================================

def objective(height):

    result = fit_fixed_height(
        height,
        x,
        detrended_linear
    )

    return result["sse"]


# =========================================================
# GLOBAL HEIGHT SEARCH
# =========================================================
#
# The LS objective is multimodal.
#
# Therefore:
#
# 1. Search the entire height range.
# 2. Find the global grid minimum.
# 3. Refine locally around that minimum.
#
# =========================================================

number_of_search_heights = 5000


height_grid = np.linspace(
    minimum_height,
    maximum_height,
    number_of_search_heights
)


sse_values = np.array(
    [
        objective(height)
        for height in height_grid
    ]
)


# =========================================================
# FIND GLOBAL GRID MINIMUM
# =========================================================

minimum_index = np.argmin(
    sse_values
)


grid_best_height = height_grid[
    minimum_index
]


# =========================================================
# LOCAL REFINEMENT
# =========================================================

grid_spacing = (
    height_grid[1]
    - height_grid[0]
)


refinement_lower = max(
    minimum_height,
    grid_best_height
    - 2.0 * grid_spacing
)


refinement_upper = min(
    maximum_height,
    grid_best_height
    + 2.0 * grid_spacing
)


optimization = minimize_scalar(
    objective,
    bounds=(
        refinement_lower,
        refinement_upper,
    ),
    method="bounded",
    options={
        "xatol": 1e-10
    }
)


# =========================================================
# FINAL LS HEIGHT
# =========================================================

ls_height = optimization.x


# =========================================================
# FINAL LS FIT
# =========================================================

fit_result = fit_fixed_height(
    ls_height,
    x,
    detrended_linear
)


# =========================================================
# EXTRACT LS RESULTS
# =========================================================

ls_frequency = fit_result["frequency"]

ls_amplitude = fit_result["amplitude"]

ls_phase = fit_result["phase"]

ls_rmse = fit_result["rmse"]

ls_r_squared = fit_result["r_squared"]

fitted_signal = fit_result["fitted"]

residuals = fit_result["residual"]


# =========================================================
# HEIGHT DIFFERENCE
# =========================================================

height_difference = (
    ls_height
    - lsp_height
)


absolute_height_difference = abs(
    height_difference
)


# =========================================================
# SSE AT LSP HEIGHT
# =========================================================

lsp_fit_result = fit_fixed_height(
    lsp_height,
    x,
    detrended_linear
)


lsp_sse = lsp_fit_result["sse"]

ls_sse = fit_result["sse"]


# =========================================================
# PRINT RESULTS
# =========================================================

print()
print("GNSS-IR SINGLE ARC — COMPLETE ANALYSIS")
print("=======================================")

print()
print("ARC INFORMATION")
print("----------------")

print(
    "Satellite:",
    satellite_number
)

print(
    "Arc number:",
    arc_number
)

print(
    "Arc type:",
    arc_type
)

print(
    "Start:",
    f"{arc['start_seconds']:.1f}",
    "s"
)

print(
    "End:",
    f"{arc['end_seconds']:.1f}",
    "s"
)

print(
    "Duration:",
    f"{arc['duration_sec']:.1f}",
    "s"
)

print(
    "Observations:",
    len(sat)
)

print(
    "Elevation:",
    f"{sat['elevation_deg'].min():.4f}°",
    "to",
    f"{sat['elevation_deg'].max():.4f}°"
)


# =========================================================
# PREPROCESSING
# =========================================================

print()
print("PREPROCESSING")
print("-------------")

print(
    "SNR representation:",
    "linear amplitude"
)

print(
    "Polynomial degree:",
    polynomial_degree
)

print(
    "Detrended mean:",
    f"{np.mean(detrended_linear):.10f}"
)


# =========================================================
# LSP RESULTS
# =========================================================

print()
print("LOMB-SCARGLE")
print("------------")

print(
    "Wavelength:",
    f"{wavelength:.8f}",
    "m"
)

print(
    "LSP reflector height:",
    f"{lsp_height:.6f}",
    "m"
)

print(
    "LSP frequency:",
    f"{lsp_frequency:.6f}",
    "cycles per unit sin(E)"
)

print(
    "Maximum LSP power:",
    f"{lsp_power:.6f}"
)


# =========================================================
# PLOT — LOMB-SCARGLE POWER VS REFLECTOR HEIGHT
# =========================================================

plt.figure(
    figsize=(10, 6)
)

plt.plot(
    reflector_heights,
    power,
    linewidth=2
)

plt.axvline(
    lsp_height,
    linestyle="--",
    linewidth=1,
    label=f"LSP H = {lsp_height:.3f} m"
)

plt.xlabel(
    "Reflector Height (m)"
)

plt.ylabel(
    "Normalized Lomb-Scargle Power"
)

plt.title(
    "GPS PRN 1 — Lomb-Scargle Power vs Reflector Height"
)

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.show()


# =========================================================
# LEAST-SQUARES RESULTS
# =========================================================

print()
print("LEAST-SQUARES")
print("-------------")

print(
    "LS reflector height:",
    f"{ls_height:.9f}",
    "m"
)

print(
    "LS frequency:",
    f"{ls_frequency:.6f}",
    "cycles per unit sin(E)"
)

print(
    "Amplitude:",
    f"{ls_amplitude:.6f}"
)

print(
    "Phase:",
    f"{ls_phase:.6f}",
    "rad"
)


# =========================================================
# LSP vs LS
# =========================================================

print()
print("LSP vs LS")
print("---------")

print(
    "Height difference:",
    f"{height_difference:.9f}",
    "m"
)

print(
    "Absolute difference:",
    f"{absolute_height_difference:.9f}",
    "m"
)

print(
    "Absolute difference:",
    f"{absolute_height_difference * 1000:.3f}",
    "mm"
)


# =========================================================
# FIT QUALITY
# =========================================================

print()
print("FIT QUALITY")
print("-----------")

print(
    "RMSE:",
    f"{ls_rmse:.6f}"
)

print(
    "R²:",
    f"{ls_r_squared:.6f}"
)


# =========================================================
# SSE COMPARISON
# =========================================================

print()
print("SSE COMPARISON")
print("--------------")

print(
    "SSE at LSP height:",
    f"{lsp_sse:.6f}"
)

print(
    "SSE at LS height:",
    f"{ls_sse:.6f}"
)

print(
    "SSE reduction:",
    f"{lsp_sse - ls_sse:.6f}"
)


# =========================================================
# OPTIMIZATION
# =========================================================

print()
print("OPTIMIZATION")
print("------------")

print(
    "Best grid height:",
    f"{grid_best_height:.9f}",
    "m"
)

print(
    "Refined LS height:",
    f"{ls_height:.9f}",
    "m"
)

print(
    "Optimization successful:",
    optimization.success
)

print(
    "Optimizer message:",
    optimization.message
)


# =========================================================
# REFACTORED IMPLEMENTATION REGRESSION CHECK
# =========================================================

refactored_result = process_gnss_ir_arc_from_files(
    snr_file=SNR_FILE,
    arc_inventory_file=ARC_INVENTORY_FILE,
    satellite_number=satellite_number,
    arc_type=arc_type,
    arc_number=arc_number,
    station="p041",
    year=2016,
    doy=1,
)

if refactored_result.processing_status != "success":
    raise RuntimeError(
        refactored_result.failure_reason
    )

print()
print("REFACTORED IMPLEMENTATION CHECK")
print("-------------------------------")

print(
    "LSP height difference:",
    f"{refactored_result.lsp_reflector_height_m - lsp_height:.12e}",
    "m"
)

print(
    "LS height difference:",
    f"{refactored_result.ls_reflector_height_m - ls_height:.12e}",
    "m"
)

print(
    "Amplitude difference:",
    f"{refactored_result.ls_amplitude - ls_amplitude:.12e}"
)

print(
    "Phase difference:",
    f"{refactored_result.ls_phase - ls_phase:.12e}",
    "rad"
)

print(
    "RMSE difference:",
    f"{refactored_result.ls_rmse - ls_rmse:.12e}"
)

print(
    "R-squared difference:",
    f"{refactored_result.ls_r_squared - ls_r_squared:.12e}"
)

print(
    "SSE difference:",
    f"{refactored_result.ls_sse - ls_sse:.12e}"
)


# =========================================================
# PLOT 1 — OBSERVED VS LS FIT
# =========================================================

plt.figure(
    figsize=(10, 6)
)

sort_index = np.argsort(x)


plt.plot(
    x,
    detrended_linear,
    ".",
    markersize=4,
    label="Observed detrended SNR"
)


plt.plot(
    x[sort_index],
    fitted_signal[sort_index],
    linewidth=2,
    label=(
        f"LS fit, H = {ls_height:.3f} m"
    )
)


plt.xlabel(
    "sin(Elevation)"
)

plt.ylabel(
    "Detrended Linear SNR"
)

plt.title(
    "GPS PRN 1 — LSP-Informed Least-Squares Fit"
)

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.show()


# =========================================================
# PLOT 2 — RESIDUALS
# =========================================================

plt.figure(
    figsize=(10, 6)
)


plt.plot(
    x,
    residuals,
    ".",
    markersize=4
)


plt.axhline(
    0,
    linewidth=1
)


plt.xlabel(
    "sin(Elevation)"
)

plt.ylabel(
    "Residual"
)

plt.title(
    "GPS PRN 1 — Least-Squares Residuals"
)

plt.grid(True)

plt.tight_layout()

plt.show()


# =========================================================
# PLOT 3 — SSE VS REFLECTOR HEIGHT
# =========================================================

plt.figure(
    figsize=(10, 6)
)


plt.plot(
    height_grid,
    sse_values
)


plt.axvline(
    lsp_height,
    linestyle="--",
    linewidth=1,
    label=(
        f"LSP H = {lsp_height:.3f} m"
    )
)


plt.axvline(
    ls_height,
    linestyle="--",
    linewidth=1,
    label=(
        f"LS H = {ls_height:.3f} m"
    )
)


plt.xlabel(
    "Reflector Height (m)"
)

plt.ylabel(
    "Sum of Squared Errors"
)

plt.title(
    "Least-Squares Objective vs Reflector Height"
)

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.show()


# =========================================================
# FINAL STATUS
# =========================================================

print()
print("=======================================")
print("COMPLETE GNSS-IR ARC ANALYSIS SUCCESSFUL")
print("=======================================")
