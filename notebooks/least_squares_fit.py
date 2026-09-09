from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar


# =========================================================
# PROJECT PATH
# =========================================================

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


data = pd.read_csv(
    SNR_FILE,
    sep=r"\s+",
    header=None,
    names=columns,
)


# =========================================================
# SELECT THE SAME ARC USED FOR LOMB-SCARGLE
# =========================================================

satellite_number = 1

sat = data[
    data["satellite"] == satellite_number
].copy()


sat = sat[
    (sat["elevation_deg"] >= 5.0)
    & (sat["elevation_deg"] <= 25.0)
    & (sat["elevation_rate"] > 0)
    & (sat["seconds_of_day"] >= 4.6 * 3600)
    & (sat["seconds_of_day"] <= 6.0 * 3600)
].copy()


sat = sat.sort_values(
    "seconds_of_day"
).reset_index(drop=True)


# =========================================================
# REMOVE INVALID SNR VALUES
# =========================================================

sat = sat[
    np.isfinite(sat["S1"])
    & (sat["S1"] > 0)
].copy()


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
# LOMB-SCARGLE REFLECTOR HEIGHT
# =========================================================

lsp_height = 1.740  # metres


# =========================================================
# GNSS-IR MODEL
# =========================================================
#
# dSNR =
#
# A cos(4*pi*H/lambda * sin(E) + phi)
#
# Instead of fitting A and phi directly, write:
#
# A cos(theta + phi)
#
# = C cos(theta) + S sin(theta)
#
# where:
#
# C = A cos(phi)
# S = -A sin(phi)
#
# For a fixed H, C and S can be obtained
# directly using linear least squares.
#
# This makes the height search much more stable.
# =========================================================


def fit_fixed_height(height, x, observed):
    """
    Fit amplitude and phase for one fixed reflector height.

    Parameters
    ----------
    height : float
        Reflector height in metres.

    x : ndarray
        sin(elevation).

    observed : ndarray
        Detrended linear SNR.

    Returns
    -------
    dictionary
        Fitted parameters and diagnostics.
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

    # Linear least-squares solution
    coefficients, _, _, _ = np.linalg.lstsq(
        design_matrix,
        observed,
        rcond=None
    )

    c = coefficients[0]
    s = coefficients[1]

    fitted = (
        c * cosine_component
        + s * sine_component
    )

    residual = (
        observed - fitted
    )

    sse = np.sum(
        residual ** 2
    )

    rmse = np.sqrt(
        np.mean(
            residual ** 2
        )
    )

    ss_tot = np.sum(
        (
            observed
            - np.mean(observed)
        ) ** 2
    )

    r_squared = (
        1.0
        - sse / ss_tot
    )

    amplitude = np.sqrt(
        c**2 + s**2
    )

    # Because:
    #
    # C = A cos(phi)
    # S = -A sin(phi)
    #
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
# HEIGHT SEARCH
# =========================================================
#
# First search a broad range.
# =========================================================

minimum_height = 0.5
maximum_height = 5.0


# =========================================================
# OBJECTIVE FUNCTION
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
# The LS objective is multimodal, so we first perform
# a dense global search instead of using a single
# bounded scalar optimization over the entire range.
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
# LOCAL REFINEMENT AROUND GLOBAL MINIMUM
# =========================================================

grid_spacing = (
    height_grid[1]
    - height_grid[0]
)

refinement_lower = max(
    minimum_height,
    grid_best_height - 2.0 * grid_spacing
)

refinement_upper = min(
    maximum_height,
    grid_best_height + 2.0 * grid_spacing
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
# FINAL BEST-FIT SOLUTION
# =========================================================

ls_height = optimization.x

fit_result = fit_fixed_height(
    ls_height,
    x,
    detrended_linear
)


# =========================================================
# SSE AT THE LSP HEIGHT
# =========================================================

lsp_fit_result = fit_fixed_height(
    lsp_height,
    x,
    detrended_linear
)

lsp_sse = lsp_fit_result["sse"]
ls_sse = fit_result["sse"]

print()
print("SSE COMPARISON")
print("--------------")

print(
    "SSE at LSP height (1.740 m):",
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

print(
    "LSP-height RMSE:",
    f"{lsp_fit_result['rmse']:.6f}"
)

print(
    "LSP-height R²:",
    f"{lsp_fit_result['r_squared']:.6f}"
)


# =========================================================
# EXTRACT RESULTS
# =========================================================

ls_frequency = fit_result["frequency"]

ls_amplitude = fit_result["amplitude"]

ls_phase = fit_result["phase"]

ls_rmse = fit_result["rmse"]

ls_r_squared = fit_result["r_squared"]

fitted_signal = fit_result["fitted"]

residuals = fit_result["residual"]


# =========================================================
# HEIGHT COMPARISON
# =========================================================

height_difference = (
    ls_height - lsp_height
)

absolute_height_difference = abs(
    height_difference
)


# =========================================================
# PRINT RESULTS
# =========================================================

print()
print("GNSS-IR LEAST-SQUARES VALIDATION")
print("================================")

print(
    "Satellite:",
    satellite_number
)

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

print()

print("Lomb–Scargle")
print("------------")

print(
    "LSP reflector height:",
    f"{lsp_height:.3f} m"
)

lsp_frequency = (
    2.0
    * lsp_height
    / wavelength
)

print(
    "LSP frequency:",
    f"{lsp_frequency:.6f}",
    "cycles per unit sin(E)"
)

print()

print("Least-squares solution")
print("----------------------")

print(
    "LS reflector height:",
    f"{ls_height:.6f} m"
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

print()

print("Comparison")
print("----------")

print(
    "Height difference (LS - LSP):",
    f"{height_difference:.6f} m"
)

print(
    "Absolute height difference:",
    f"{absolute_height_difference:.6f} m"
)

print()

print("Fit quality")
print("-----------")

print(
    "RMSE:",
    f"{ls_rmse:.6f}"
)

print(
    "R²:",
    f"{ls_r_squared:.6f}"
)

print()

print(
    "Optimization successful:",
    optimization.success
)

print(
    "Optimizer message:",
    optimization.message
)


# =========================================================
# PLOT 1 — OBSERVED VS BEST LS FIT
# =========================================================

plt.figure(
    figsize=(10, 6)
)

plt.plot(
    x,
    detrended_linear,
    ".",
    markersize=4,
    label="Observed detrended SNR"
)

sort_index = np.argsort(x)

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

height_grid = np.linspace(
    minimum_height,
    maximum_height,
    1000
)

sse_values = np.array(
    [
        objective(h)
        for h in height_grid
    ]
)

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

print()
print("HEIGHT SEARCH DIAGNOSTIC")
print("------------------------")

print(
    "Best grid-search height:",
    f"{grid_best_height:.6f} m"
)

print(
    "Refined LS height:",
    f"{ls_height:.6f} m"
)

print(
    "SSE at LSP height:",
    f"{objective(lsp_height):.6f}"
)

print(
    "SSE at grid minimum:",
    f"{objective(grid_best_height):.6f}"
)

print(
    "SSE at refined LS height:",
    f"{objective(ls_height):.6f}"
)