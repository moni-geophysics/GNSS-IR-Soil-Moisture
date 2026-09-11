"""
Reusable GNSS-IR single-arc processing functions.

The defaults in this module preserve the current validated
P041 DOY 001 single-arc workflow.
"""

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.signal import lombscargle

from gnss_ir.snr import read_snr_file


GPS_L1_WAVELENGTH_M = 0.19029367


@dataclass
class DetrendingResult:
    polynomial_degree: int
    coefficients: np.ndarray
    trend_linear: np.ndarray
    detrended_linear: np.ndarray


@dataclass
class LombScargleResult:
    reflector_heights: np.ndarray
    frequencies: np.ndarray
    angular_frequencies: np.ndarray
    power: np.ndarray
    lsp_reflector_height_m: float
    lsp_frequency: float
    lsp_peak_power: float


@dataclass
class LeastSquaresFitResult:
    reflector_height_m: float
    frequency: float
    c: float
    s: float
    amplitude: float
    phase: float
    fitted: np.ndarray
    residual: np.ndarray
    sse: float
    rmse: float
    r_squared: float


@dataclass
class HeightSearchResult:
    fit: LeastSquaresFitResult
    height_grid: np.ndarray
    sse_values: np.ndarray
    grid_best_height_m: float
    optimization_success: bool
    optimization_message: str


@dataclass
class GnssIrArcResult:
    station: Optional[str]
    year: Optional[int]
    doy: Optional[int]
    prn: int
    arc_number: Optional[int]
    arc_type: str
    start_seconds: Optional[float]
    end_seconds: Optional[float]
    number_of_observations: int
    elevation_min_deg: Optional[float]
    elevation_max_deg: Optional[float]
    azimuth_min_deg: Optional[float]
    azimuth_max_deg: Optional[float]
    azimuth_mean_deg: Optional[float]
    snr_representation: str
    polynomial_degree: int
    wavelength_m: float
    lsp_reflector_height_m: Optional[float]
    lsp_frequency: Optional[float]
    lsp_peak_power: Optional[float]
    ls_reflector_height_m: Optional[float]
    ls_frequency: Optional[float]
    ls_amplitude: Optional[float]
    ls_phase: Optional[float]
    ls_rmse: Optional[float]
    ls_r_squared: Optional[float]
    ls_sse: Optional[float]
    lsp_sse: Optional[float]
    sse_reduction: Optional[float]
    lsp_vs_ls_height_difference_m: Optional[float]
    lsp_vs_ls_absolute_height_difference_m: Optional[float]
    processing_status: str
    failure_reason: Optional[str] = None
    x: Optional[np.ndarray] = field(default=None, repr=False)
    snr_linear: Optional[np.ndarray] = field(default=None, repr=False)
    detrended_linear: Optional[np.ndarray] = field(default=None, repr=False)
    fitted_signal: Optional[np.ndarray] = field(default=None, repr=False)
    residuals: Optional[np.ndarray] = field(default=None, repr=False)
    lsp_reflector_heights: Optional[np.ndarray] = field(default=None, repr=False)
    lsp_power: Optional[np.ndarray] = field(default=None, repr=False)
    height_grid: Optional[np.ndarray] = field(default=None, repr=False)
    sse_values: Optional[np.ndarray] = field(default=None, repr=False)
    arc_observations: Optional[pd.DataFrame] = field(default=None, repr=False)

    def to_dict(self, include_arrays=False):
        """
        Convert the result to a machine-readable dictionary.

        Array and DataFrame diagnostics are excluded by default so the
        dictionary is suitable for one-row tabular outputs.
        """

        result = asdict(self)

        if not include_arrays:
            for key in [
                "x",
                "snr_linear",
                "detrended_linear",
                "fitted_signal",
                "residuals",
                "lsp_reflector_heights",
                "lsp_power",
                "height_grid",
                "sse_values",
                "arc_observations",
            ]:
                result.pop(key, None)

        return result


def dbhz_to_linear_amplitude(snr_db):
    """
    Convert SNR from dB-Hz to the linear amplitude representation used
    by the current GNSS-IR workflow.
    """

    return 10 ** (np.asarray(snr_db, dtype=float) / 20.0)


def add_sine_elevation(data):
    """
    Add sin(elevation) to a copy of an SNR observation DataFrame.
    """

    data = data.copy()
    data["sin_elevation"] = np.sin(np.deg2rad(data["elevation_deg"]))
    return data


def detrend_polynomial(
    x,
    snr_linear,
    polynomial_degree=2,
    remove_mean=True,
):
    """
    Remove a polynomial trend from linear SNR observations.
    """

    x = np.asarray(x, dtype=float)
    snr_linear = np.asarray(snr_linear, dtype=float)

    coefficients = np.polyfit(
        x,
        snr_linear,
        polynomial_degree,
    )

    trend_linear = np.polyval(
        coefficients,
        x,
    )

    detrended_linear = snr_linear - trend_linear

    if remove_mean:
        detrended_linear = detrended_linear - np.mean(detrended_linear)

    return DetrendingResult(
        polynomial_degree=polynomial_degree,
        coefficients=coefficients,
        trend_linear=trend_linear,
        detrended_linear=detrended_linear,
    )


def compute_lomb_scargle_reflector_height(
    x,
    detrended_linear,
    wavelength_m=GPS_L1_WAVELENGTH_M,
    minimum_height_m=0.1,
    maximum_height_m=5.0,
    number_of_heights=5000,
):
    """
    Estimate reflector height from a Lomb-Scargle periodogram.
    """

    x = np.asarray(x, dtype=float)
    detrended_linear = np.asarray(detrended_linear, dtype=float)

    reflector_heights = np.linspace(
        minimum_height_m,
        maximum_height_m,
        number_of_heights,
    )

    frequencies = 2.0 * reflector_heights / wavelength_m
    angular_frequencies = 2.0 * np.pi * frequencies

    power = lombscargle(
        x,
        detrended_linear,
        angular_frequencies,
        normalize=True,
    )

    peak_index = np.argmax(power)

    return LombScargleResult(
        reflector_heights=reflector_heights,
        frequencies=frequencies,
        angular_frequencies=angular_frequencies,
        power=power,
        lsp_reflector_height_m=float(reflector_heights[peak_index]),
        lsp_frequency=float(frequencies[peak_index]),
        lsp_peak_power=float(power[peak_index]),
    )


def fit_fixed_height(
    height_m,
    x,
    observed,
    wavelength_m=GPS_L1_WAVELENGTH_M,
):
    """
    Fit amplitude and phase for one fixed reflector height.
    """

    x = np.asarray(x, dtype=float)
    observed = np.asarray(observed, dtype=float)

    frequency = 2.0 * height_m / wavelength_m
    argument = 2.0 * np.pi * frequency * x

    cosine_component = np.cos(argument)
    sine_component = np.sin(argument)

    design_matrix = np.column_stack(
        (
            cosine_component,
            sine_component,
        )
    )

    coefficients, _, _, _ = np.linalg.lstsq(
        design_matrix,
        observed,
        rcond=None,
    )

    c = coefficients[0]
    s = coefficients[1]

    fitted = c * cosine_component + s * sine_component
    residual = observed - fitted

    sse = np.sum(residual ** 2)
    rmse = np.sqrt(np.mean(residual ** 2))

    ss_tot = np.sum((observed - np.mean(observed)) ** 2)

    if ss_tot > 0:
        r_squared = 1.0 - sse / ss_tot
    else:
        r_squared = np.nan

    amplitude = np.sqrt(c**2 + s**2)
    phase = np.arctan2(-s, c)

    return LeastSquaresFitResult(
        reflector_height_m=float(height_m),
        frequency=float(frequency),
        c=float(c),
        s=float(s),
        amplitude=float(amplitude),
        phase=float(phase),
        fitted=fitted,
        residual=residual,
        sse=float(sse),
        rmse=float(rmse),
        r_squared=float(r_squared),
    )


def fit_reflector_height_least_squares(
    x,
    detrended_linear,
    wavelength_m=GPS_L1_WAVELENGTH_M,
    minimum_height_m=0.1,
    maximum_height_m=5.0,
    number_of_search_heights=5000,
    xatol=1e-10,
):
    """
    Fit reflector height using the current grid search plus local
    bounded scalar refinement strategy.
    """

    def objective(height):
        result = fit_fixed_height(
            height,
            x,
            detrended_linear,
            wavelength_m=wavelength_m,
        )
        return result.sse

    height_grid = np.linspace(
        minimum_height_m,
        maximum_height_m,
        number_of_search_heights,
    )

    sse_values = np.array(
        [
            objective(height)
            for height in height_grid
        ]
    )

    minimum_index = np.argmin(sse_values)
    grid_best_height_m = float(height_grid[minimum_index])

    grid_spacing = height_grid[1] - height_grid[0]

    refinement_lower = max(
        minimum_height_m,
        grid_best_height_m - 2.0 * grid_spacing,
    )

    refinement_upper = min(
        maximum_height_m,
        grid_best_height_m + 2.0 * grid_spacing,
    )

    optimization = minimize_scalar(
        objective,
        bounds=(
            refinement_lower,
            refinement_upper,
        ),
        method="bounded",
        options={
            "xatol": xatol,
        },
    )

    fit = fit_fixed_height(
        optimization.x,
        x,
        detrended_linear,
        wavelength_m=wavelength_m,
    )

    return HeightSearchResult(
        fit=fit,
        height_grid=height_grid,
        sse_values=sse_values,
        grid_best_height_m=grid_best_height_m,
        optimization_success=bool(optimization.success),
        optimization_message=str(optimization.message),
    )


def select_arc_from_inventory(
    snr_data,
    arc_inventory,
    satellite_number,
    arc_type,
    arc_number,
    snr_column="S1",
):
    """
    Select one arc from raw SNR data using an arc-inventory row.
    """

    selected_arc = arc_inventory[
        (arc_inventory["satellite"] == satellite_number)
        & (arc_inventory["arc_type"] == arc_type)
        & (arc_inventory["arc_number"] == arc_number)
    ].copy()

    if selected_arc.empty:
        raise ValueError("Requested arc was not found in the arc inventory.")

    arc = selected_arc.iloc[0]

    sat = snr_data[snr_data["satellite"] == satellite_number].copy()

    sat = sat[
        (sat["seconds_of_day"] >= arc["start_seconds"])
        & (sat["seconds_of_day"] <= arc["end_seconds"])
    ].copy()

    sat = sat.sort_values("seconds_of_day").reset_index(drop=True)

    if arc_type == "rising":
        sat = sat[sat["elevation_rate"] > 0].copy()
    elif arc_type == "setting":
        sat = sat[sat["elevation_rate"] < 0].copy()

    sat = sat[
        np.isfinite(sat[snr_column])
        & (sat[snr_column] > 0)
    ].copy()

    sat = sat.reset_index(drop=True)

    if sat.empty:
        raise ValueError("No valid SNR observations remain.")

    return arc, sat


def process_gnss_ir_arc(
    snr_data,
    arc_inventory,
    satellite_number,
    arc_type,
    arc_number,
    station=None,
    year=None,
    doy=None,
    snr_column="S1",
    polynomial_degree=2,
    wavelength_m=GPS_L1_WAVELENGTH_M,
    minimum_height_m=0.1,
    maximum_height_m=5.0,
    number_of_heights=5000,
    number_of_search_heights=5000,
):
    """
    Process one GNSS-IR arc and return a structured result.
    """

    try:
        arc, sat = select_arc_from_inventory(
            snr_data=snr_data,
            arc_inventory=arc_inventory,
            satellite_number=satellite_number,
            arc_type=arc_type,
            arc_number=arc_number,
            snr_column=snr_column,
        )

        sat = add_sine_elevation(sat)
        sat = sat.sort_values("sin_elevation").reset_index(drop=True)

        x = sat["sin_elevation"].to_numpy()
        snr_db = sat[snr_column].to_numpy()
        snr_linear = dbhz_to_linear_amplitude(snr_db)

        detrending = detrend_polynomial(
            x,
            snr_linear,
            polynomial_degree=polynomial_degree,
            remove_mean=True,
        )

        lsp = compute_lomb_scargle_reflector_height(
            x,
            detrending.detrended_linear,
            wavelength_m=wavelength_m,
            minimum_height_m=minimum_height_m,
            maximum_height_m=maximum_height_m,
            number_of_heights=number_of_heights,
        )

        height_search = fit_reflector_height_least_squares(
            x,
            detrending.detrended_linear,
            wavelength_m=wavelength_m,
            minimum_height_m=minimum_height_m,
            maximum_height_m=maximum_height_m,
            number_of_search_heights=number_of_search_heights,
        )

        lsp_fit = fit_fixed_height(
            lsp.lsp_reflector_height_m,
            x,
            detrending.detrended_linear,
            wavelength_m=wavelength_m,
        )

        ls_fit = height_search.fit
        height_difference = (
            ls_fit.reflector_height_m
            - lsp.lsp_reflector_height_m
        )

        return GnssIrArcResult(
            station=station,
            year=year,
            doy=doy,
            prn=int(satellite_number),
            arc_number=int(arc_number),
            arc_type=arc_type,
            start_seconds=float(arc["start_seconds"]),
            end_seconds=float(arc["end_seconds"]),
            number_of_observations=len(sat),
            elevation_min_deg=float(sat["elevation_deg"].min()),
            elevation_max_deg=float(sat["elevation_deg"].max()),
            azimuth_min_deg=float(sat["azimuth_deg"].min()),
            azimuth_max_deg=float(sat["azimuth_deg"].max()),
            azimuth_mean_deg=float(sat["azimuth_deg"].mean()),
            snr_representation="linear amplitude",
            polynomial_degree=polynomial_degree,
            wavelength_m=wavelength_m,
            lsp_reflector_height_m=lsp.lsp_reflector_height_m,
            lsp_frequency=lsp.lsp_frequency,
            lsp_peak_power=lsp.lsp_peak_power,
            ls_reflector_height_m=ls_fit.reflector_height_m,
            ls_frequency=ls_fit.frequency,
            ls_amplitude=ls_fit.amplitude,
            ls_phase=ls_fit.phase,
            ls_rmse=ls_fit.rmse,
            ls_r_squared=ls_fit.r_squared,
            ls_sse=ls_fit.sse,
            lsp_sse=lsp_fit.sse,
            sse_reduction=lsp_fit.sse - ls_fit.sse,
            lsp_vs_ls_height_difference_m=height_difference,
            lsp_vs_ls_absolute_height_difference_m=abs(height_difference),
            processing_status="success",
            failure_reason=None,
            x=x,
            snr_linear=snr_linear,
            detrended_linear=detrending.detrended_linear,
            fitted_signal=ls_fit.fitted,
            residuals=ls_fit.residual,
            lsp_reflector_heights=lsp.reflector_heights,
            lsp_power=lsp.power,
            height_grid=height_search.height_grid,
            sse_values=height_search.sse_values,
            arc_observations=sat,
        )

    except Exception as exc:
        return GnssIrArcResult(
            station=station,
            year=year,
            doy=doy,
            prn=int(satellite_number),
            arc_number=arc_number,
            arc_type=arc_type,
            start_seconds=None,
            end_seconds=None,
            number_of_observations=0,
            elevation_min_deg=None,
            elevation_max_deg=None,
            azimuth_min_deg=None,
            azimuth_max_deg=None,
            azimuth_mean_deg=None,
            snr_representation="linear amplitude",
            polynomial_degree=polynomial_degree,
            wavelength_m=wavelength_m,
            lsp_reflector_height_m=None,
            lsp_frequency=None,
            lsp_peak_power=None,
            ls_reflector_height_m=None,
            ls_frequency=None,
            ls_amplitude=None,
            ls_phase=None,
            ls_rmse=None,
            ls_r_squared=None,
            ls_sse=None,
            lsp_sse=None,
            sse_reduction=None,
            lsp_vs_ls_height_difference_m=None,
            lsp_vs_ls_absolute_height_difference_m=None,
            processing_status="failed",
            failure_reason=str(exc),
        )


def process_gnss_ir_arc_from_files(
    snr_file,
    arc_inventory_file,
    satellite_number,
    arc_type,
    arc_number,
    station=None,
    year=None,
    doy=None,
    **kwargs,
):
    """
    Process one GNSS-IR arc from SNR and arc-inventory files.
    """

    snr_data = read_snr_file(Path(snr_file))
    arc_inventory = pd.read_csv(Path(arc_inventory_file))

    return process_gnss_ir_arc(
        snr_data=snr_data,
        arc_inventory=arc_inventory,
        satellite_number=satellite_number,
        arc_type=arc_type,
        arc_number=arc_number,
        station=station,
        year=year,
        doy=doy,
        **kwargs,
    )
