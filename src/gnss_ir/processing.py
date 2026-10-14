"""
Reusable GNSS-IR single-arc processing functions.

The defaults in this module preserve the current validated
P041 DOY 001 single-arc workflow.
"""

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd
from numpy.polynomial import Polynomial
from scipy.optimize import minimize_scalar
from scipy.signal import lombscargle

from gnss_ir.refraction import (
    bennett_refraction_correction,
    get_gpt2_refraction_parameters,
)
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
    elevation_raw_min_deg: Optional[float] = None
    elevation_raw_max_deg: Optional[float] = None
    azimuth_min_deg: Optional[float] = None
    azimuth_max_deg: Optional[float] = None
    azimuth_mean_deg: Optional[float] = None
    refraction_applied: bool = False
    pressure_hpa: Optional[float] = None
    temperature_c: Optional[float] = None
    pele_min_deg: Optional[float] = None
    pele_max_deg: Optional[float] = None
    e1_deg: Optional[float] = None
    e2_deg: Optional[float] = None
    snr_representation: str = "linear amplitude"
    polynomial_degree: int = 4
    wavelength_m: float = GPS_L1_WAVELENGTH_M
    lsp_reflector_height_m: Optional[float] = None
    lsp_frequency: Optional[float] = None
    lsp_peak_power: Optional[float] = None
    ls_reflector_height_m: Optional[float] = None
    ls_frequency: Optional[float] = None
    ls_amplitude: Optional[float] = None
    normalized_amplitude_empirical: Optional[float] = None
    ls_phase: Optional[float] = None
    ls_rmse: Optional[float] = None
    ls_r_squared: Optional[float] = None
    ls_sse: Optional[float] = None
    lsp_sse: Optional[float] = None
    sse_reduction: Optional[float] = None
    lsp_vs_ls_height_difference_m: Optional[float] = None
    lsp_vs_ls_absolute_height_difference_m: Optional[float] = None
    processing_status: str = "success"
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


def add_sine_elevation(data, elevation_column="elevation_deg"):
    """
    Add sin(elevation) to a copy of an SNR observation DataFrame.
    """

    data = data.copy()
    data["sin_elevation"] = np.sin(np.deg2rad(data[elevation_column]))
    return data


def detrend_polynomial(
    x,
    snr_linear,
    polynomial_degree=4,
    x_fit=None,
    snr_linear_fit=None,
    remove_mean=True,
):
    """
    Remove a polynomial trend from linear SNR observations.

    Parameters
    ----------
    x : array_like
        Independent variable coordinates for evaluating and subtracting the trend.
    snr_linear : array_like
        Linear SNR values corresponding to `x`.
    polynomial_degree : int, default=4
        Degree of the detrending polynomial.
    x_fit : array_like, optional
        Independent variable coordinates used to fit the polynomial (e.g. over pele).
        If None, `x` is used for fitting.
    snr_linear_fit : array_like, optional
        Linear SNR values corresponding to `x_fit`.
        If None, `snr_linear` is used for fitting.
    remove_mean : bool, default=True
        Whether to subtract the mean of the detrended residual.
    """

    x = np.asarray(x, dtype=float)
    snr_linear = np.asarray(snr_linear, dtype=float)

    if x_fit is not None and snr_linear_fit is not None:
        fit_x = np.asarray(x_fit, dtype=float)
        fit_y = np.asarray(snr_linear_fit, dtype=float)
    else:
        fit_x = x
        fit_y = snr_linear

    # Use numpy Polynomial.fit for well-conditioned scaling
    poly = Polynomial.fit(fit_x, fit_y, deg=polynomial_degree)
    trend_linear = poly(x)
    detrended_linear = snr_linear - trend_linear

    if remove_mean:
        detrended_linear = detrended_linear - np.mean(detrended_linear)

    return DetrendingResult(
        polynomial_degree=polynomial_degree,
        coefficients=poly.convert().coef,
        trend_linear=trend_linear,
        detrended_linear=detrended_linear,
    )


def compute_normalized_amplitude_empirical(
    ls_amplitude: Optional[float],
    polynomial_background: np.ndarray,
) -> Optional[float]:
    """
    Calculate the empirical candidate feature alpha_arc = A_LS / mean(P(theta)).

    Parameters
    ----------
    ls_amplitude : float, optional
        Fitted sinusoidal oscillation amplitude (A_LS) from least-squares fit.
    polynomial_background : np.ndarray
        Polynomial background trend P(theta_i) evaluated at the identical samples
        belonging to the shared analysis window.

    Returns
    -------
    float or None
        alpha_arc = A_LS / mean(P(theta)) if valid, finite, and strictly positive denominator;
        None otherwise.

    Notes
    -----
    This feature is explicitly labeled as an empirical normalized-amplitude candidate feature.
    It is NOT a physical reflection coefficient or dielectric retrieval.
    """
    if ls_amplitude is None or not np.isfinite(ls_amplitude):
        return None

    if polynomial_background is None or len(polynomial_background) == 0:
        return None

    bg = np.asarray(polynomial_background, dtype=float)
    if not np.all(np.isfinite(bg)):
        return None

    mean_p = float(np.mean(bg))
    # Denominator must be strictly positive and finite
    if not np.isfinite(mean_p) or mean_p <= 0.0:
        return None

    return float(ls_amplitude / mean_p)



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
    buffer_seconds=1800.0,
):
    """
    Select one arc from raw SNR data using an arc-inventory row.
    Extracts the contiguous satellite track containing the arc so that
    extended fitting windows (such as pele=[5, 30]) have complete observation support.
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
    sat = sat.sort_values("seconds_of_day").reset_index(drop=True)

    sat = sat[
        np.isfinite(sat[snr_column])
        & (sat[snr_column] > 0)
    ].copy()

    # Extract the contiguous pass overlapping the inventory arc
    time_mask = (sat["seconds_of_day"] >= arc["start_seconds"] - buffer_seconds) & (
        sat["seconds_of_day"] <= arc["end_seconds"] + buffer_seconds
    )
    sat = sat[time_mask].copy().reset_index(drop=True)

    if not sat.empty:
        gaps = sat["seconds_of_day"].diff() > 300.0
        clusters = gaps.cumsum()
        mid_time = (arc["start_seconds"] + arc["end_seconds"]) / 2.0
        closest_idx = (sat["seconds_of_day"] - mid_time).abs().argmin()
        target_cluster = clusters.iloc[closest_idx]
        sat = sat[clusters == target_cluster].copy().reset_index(drop=True)

    # Direction filtering (support files where elevation_rate column is zero or unpopulated)
    if not sat.empty:
        if "elevation_rate" not in sat.columns or sat["elevation_rate"].abs().max() == 0:
            dt = sat["seconds_of_day"].diff()
            de = sat["elevation_deg"].diff()
            edot = (de / dt).bfill()
        else:
            edot = sat["elevation_rate"]

        if arc_type == "rising":
            sat = sat[edot >= 0].copy()
        elif arc_type == "setting":
            sat = sat[edot <= 0].copy()

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
    polynomial_degree=4,
    wavelength_m=GPS_L1_WAVELENGTH_M,
    minimum_height_m=0.5,
    maximum_height_m=8.0,
    number_of_heights=5000,
    number_of_search_heights=5000,
    refraction=True,
    pressure_hpa=None,
    temperature_c=None,
    station_config=None,
    pele=(5.0, 30.0),
    e1=5.0,
    e2=25.0,
):
    """
    Process one GNSS-IR arc with optional atmospheric refraction, separate
    fitting (pele) and analysis (e1, e2) windows, and return a structured result.
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

        # Preserve raw elevation
        sat["elevation_raw_deg"] = sat["elevation_deg"].copy()
        refraction_applied = False

        # Apply atmospheric refraction if enabled
        if refraction:
            p_val = pressure_hpa
            t_val = temperature_c

            if p_val is None or t_val is None:
                # Attempt to retrieve via station_config or standard file path
                cfg = station_config
                if cfg is None and station is not None:
                    candidates = [
                        Path(__file__).resolve().parent.parent.parent
                        / "data" / "raw" / "gnss" / "input" / station.lower() / f"{station.lower()}.json",
                        Path(__file__).resolve().parent.parent.parent
                        / "data" / "raw" / "gnss" / "input" / f"{station.lower()}.json",
                    ]
                    for candidate in candidates:
                        if candidate.exists():
                            cfg = candidate
                            break

                if cfg is not None and year is not None and doy is not None:
                    p_val, t_val = get_gpt2_refraction_parameters(cfg, year, doy)

            if p_val is not None and t_val is not None:
                # Bennett correction is physically valid for positive elevations (E > 0°).
                # Buffers may include below-horizon (E <= 0°) points which must be masked
                # rather than failing the valid arc.
                raw_el = sat["elevation_raw_deg"].to_numpy()
                positive_mask = np.isfinite(raw_el) & (raw_el > 0.0)
                refr_el = np.full_like(raw_el, np.nan, dtype=float)
                if np.any(positive_mask):
                    refr_el[positive_mask] = bennett_refraction_correction(
                        raw_el[positive_mask],
                        pressure_hpa=p_val,
                        temperature_c=t_val,
                    )
                sat["elevation_refr_deg"] = refr_el
                sat["elevation_deg"] = sat["elevation_refr_deg"]
                refraction_applied = True
                pressure_hpa = p_val
                temperature_c = t_val
            else:
                raise ValueError(
                    f"Refraction correction was requested (refraction=True), but atmospheric parameters "
                    f"(pressure, temperature) could not be retrieved for station='{station}', year={year}, doy={doy}."
                )
        else:
            sat["elevation_refr_deg"] = sat["elevation_raw_deg"]

        # Drop any non-finite elevation observations (e.g. masked sub-horizon buffer points)
        # before fitting and analysis window filtering
        sat = sat[np.isfinite(sat["elevation_deg"])].copy()
        if sat.empty:
            raise ValueError("No valid positive elevation observations remain in arc.")

        # 1. Filter fitting window over pele (e.g. 5° to 30°)
        sat_pele = sat[
            (sat["elevation_deg"] >= pele[0]) & (sat["elevation_deg"] <= pele[1])
        ].copy()
        if sat_pele.empty:
            raise ValueError(
                f"No observations remain in fitting window pele=[{pele[0]}, {pele[1]}]."
            )

        # 2. Filter analysis window over [e1, e2] (e.g. 5° to 25°)
        sat_analysis = sat_pele[
            (sat_pele["elevation_deg"] >= e1) & (sat_pele["elevation_deg"] <= e2)
        ].copy()
        if sat_analysis.empty:
            raise ValueError(
                f"No observations remain in spectral analysis window [{e1}, {e2}]."
            )

        # 3. Detrending (fit on pele, evaluate on analysis window)
        fit_ele = sat_pele["elevation_deg"].to_numpy()
        fit_snr = dbhz_to_linear_amplitude(sat_pele[snr_column].to_numpy())
        eval_ele = sat_analysis["elevation_deg"].to_numpy()
        eval_snr = dbhz_to_linear_amplitude(sat_analysis[snr_column].to_numpy())

        detrending = detrend_polynomial(
            x=eval_ele,
            snr_linear=eval_snr,
            polynomial_degree=polynomial_degree,
            x_fit=fit_ele,
            snr_linear_fit=fit_snr,
            remove_mean=True,
        )

        # 4. Geometry and spectral estimation
        sat_analysis = add_sine_elevation(sat_analysis, elevation_column="elevation_deg")
        sat_analysis = sat_analysis.sort_values("sin_elevation").reset_index(drop=True)

        x = sat_analysis["sin_elevation"].to_numpy()
        snr_linear = eval_snr
        detrended_linear = detrending.detrended_linear

        lsp = compute_lomb_scargle_reflector_height(
            x,
            detrended_linear,
            wavelength_m=wavelength_m,
            minimum_height_m=minimum_height_m,
            maximum_height_m=maximum_height_m,
            number_of_heights=number_of_heights,
        )

        height_search = fit_reflector_height_least_squares(
            x,
            detrended_linear,
            wavelength_m=wavelength_m,
            minimum_height_m=minimum_height_m,
            maximum_height_m=maximum_height_m,
            number_of_search_heights=number_of_search_heights,
        )

        lsp_fit = fit_fixed_height(
            lsp.lsp_reflector_height_m,
            x,
            detrended_linear,
            wavelength_m=wavelength_m,
        )

        ls_fit = height_search.fit
        height_difference = (
            ls_fit.reflector_height_m
            - lsp.lsp_reflector_height_m
        )

        # Compute empirical normalized amplitude feature using identical shared analysis samples
        norm_amp_emp = compute_normalized_amplitude_empirical(
            ls_amplitude=ls_fit.amplitude,
            polynomial_background=detrending.trend_linear,
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
            number_of_observations=len(sat_analysis),
            elevation_min_deg=float(sat_analysis["elevation_deg"].min()),
            elevation_max_deg=float(sat_analysis["elevation_deg"].max()),
            elevation_raw_min_deg=float(sat_analysis["elevation_raw_deg"].min()),
            elevation_raw_max_deg=float(sat_analysis["elevation_raw_deg"].max()),
            azimuth_min_deg=float(sat_analysis["azimuth_deg"].min()),
            azimuth_max_deg=float(sat_analysis["azimuth_deg"].max()),
            azimuth_mean_deg=float(sat_analysis["azimuth_deg"].mean()),
            refraction_applied=refraction_applied,
            pressure_hpa=pressure_hpa,
            temperature_c=temperature_c,
            pele_min_deg=float(pele[0]),
            pele_max_deg=float(pele[1]),
            e1_deg=float(e1),
            e2_deg=float(e2),
            snr_representation="linear amplitude",
            polynomial_degree=polynomial_degree,
            wavelength_m=wavelength_m,
            lsp_reflector_height_m=lsp.lsp_reflector_height_m,
            lsp_frequency=lsp.lsp_frequency,
            lsp_peak_power=lsp.lsp_peak_power,
            ls_reflector_height_m=ls_fit.reflector_height_m,
            ls_frequency=ls_fit.frequency,
            ls_amplitude=ls_fit.amplitude,
            normalized_amplitude_empirical=norm_amp_emp,
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
            detrended_linear=detrended_linear,
            fitted_signal=ls_fit.fitted,
            residuals=ls_fit.residual,
            lsp_reflector_heights=lsp.reflector_heights,
            lsp_power=lsp.power,
            height_grid=height_search.height_grid,
            sse_values=height_search.sse_values,
            arc_observations=sat_analysis,
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
            elevation_raw_min_deg=None,
            elevation_raw_max_deg=None,
            azimuth_min_deg=None,
            azimuth_max_deg=None,
            azimuth_mean_deg=None,
            refraction_applied=False,
            pressure_hpa=pressure_hpa,
            temperature_c=temperature_c,
            pele_min_deg=float(pele[0]) if pele else None,
            pele_max_deg=float(pele[1]) if pele else None,
            e1_deg=float(e1) if e1 is not None else None,
            e2_deg=float(e2) if e2 is not None else None,
            snr_representation="linear amplitude",
            polynomial_degree=polynomial_degree,
            wavelength_m=wavelength_m,
            lsp_reflector_height_m=None,
            lsp_frequency=None,
            lsp_peak_power=None,
            ls_reflector_height_m=None,
            ls_frequency=None,
            ls_amplitude=None,
            normalized_amplitude_empirical=None,
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
    station_config_file=None,
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
        station_config=station_config_file,
        **kwargs,
    )

