"""
Atmospheric refraction correction for GNSS elevation angles.

This module implements the empirical atmospheric refraction bending model
developed by G. G. Bennett (1982), which is used by gnssrefl 4.2.2 to compute
elevation-angle corrections from surface atmospheric pressure and temperature.

References
----------
Bennett, G. G. (1982). The Calculation of Astronomical Refraction in Marine
    Navigation. The Journal of Navigation, 35(2), 255-259.
"""

from typing import Union
import numpy as np


def bennett_refraction_correction(
    elevation_deg: Union[float, int, np.ndarray, list],
    pressure_hpa: float,
    temperature_c: float,
) -> Union[float, np.ndarray]:
    """
    Compute atmospheric refraction correction for GNSS elevation angles using
    the Bennett (1982) empirical formula.

    The model computes the angular bending (in degrees) added to the apparent
    (raw) elevation angle as a function of surface pressure and temperature:

        corr_arcmin = 510 / (9/5 * T + 492) * (P / 1010.16) / tan(E + 7.31 / (E + 4.4))
        corr_deg = corr_arcmin / 60
        E_corrected = E + corr_deg

    Parameters
    ----------
    elevation_deg : float, int, list, or np.ndarray
        Raw (apparent) elevation angle(s) in degrees. Values must be > 0°.
    pressure_hpa : float
        Surface atmospheric pressure in hPa (mbar). Must be > 0.
    temperature_c : float
        Surface atmospheric temperature in degrees Celsius (°C). Must be > -273.15°C.

    Returns
    -------
    float or np.ndarray
        Refraction-corrected elevation angle(s) in degrees. The return type
        matches the scalar or array nature of the input.

    Raises
    ------
    ValueError
        If pressure <= 0, temperature <= -273.15°C, any elevation angle <= 0°,
        or if any input contains non-finite values (NaN / Inf).

    Notes
    -----
    - Model Applicable Range: Developed for marine navigation and optical/radio
      astronomy down to the horizon ($E \\ge 1.5^\\circ$ recommended, valid
      for $E > 0^\\circ$). The formula contains a singularity at $E = -4.4^\\circ$.
    - Agreement: Matches `gnssrefl.refraction.corr_el_angles` identically.
    """
    # 1. Validate atmospheric scalar inputs
    if not np.isfinite(pressure_hpa) or pressure_hpa <= 0:
        raise ValueError(
            f"Atmospheric pressure must be positive and finite, got {pressure_hpa} hPa."
        )

    if not np.isfinite(temperature_c) or temperature_c <= -273.15:
        raise ValueError(
            f"Atmospheric temperature must be above absolute zero (-273.15°C) and finite, got {temperature_c}°C."
        )

    # 2. Determine input structure
    is_scalar = isinstance(elevation_deg, (int, float, np.floating, np.integer))
    el_arr = np.asarray(elevation_deg, dtype=float)

    if not np.all(np.isfinite(el_arr)):
        raise ValueError("Elevation angles must be finite (no NaN or Inf values).")

    if np.any(el_arr <= 0.0):
        raise ValueError(
            "Elevation angles must be strictly positive (> 0°) for Bennett refraction."
        )

    # 3. Bennett (1982) formula (exact match to gnssrefl.refraction.corr_el_angles)
    temp_factor = 510.0 / ((9.0 / 5.0) * temperature_c + 492.0)
    press_factor = pressure_hpa / 1010.16
    apparent_argument_deg = el_arr + 7.31 / (el_arr + 4.4)
    corr_el_arc_min = temp_factor * press_factor * (1.0 / np.tan(np.deg2rad(apparent_argument_deg)))
    correction_deg = corr_el_arc_min / 60.0

    corrected_el = el_arr + correction_deg

    if is_scalar:
        return float(corrected_el.item())
    return corrected_el


def get_gpt2_refraction_parameters(
    station_config: Union[dict, str],
    year: int,
    doy: int,
) -> tuple[float, float]:
    """
    Retrieve GPT2 surface pressure (hPa) and temperature (°C) for a given
    station configuration and epoch (year, doy).

    Parameters
    ----------
    station_config : dict or str
        Station configuration dictionary or path to station JSON file containing
        at minimum: 'station', 'lat', 'lon', 'ht'.
    year : int
        Calendar year (e.g. 2016).
    doy : int
        Day of year (e.g. 1).

    Returns
    -------
    tuple of (float, float)
        (pressure_hpa, temperature_c) computed by the GPT2 model.

    Raises
    ------
    ValueError
        If required station metadata ('station', 'lat', 'lon', 'ht') is missing.
    ImportError
        If gnssrefl.refraction or gnssrefl.gps is not available.
    """
    import json
    from pathlib import Path

    if isinstance(station_config, (str, Path)):
        config_path = Path(station_config)
        if not config_path.is_file():
            raise FileNotFoundError(
                f"Station configuration file does not exist: {config_path}"
            )
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    elif isinstance(station_config, dict):
        cfg = station_config
    else:
        raise ValueError(
            f"station_config must be a dict or file path, got {type(station_config)}."
        )

    for key in ["station", "lat", "lon", "ht"]:
        if key not in cfg:
            raise ValueError(f"Missing required key '{key}' in station_config.")

    if not isinstance(year, int) or year < 1:
        raise ValueError(f"Year must be a positive integer, got {year}.")

    if not isinstance(doy, int) or not (1 <= doy <= 366):
        raise ValueError(f"Day of year (doy) must be between 1 and 366, got {doy}.")

    try:
        lat = float(cfg["lat"])
        lon = float(cfg["lon"])
        ht = float(cfg["ht"])
    except (ValueError, TypeError) as exc:
        raise ValueError(f"Station coordinates ('lat', 'lon', 'ht') must be numeric: {exc}") from exc

    if not np.isfinite(lat) or not (-90.0 <= lat <= 90.0):
        raise ValueError(f"Latitude must be finite and within [-90, 90] degrees, got {lat}.")

    if not np.isfinite(lon) or not (-360.0 <= lon <= 360.0):
        raise ValueError(f"Longitude must be finite and within [-360, 360] degrees, got {lon}.")

    if not np.isfinite(ht):
        raise ValueError(f"Height must be finite, got {ht}.")

    try:
        import os
        from gnssrefl.refraction import gpt2_1w, readWrite_gpt2_1w
        import gnssrefl.gps as g
    except ImportError as exc:
        raise ImportError(
            "gnssrefl package is required for GPT2 atmospheric parameter retrieval."
        ) from exc

    station = str(cfg["station"]).lower()
    refr_model = int(cfg.get("refr_model", 1))

    # Compute modified Julian date for DOY at 00:00 UTC
    try:
        d = g.doy2ymd(int(year), int(doy))
        dmjd, _ = g.mjd(int(year), d.month, d.day, 0, 0, 0)
    except Exception as exc:
        raise ValueError(f"Failed to compute MJD for year={year}, doy={doy}: {exc}") from exc

    # Resolve REFL_CODE for GPT grid cache
    xdir = os.environ.get("REFL_CODE")
    if not xdir:
        # Fallback to repository data directory if present
        default_refl = Path(__file__).resolve().parent.parent.parent / "data" / "raw" / "gnss"
        if default_refl.exists():
            xdir = str(default_refl)
            os.environ["REFL_CODE"] = xdir

    if xdir:
        readWrite_gpt2_1w(xdir, station, lat, lon)

    it = 0 if refr_model in [2, 4, 5, 6] else 1
    dlat = lat * np.pi / 180.0
    dlong = lon * np.pi / 180.0

    p, T, _, _, _, _, _, _, _ = gpt2_1w(station, dmjd, dlat, dlong, ht, it)
    if not np.isfinite(p) or not np.isfinite(T):
        raise ValueError(f"GPT2 returned non-finite atmospheric parameters (p={p}, T={T}).")

    return float(p), float(T)


