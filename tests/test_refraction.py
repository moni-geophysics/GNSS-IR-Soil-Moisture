"""
Unit tests for the Bennett atmospheric refraction utility.
"""

import numpy as np
import pytest
from gnss_ir.refraction import bennett_refraction_correction

try:
    from gnssrefl.refraction import corr_el_angles as gnssrefl_corr_el_angles
    HAS_GNSSREFL = True
except ImportError:
    HAS_GNSSREFL = False


# Reference benchmark atmospheric conditions (P041 DOY001)
P_REF_HPA = 824.934
T_REF_C = 11.523


def test_bennett_refraction_reference_point_4_9036():
    """
    Test reference calculation: E = 4.9036°, P = 824.934 hPa, T = 11.523°C
    Expected corrected elevation: ~5.039488°
    """
    raw_ele = 4.9036
    corrected = bennett_refraction_correction(raw_ele, P_REF_HPA, T_REF_C)

    assert isinstance(corrected, float)
    assert corrected == pytest.approx(5.039488276, abs=1e-6)
    # Check correction delta is ~0.135888° (~8.15 arcmin)
    correction_deg = corrected - raw_ele
    assert correction_deg == pytest.approx(0.135888276, abs=1e-6)


def test_bennett_refraction_reference_point_4_9657():
    """
    Test reference calculation: E = 4.9657°, P = 824.934 hPa, T = 11.523°C
    Expected corrected elevation: ~5.100234°
    """
    raw_ele = 4.9657
    corrected = bennett_refraction_correction(raw_ele, P_REF_HPA, T_REF_C)

    assert isinstance(corrected, float)
    assert corrected == pytest.approx(5.100233804, abs=1e-6)
    # Check correction delta is ~0.134534° (~8.07 arcmin)
    correction_deg = corrected - raw_ele
    assert correction_deg == pytest.approx(0.134533804, abs=1e-6)


def test_bennett_refraction_vectorized_and_shape():
    """
    Test 1D numpy array and Python list inputs.
    Verify output shape and array type.
    """
    elevations = np.array([4.9036, 4.9657, 10.0, 20.0, 30.0])
    corrected = bennett_refraction_correction(elevations, P_REF_HPA, T_REF_C)

    assert isinstance(corrected, np.ndarray)
    assert corrected.shape == elevations.shape
    assert corrected[0] == pytest.approx(5.039488, abs=1e-6)
    assert corrected[1] == pytest.approx(5.100234, abs=1e-6)

    # Test list input returns ndarray
    list_input = [4.9036, 4.9657]
    corrected_list = bennett_refraction_correction(list_input, P_REF_HPA, T_REF_C)
    assert isinstance(corrected_list, np.ndarray)
    assert len(corrected_list) == 2


@pytest.mark.skipif(not HAS_GNSSREFL, reason="gnssrefl package not available")
def test_bennett_refraction_exact_gnssrefl_parity():
    """
    Verify exact parity against gnssrefl.refraction.corr_el_angles across
    the full GNSS-IR elevation range [5°, 30°].
    """
    elevations = np.linspace(1.0, 30.0, 100)
    our_result = bennett_refraction_correction(elevations, P_REF_HPA, T_REF_C)
    gnssrefl_result = gnssrefl_corr_el_angles(elevations, P_REF_HPA, T_REF_C)

    np.testing.assert_allclose(our_result, gnssrefl_result, rtol=1e-14, atol=1e-14)


def test_bennett_refraction_physical_monotonicity():
    """
    Physical checks:
    1. Correction is strictly positive (bending upward).
    2. Correction magnitude decreases as elevation increases.
    3. Correction increases with higher atmospheric pressure.
    """
    elevations = np.array([5.0, 10.0, 15.0, 20.0, 25.0, 30.0])
    corrected = bennett_refraction_correction(elevations, P_REF_HPA, T_REF_C)
    corrections = corrected - elevations

    # Positive bending
    assert np.all(corrections > 0)
    # Monotonically decreasing with elevation
    assert np.all(np.diff(corrections) < 0)

    # Higher pressure -> larger correction
    corr_high_p = bennett_refraction_correction(elevations, 1013.25, T_REF_C) - elevations
    assert np.all(corr_high_p > corrections)


def test_bennett_refraction_invalid_inputs():
    """
    Test input validation and error handling for out-of-bound inputs.
    """
    # Negative / zero pressure
    with pytest.raises(ValueError, match="Atmospheric pressure must be positive"):
        bennett_refraction_correction(5.0, pressure_hpa=0.0, temperature_c=10.0)
    with pytest.raises(ValueError, match="Atmospheric pressure must be positive"):
        bennett_refraction_correction(5.0, pressure_hpa=-100.0, temperature_c=10.0)

    # Temperature below absolute zero
    with pytest.raises(ValueError, match="Atmospheric temperature must be above absolute zero"):
        bennett_refraction_correction(5.0, pressure_hpa=1013.25, temperature_c=-300.0)

    # Negative / zero elevation
    with pytest.raises(ValueError, match="Elevation angles must be strictly positive"):
        bennett_refraction_correction(0.0, pressure_hpa=1013.25, temperature_c=15.0)
    with pytest.raises(ValueError, match="Elevation angles must be strictly positive"):
        bennett_refraction_correction(-5.0, pressure_hpa=1013.25, temperature_c=15.0)
    with pytest.raises(ValueError, match="Elevation angles must be strictly positive"):
        bennett_refraction_correction([5.0, -1.0], pressure_hpa=1013.25, temperature_c=15.0)

    # Non-finite values
    with pytest.raises(ValueError, match="finite"):
        bennett_refraction_correction([5.0, np.nan], pressure_hpa=1013.25, temperature_c=15.0)
    with pytest.raises(ValueError, match="finite"):
        bennett_refraction_correction(5.0, pressure_hpa=np.nan, temperature_c=15.0)
    with pytest.raises(ValueError, match="finite"):
        bennett_refraction_correction(5.0, pressure_hpa=1013.25, temperature_c=np.inf)


# ============================================================================
# Tests for get_gpt2_refraction_parameters
# ============================================================================

from pathlib import Path
from gnss_ir.refraction import get_gpt2_refraction_parameters

PROJECT_ROOT = Path(__file__).resolve().parent.parent
P041_JSON = PROJECT_ROOT / "data" / "raw" / "gnss" / "input" / "p041" / "p041.json"

P041_CONFIG_DICT = {
    "station": "p041",
    "lat": 39.949492042,
    "lon": -105.194266387,
    "ht": 1728.8555755,
    "refr_model": 1,
}

P037_CONFIG_DICT = {
    "station": "p037",
    "lat": 38.42187,
    "lon": -105.1046,
    "ht": 1621.074,
    "refr_model": 1,
}


@pytest.mark.skipif(not HAS_GNSSREFL, reason="gnssrefl package not available")
def test_get_gpt2_refraction_parameters_dict_and_json():
    """
    Test atmospheric parameter retrieval using dictionary and JSON file path.
    """
    p_dict, t_dict = get_gpt2_refraction_parameters(P041_CONFIG_DICT, year=2016, doy=1)
    p_json, t_json = get_gpt2_refraction_parameters(P041_JSON, year=2016, doy=1)

    assert p_dict == pytest.approx(824.93436, abs=1e-3)
    assert t_dict == pytest.approx(11.52271, abs=1e-3)
    assert p_json == pytest.approx(p_dict, abs=1e-5)
    assert t_json == pytest.approx(t_dict, abs=1e-5)


@pytest.mark.skipif(not HAS_GNSSREFL, reason="gnssrefl package not available")
def test_get_gpt2_refraction_parameters_dates_and_refr_models():
    """
    Test date handling across DOY 1, DOY 180, and leap-year DOY 366.
    Verify refr_model=1 gives static mean values, while refr_model=2 gives time-varying values.
    """
    # refr_model=1 (static model: mean annual values)
    cfg_static = dict(P041_CONFIG_DICT, refr_model=1)
    p_doy1, t_doy1 = get_gpt2_refraction_parameters(cfg_static, year=2016, doy=1)
    p_doy180, t_doy180 = get_gpt2_refraction_parameters(cfg_static, year=2016, doy=180)
    p_doy366, t_doy366 = get_gpt2_refraction_parameters(cfg_static, year=2016, doy=366)

    assert p_doy1 == pytest.approx(p_doy180, abs=1e-6)
    assert t_doy1 == pytest.approx(t_doy180, abs=1e-6)
    assert p_doy1 == pytest.approx(p_doy366, abs=1e-6)
    assert t_doy1 == pytest.approx(t_doy366, abs=1e-6)

    # refr_model=2 (time-varying model: annual/semiannual amplitudes)
    cfg_dynamic = dict(P041_CONFIG_DICT, refr_model=2)
    p_dyn1, t_dyn1 = get_gpt2_refraction_parameters(cfg_dynamic, year=2016, doy=1)
    p_dyn180, t_dyn180 = get_gpt2_refraction_parameters(cfg_dynamic, year=2016, doy=180)

    # Winter (DOY 1) vs Summer (DOY 180) temperature difference
    assert t_dyn1 == pytest.approx(1.8355, abs=1e-2)
    assert t_dyn180 == pytest.approx(21.2845, abs=1e-2)
    assert t_dyn180 > t_dyn1 + 15.0  # Significant seasonal variation


@pytest.mark.skipif(not HAS_GNSSREFL, reason="gnssrefl package not available")
def test_get_gpt2_refraction_parameters_second_station_p037():
    """
    Verify parameter retrieval for a second configured station (P037).
    """
    p_p037_1, t_p037_1 = get_gpt2_refraction_parameters(P037_CONFIG_DICT, year=2016, doy=1)
    p_p037_180, t_p037_180 = get_gpt2_refraction_parameters(P037_CONFIG_DICT, year=2016, doy=180)

    assert p_p037_1 == pytest.approx(835.8079, abs=1e-3)
    assert t_p037_1 == pytest.approx(13.4932, abs=1e-3)
    assert p_p037_1 == pytest.approx(p_p037_180, abs=1e-6)

    # Verify P037 differs from P041 due to different elevation/coordinates
    p_p041, _ = get_gpt2_refraction_parameters(P041_CONFIG_DICT, year=2016, doy=1)
    assert p_p037_1 != pytest.approx(p_p041, abs=1.0)


def test_get_gpt2_refraction_parameters_invalid_station_config():
    """
    Test error handling for missing keys, invalid types, and missing files.
    """
    # Missing required keys
    for key in ["station", "lat", "lon", "ht"]:
        bad_cfg = dict(P041_CONFIG_DICT)
        del bad_cfg[key]
        with pytest.raises(ValueError, match=f"Missing required key '{key}'"):
            get_gpt2_refraction_parameters(bad_cfg, year=2016, doy=1)

    # Invalid config type
    with pytest.raises(ValueError, match="station_config must be a dict or file path"):
        get_gpt2_refraction_parameters(12345, year=2016, doy=1)
    with pytest.raises(ValueError, match="station_config must be a dict or file path"):
        get_gpt2_refraction_parameters(["station", "p041"], year=2016, doy=1)

    # Non-existent file
    with pytest.raises(FileNotFoundError, match="Station configuration file does not exist"):
        get_gpt2_refraction_parameters("nonexistent_station_config.json", year=2016, doy=1)


def test_get_gpt2_refraction_parameters_invalid_dates_and_coordinates():
    """
    Test input validation for invalid year, DOY, and station coordinates.
    """
    # Invalid DOY
    with pytest.raises(ValueError, match="Day of year.*between 1 and 366"):
        get_gpt2_refraction_parameters(P041_CONFIG_DICT, year=2016, doy=0)
    with pytest.raises(ValueError, match="Day of year.*between 1 and 366"):
        get_gpt2_refraction_parameters(P041_CONFIG_DICT, year=2016, doy=400)

    # Invalid year
    with pytest.raises(ValueError, match="Year must be a positive integer"):
        get_gpt2_refraction_parameters(P041_CONFIG_DICT, year=0, doy=1)
    with pytest.raises(ValueError, match="Year must be a positive integer"):
        get_gpt2_refraction_parameters(P041_CONFIG_DICT, year=-2016, doy=1)

    # Invalid latitude / longitude
    bad_lat = dict(P041_CONFIG_DICT, lat=120.0)
    with pytest.raises(ValueError, match="Latitude must be finite and within"):
        get_gpt2_refraction_parameters(bad_lat, year=2016, doy=1)

    bad_coords = dict(P041_CONFIG_DICT, ht="not_a_number")
    with pytest.raises(ValueError, match="Station coordinates.*must be numeric"):
        get_gpt2_refraction_parameters(bad_coords, year=2016, doy=1)

