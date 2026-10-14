from pathlib import Path

import numpy as np
import pytest

from gnss_ir.processing import (
    dbhz_to_linear_amplitude,
    fit_fixed_height,
    process_gnss_ir_arc_from_files,
)
from gnss_ir.snr import read_snr_file


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

ARC_INVENTORY_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "p041"
    / "p041_doy001_arc_inventory.csv"
)


def test_read_snr_file_has_expected_columns():
    data = read_snr_file(SNR_FILE)

    assert len(data) == 33908
    assert list(data.columns) == [
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


def test_dbhz_to_linear_amplitude_matches_current_formula():
    snr_db = np.array([0.0, 20.0, 40.0])

    linear = dbhz_to_linear_amplitude(snr_db)

    np.testing.assert_allclose(
        linear,
        np.array([1.0, 10.0, 100.0]),
    )


def test_fixed_height_fit_returns_expected_lsp_sse():
    result = process_gnss_ir_arc_from_files(
        snr_file=SNR_FILE,
        arc_inventory_file=ARC_INVENTORY_FILE,
        satellite_number=1,
        arc_type="rising",
        arc_number=1,
        station="p041",
        year=2016,
        doy=1,
    )

    lsp_fit = fit_fixed_height(
        result.lsp_reflector_height_m,
        result.x,
        result.detrended_linear,
    )

    assert lsp_fit.sse == pytest.approx(result.lsp_sse, abs=1e-6)
    assert lsp_fit.sse == pytest.approx(29105.929064, abs=1e-3)


def test_p041_doy001_prn1_benchmark_regression():
    """
    Validate standard benchmark mode (Bennett refraction + pele [5, 30] + polyV=4 + [5, 25])
    on PRN 1 rising arc (arc #3, t in [70680, 73965] s).
    Target: H_LSP ≈ 1.935 m, H_LS ≈ 1.934 m, N = 216 (refraction adds lower boundary points).
    """
    result = process_gnss_ir_arc_from_files(
        snr_file=SNR_FILE,
        arc_inventory_file=ARC_INVENTORY_FILE,
        satellite_number=1,
        arc_type="rising",
        arc_number=3,
        station="p041",
        year=2016,
        doy=1,
        refraction=True,
        polynomial_degree=4,
        pele=(5.0, 30.0),
        e1=5.0,
        e2=25.0,
        minimum_height_m=0.5,
        maximum_height_m=8.0,
    )

    assert result.processing_status == "success"
    assert result.failure_reason is None
    assert result.refraction_applied is True
    assert result.pressure_hpa == pytest.approx(824.934, abs=1e-2)
    assert result.temperature_c == pytest.approx(11.523, abs=1e-2)

    # Observation count and elevation verification: raw vs corrected
    assert result.number_of_observations == 216
    assert result.elevation_raw_min_deg == pytest.approx(4.9036, abs=1e-4)
    assert result.elevation_min_deg == pytest.approx(5.0395, abs=1e-4)
    assert result.elevation_max_deg == pytest.approx(24.9452, abs=1e-4)

    # Reflector height retrieval (benchmark match)
    assert result.lsp_reflector_height_m == pytest.approx(1.9343, abs=0.005)
    assert result.ls_reflector_height_m == pytest.approx(1.9339, abs=0.005)
    assert result.ls_amplitude == pytest.approx(8.14, abs=0.1)

    # Output structure
    row = result.to_dict()
    assert row["refraction_applied"] is True
    assert row["pressure_hpa"] is not None
    assert row["elevation_raw_min_deg"] is not None


def test_raw_and_corrected_elevation_handling():
    """
    Ensure raw elevations and refraction-corrected elevations are both preserved
    and distinct when refraction is applied.
    """
    result = process_gnss_ir_arc_from_files(
        snr_file=SNR_FILE,
        arc_inventory_file=ARC_INVENTORY_FILE,
        satellite_number=1,
        arc_type="rising",
        arc_number=3,
        station="p041",
        year=2016,
        doy=1,
        refraction=True,
        pressure_hpa=824.934,
        temperature_c=11.523,
    )

    assert result.refraction_applied is True
    assert result.elevation_raw_min_deg < result.elevation_min_deg
    assert result.elevation_raw_min_deg == pytest.approx(4.9036, abs=1e-4)
    assert result.elevation_min_deg == pytest.approx(5.0395, abs=1e-4)
    assert result.arc_observations is not None
    assert "elevation_raw_deg" in result.arc_observations.columns
    assert "elevation_refr_deg" in result.arc_observations.columns
    assert "sin_elevation" in result.arc_observations.columns


def test_refraction_missing_parameters_fails_cleanly():
    """
    Ensure processing fails with a clear failure_reason when refraction is requested
    but atmospheric parameters/station config cannot be retrieved.
    """
    result = process_gnss_ir_arc_from_files(
        snr_file=SNR_FILE,
        arc_inventory_file=ARC_INVENTORY_FILE,
        satellite_number=1,
        arc_type="rising",
        arc_number=3,
        station="unknown_station_xyz",
        year=2016,
        doy=1,
        refraction=True,
        pressure_hpa=None,
        temperature_c=None,
    )

    assert result.processing_status == "failed"
    assert result.refraction_applied is False
    assert result.failure_reason is not None
    assert "Refraction correction was requested" in result.failure_reason


def test_no_refraction_mode_succeeds_without_atmospheric_parameters():
    """
    Ensure processing succeeds without requiring atmospheric parameters when
    refraction is explicitly disabled (refraction=False).
    """
    result = process_gnss_ir_arc_from_files(
        snr_file=SNR_FILE,
        arc_inventory_file=ARC_INVENTORY_FILE,
        satellite_number=1,
        arc_type="rising",
        arc_number=3,
        station="unknown_station_xyz",
        year=2016,
        doy=1,
        refraction=False,
    )

    assert result.processing_status == "success"
    assert result.refraction_applied is False
    assert result.failure_reason is None
    assert result.number_of_observations > 0


def test_below_horizon_observations_in_buffer_masked_safely():
    """
    Ensure that below-horizon observations (elevation <= 0°) in the contiguous arc buffer
    are safely masked during Bennett refraction calculation and do not cause processing to fail,
    while valid in-band observations are processed correctly.
    """
    import pandas as pd
    from gnss_ir.processing import process_gnss_ir_arc

    snr_data = read_snr_file(SNR_FILE)
    arc_inventory = pd.read_csv(ARC_INVENTORY_FILE)

    # Artificially inject a nonpositive (<= 0) elevation point in PRN 1 observations outside pele
    modified_snr = snr_data.copy()
    prn1_mask = modified_snr["satellite"] == 1
    # Find points with elevation < 5 deg or create one
    idx_to_modify = modified_snr[prn1_mask & (modified_snr["elevation_deg"] < 5.0)].index
    if len(idx_to_modify) > 0:
        modified_snr.loc[idx_to_modify[0], "elevation_deg"] = 0.0

    result = process_gnss_ir_arc(
        snr_data=modified_snr,
        arc_inventory=arc_inventory,
        satellite_number=1,
        arc_type="rising",
        arc_number=3,
        station="p041",
        year=2016,
        doy=1,
        refraction=True,
        pressure_hpa=824.934,
        temperature_c=11.523,
    )

    assert result.processing_status == "success"
    assert result.failure_reason is None
    assert result.refraction_applied is True
    assert result.number_of_observations > 0
    # Reflector height remains consistent with the benchmark
    assert result.lsp_reflector_height_m == pytest.approx(1.9343, abs=0.005)


def test_genuine_out_of_band_window_fails_cleanly():
    """
    Ensure that if an arc has no observations in the required fitting window (pele)
    or analysis window (e1, e2), processing fails cleanly with a descriptive failure_reason.
    """
    import pandas as pd
    from gnss_ir.processing import process_gnss_ir_arc

    snr_data = read_snr_file(SNR_FILE)
    arc_inventory = pd.read_csv(ARC_INVENTORY_FILE)

    # Set elevation window to an impossible range [80, 85] deg
    result = process_gnss_ir_arc(
        snr_data=snr_data,
        arc_inventory=arc_inventory,
        satellite_number=1,
        arc_type="rising",
        arc_number=3,
        station="p041",
        year=2016,
        doy=1,
        refraction=True,
        pressure_hpa=824.934,
        temperature_c=11.523,
        pele=(80.0, 85.0),
        e1=80.0,
        e2=85.0,
    )

    assert result.processing_status == "failed"
    assert result.failure_reason is not None
    assert "No observations remain in fitting window" in result.failure_reason


def test_compute_normalized_amplitude_empirical_unit():
    """
    Test compute_normalized_amplitude_empirical for valid cases, invalid denominators,
    zero/negative values, non-finite values, and empty inputs.
    """
    from gnss_ir.processing import compute_normalized_amplitude_empirical

    # 1. Valid inputs: alpha = A_LS / mean(P)
    bg = np.array([50.0, 100.0, 150.0])  # mean = 100.0
    res = compute_normalized_amplitude_empirical(ls_amplitude=10.0, polynomial_background=bg)
    assert res == pytest.approx(0.10)

    # 2. Non-finite or missing ls_amplitude
    assert compute_normalized_amplitude_empirical(None, bg) is None
    assert compute_normalized_amplitude_empirical(np.nan, bg) is None
    assert compute_normalized_amplitude_empirical(np.inf, bg) is None

    # 3. Non-finite or empty polynomial background
    assert compute_normalized_amplitude_empirical(10.0, np.array([])) is None
    assert compute_normalized_amplitude_empirical(10.0, None) is None
    assert compute_normalized_amplitude_empirical(10.0, np.array([50.0, np.nan, 100.0])) is None

    # 4. Zero denominator
    bg_zero = np.array([0.0, 0.0, 0.0])
    assert compute_normalized_amplitude_empirical(10.0, bg_zero) is None

    # 5. Negative denominator
    bg_neg = np.array([-5.0, -15.0])
    assert compute_normalized_amplitude_empirical(10.0, bg_neg) is None


def test_normalized_amplitude_empirical_p041_doy001_prn1():
    """
    Verify normalized_amplitude_empirical on standard P041 DOY 001 PRN 1 rising arc.
    Ensures shared sample window, identical observation count, and numerical consistency.
    """
    result = process_gnss_ir_arc_from_files(
        snr_file=SNR_FILE,
        arc_inventory_file=ARC_INVENTORY_FILE,
        satellite_number=1,
        arc_type="rising",
        arc_number=3,
        station="p041",
        year=2016,
        doy=1,
        refraction=True,
        pressure_hpa=824.934,
        temperature_c=11.523,
    )

    assert result.processing_status == "success"
    assert result.normalized_amplitude_empirical is not None

    # Check exact value matches A_LS / mean(P(theta)) on the shared 216 samples
    assert result.number_of_observations == 216
    assert result.ls_amplitude == pytest.approx(8.1373, abs=1e-3)
    # Mean background is ~95.7748 -> alpha_arc ~ 0.08496
    assert result.normalized_amplitude_empirical == pytest.approx(0.084963, abs=1e-5)

    # Ensure field is present in to_dict output
    row = result.to_dict()
    assert "normalized_amplitude_empirical" in row
    assert row["normalized_amplitude_empirical"] == pytest.approx(0.084963, abs=1e-5)




