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

    assert lsp_fit.sse == pytest.approx(27260.077712, abs=1e-6)


def test_p041_doy001_prn1_rising_arc_regression():
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

    assert result.processing_status == "success"
    assert result.failure_reason is None

    assert result.station == "p041"
    assert result.year == 2016
    assert result.doy == 1
    assert result.prn == 1
    assert result.arc_number == 1
    assert result.arc_type == "rising"

    assert result.start_seconds == pytest.approx(16710.0)
    assert result.end_seconds == pytest.approx(20565.0)
    assert result.number_of_observations == 258
    assert result.elevation_min_deg == pytest.approx(5.0176, abs=1e-4)
    assert result.elevation_max_deg == pytest.approx(24.9708, abs=1e-4)

    assert result.lsp_reflector_height_m == pytest.approx(
        1.740848170,
        abs=1e-9,
    )
    assert result.lsp_frequency == pytest.approx(
        18.296438,
        abs=1e-6,
    )
    assert result.lsp_peak_power == pytest.approx(
        0.271864,
        abs=1e-6,
    )

    assert result.ls_reflector_height_m == pytest.approx(
        1.740744772,
        abs=1e-9,
    )
    assert result.ls_frequency == pytest.approx(
        18.295351,
        abs=1e-6,
    )
    assert result.ls_amplitude == pytest.approx(
        8.787386,
        abs=1e-6,
    )
    assert result.ls_phase == pytest.approx(
        -0.904175,
        abs=1e-6,
    )
    assert result.ls_rmse == pytest.approx(
        10.279066,
        abs=1e-6,
    )
    assert result.ls_r_squared == pytest.approx(
        0.271864,
        abs=1e-6,
    )
    assert result.ls_sse == pytest.approx(
        27260.073606,
        abs=1e-6,
    )
    assert result.lsp_vs_ls_height_difference_m == pytest.approx(
        -0.000103398,
        abs=1e-9,
    )

    row = result.to_dict()
    assert "x" not in row
    assert row["processing_status"] == "success"
