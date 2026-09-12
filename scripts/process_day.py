"""
Process all GNSS-IR arcs for one station-day.

This script is intentionally only a daily orchestration layer. It reads
the daily SNR file and arc inventory once, then delegates the scientific
GNSS-IR processing for each arc to src/gnss_ir/processing.py.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from gnss_ir.processing import process_gnss_ir_arc
from gnss_ir.snr import read_snr_file


DEFAULT_STATION = "p041"
DEFAULT_YEAR = 2016
DEFAULT_DOY = 1
DEFAULT_OUTPUT_SUFFIX = "gnss_ir_arc_features"


def format_doy(doy: int) -> str:
    """Return a three-digit day-of-year string."""

    return f"{doy:03d}"


def default_snr_file(project_root: Path, station: str, year: int, doy: int) -> Path:
    """Build the current project SNR filename convention."""

    station = station.lower()
    doy_text = format_doy(doy)
    year_two_digits = str(year)[-2:]

    return (
        project_root
        / "data"
        / "raw"
        / "gnss"
        / str(year)
        / "snr"
        / station
        / f"{station}{doy_text}0.{year_two_digits}.snr66"
    )


def default_arc_inventory_file(project_root: Path, station: str, doy: int) -> Path:
    """Build the current project arc-inventory filename convention."""

    station = station.lower()
    doy_text = format_doy(doy)

    return (
        project_root
        / "data"
        / "processed"
        / station
        / f"{station}_doy{doy_text}_arc_inventory.csv"
    )


def default_output_file(project_root: Path, station: str, doy: int) -> Path:
    """Build the daily GNSS-IR feature-table output path."""

    station = station.lower()
    doy_text = format_doy(doy)

    return (
        project_root
        / "data"
        / "processed"
        / station
        / f"{station}_doy{doy_text}_{DEFAULT_OUTPUT_SUFFIX}.csv"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Process all GNSS-IR arcs for one station-day.",
    )

    parser.add_argument(
        "--station",
        default=DEFAULT_STATION,
        help=f"GNSS station name. Default: {DEFAULT_STATION}",
    )
    parser.add_argument(
        "--year",
        type=int,
        default=DEFAULT_YEAR,
        help=f"Four-digit year. Default: {DEFAULT_YEAR}",
    )
    parser.add_argument(
        "--doy",
        type=int,
        default=DEFAULT_DOY,
        help=f"Day of year. Default: {DEFAULT_DOY}",
    )
    parser.add_argument(
        "--snr-file",
        type=Path,
        default=None,
        help="Optional explicit path to the daily SNR file.",
    )
    parser.add_argument(
        "--arc-inventory-file",
        type=Path,
        default=None,
        help="Optional explicit path to the daily arc inventory CSV.",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=None,
        help="Optional explicit path for the daily output CSV.",
    )

    return parser.parse_args()


def build_failure_row(
    inventory_row: pd.Series,
    station: str,
    year: int,
    doy: int,
    failure_reason: str,
) -> dict:
    """Create one output row when a specific inventory arc cannot run."""

    row = {
        "station": station,
        "year": year,
        "doy": doy,
        "prn": int(inventory_row["satellite"]),
        "arc_number": int(inventory_row["arc_number"]),
        "arc_type": inventory_row["arc_type"],
        "start_seconds": inventory_row.get("start_seconds"),
        "end_seconds": inventory_row.get("end_seconds"),
        "number_of_observations": 0,
        "elevation_min_deg": None,
        "elevation_max_deg": None,
        "azimuth_min_deg": None,
        "azimuth_max_deg": None,
        "azimuth_mean_deg": None,
        "snr_representation": "linear amplitude",
        "polynomial_degree": 2,
        "wavelength_m": 0.19029367,
        "lsp_reflector_height_m": None,
        "lsp_frequency": None,
        "lsp_peak_power": None,
        "ls_reflector_height_m": None,
        "ls_frequency": None,
        "ls_amplitude": None,
        "ls_phase": None,
        "ls_rmse": None,
        "ls_r_squared": None,
        "ls_sse": None,
        "lsp_sse": None,
        "sse_reduction": None,
        "lsp_vs_ls_height_difference_m": None,
        "lsp_vs_ls_absolute_height_difference_m": None,
        "processing_status": "failed",
        "failure_reason": failure_reason,
    }

    return add_inventory_qc(row, inventory_row)


def add_inventory_qc(result_row: dict, inventory_row: pd.Series) -> dict:
    """Attach arc-inventory metadata useful for later QC decisions."""

    qc_columns = {
        "observations": "inventory_observations",
        "duration_sec": "inventory_duration_sec",
        "duration_min": "inventory_duration_min",
        "min_elevation_deg": "inventory_min_elevation_deg",
        "max_elevation_deg": "inventory_max_elevation_deg",
        "elevation_span_deg": "inventory_elevation_span_deg",
        "max_time_gap_sec": "inventory_max_time_gap_sec",
        "mean_consistency_error_deg": "inventory_mean_consistency_error_deg",
    }

    for source_column, output_column in qc_columns.items():
        if source_column in inventory_row:
            result_row[output_column] = inventory_row[source_column]

    return result_row


def process_day(
    station: str,
    year: int,
    doy: int,
    snr_file: Path,
    arc_inventory_file: Path,
) -> pd.DataFrame:
    """
    Process every arc in one daily arc inventory.

    Returns one row per inventory arc. Failed arcs are retained with
    processing_status='failed' and a failure_reason.
    """

    station = station.lower()

    snr_data = read_snr_file(snr_file)
    arc_inventory = pd.read_csv(arc_inventory_file)

    results = []

    for _, inventory_row in arc_inventory.iterrows():
        try:
            result = process_gnss_ir_arc(
                snr_data=snr_data,
                arc_inventory=arc_inventory,
                satellite_number=int(inventory_row["satellite"]),
                arc_type=str(inventory_row["arc_type"]),
                arc_number=int(inventory_row["arc_number"]),
                station=station,
                year=year,
                doy=doy,
            )

            result_row = result.to_dict()
            result_row = add_inventory_qc(result_row, inventory_row)

        except Exception as exc:
            result_row = build_failure_row(
                inventory_row=inventory_row,
                station=station,
                year=year,
                doy=doy,
                failure_reason=str(exc),
            )

        results.append(result_row)

    return pd.DataFrame(results)


def main() -> None:
    args = parse_args()

    station = args.station.lower()
    year = args.year
    doy = args.doy

    snr_file = args.snr_file or default_snr_file(
        PROJECT_ROOT,
        station,
        year,
        doy,
    )
    arc_inventory_file = args.arc_inventory_file or default_arc_inventory_file(
        PROJECT_ROOT,
        station,
        doy,
    )
    output_file = args.output_file or default_output_file(
        PROJECT_ROOT,
        station,
        doy,
    )

    print("GNSS-IR DAILY ARC PROCESSING")
    print("============================")
    print("Station:", station)
    print("Year:", year)
    print("DOY:", format_doy(doy))
    print("SNR file:", snr_file)
    print("Arc inventory:", arc_inventory_file)
    print("Output file:", output_file)

    daily_results = process_day(
        station=station,
        year=year,
        doy=doy,
        snr_file=snr_file,
        arc_inventory_file=arc_inventory_file,
    )

    output_file.parent.mkdir(parents=True, exist_ok=True)
    daily_results.to_csv(output_file, index=False)

    success_count = int((daily_results["processing_status"] == "success").sum())
    failure_count = int((daily_results["processing_status"] == "failed").sum())

    print()
    print("SUMMARY")
    print("-------")
    print("Inventory arcs:", len(daily_results))
    print("Successful arcs:", success_count)
    print("Failed arcs:", failure_count)
    print("Saved:", output_file)


if __name__ == "__main__":
    main()
