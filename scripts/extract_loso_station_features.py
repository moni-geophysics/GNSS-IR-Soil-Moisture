"""
GNSS-IR Multi-Worker Batch Feature Extraction for LOSO Stations (P037, P043).
Strictly mirrors established P041 processing pipeline:
- Degree-4 detrending (pele: 5-30 deg, analysis: 5-25 deg).
- Atmospheric refraction (GPT2 pressure/temp + Bennett correction).
- Lomb-Scargle periodogram (0.5 to 8.0 m search limits, 5000 heights).
- Fast vectorized analytical least-squares fitting (amplitude, phase, RMSE, R²).
- Arc-level QC filtering (Config A: amp>=5.0, R²>=0.10, obs>=80, |H_LSP - H_LS|<=0.05 m, H in [1.6, 2.4] m).
- Multi-processing across days for high throughput.
- Daily aggregation (mean/median, circular phase metrics).
- Zero modification to P041 files.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from gnss_ir.processing import process_gnss_ir_arc
from gnss_ir.refraction import get_gpt2_refraction_parameters
from gnss_ir.snr import read_snr_file
from scratch.process_full_year_2016 import (
    attach_qc_flags,
    extract_arc_inventory_consecutive,
)
from scripts.prepare_daily_features import (
    compute_circular_mean,
    compute_circular_median,
)


def get_station_snr_file(station: str, year: int, doy: int) -> Path:
    station = station.lower()
    doy_str = f"{doy:03d}"
    yy = str(year)[-2:]
    snr_dir = PROJECT_ROOT / "data" / "raw" / "gnss" / str(year) / "snr" / station
    raw_p = snr_dir / f"{station}{doy_str}0.{yy}.snr66"
    gz_p = snr_dir / f"{station}{doy_str}0.{yy}.snr66.gz"
    if raw_p.exists():
        return raw_p
    return gz_p


def process_single_station_day(
    station: str,
    year: int,
    doy: int,
    output_dir: Path,
    station_config_path: Path,
) -> Dict[str, Any]:
    """Process all arcs for one station-day and return daily summary."""
    station = station.lower()
    doy_str = f"{doy:03d}"
    snr_file = get_station_snr_file(station, year, doy)

    daily_features_path = output_dir / f"{station}_doy{doy_str}_gnss_ir_arc_features.csv"
    inv_file = output_dir / f"{station}_doy{doy_str}_arc_inventory.csv"

    # Fast resume if already processed
    if daily_features_path.exists() and daily_features_path.stat().st_size > 1000:
        try:
            df_day = pd.read_csv(daily_features_path)
            if len(df_day) > 0 and "qc_flagged" in df_day.columns:
                succ_df = df_day[df_day["processing_status"] == "success"]
                n_succ = len(succ_df)
                n_fail = len(df_day) - n_succ
                n_flagged = int(df_day["qc_flagged"].sum())
                p_val = float(df_day["pressure_hpa"].iloc[0]) if "pressure_hpa" in df_day and n_succ > 0 else np.nan
                t_val = float(df_day["temperature_c"].iloc[0]) if "temperature_c" in df_day and n_succ > 0 else np.nan
                return {
                    "station": station,
                    "year": year,
                    "doy": doy,
                    "status": "processed",
                    "snr_file": str(snr_file),
                    "total_arcs": len(df_day),
                    "success_arcs": n_succ,
                    "failed_arcs": n_fail,
                    "pressure_hpa": p_val,
                    "temperature_c": t_val,
                    "rh_lsp_median": float(succ_df["lsp_reflector_height_m"].median()) if n_succ > 0 else np.nan,
                    "rh_ls_median": float(succ_df["ls_reflector_height_m"].median()) if n_succ > 0 else np.nan,
                    "qc_flagged_arcs": n_flagged,
                    "resumed": True,
                }
        except Exception:
            pass

    if not snr_file.exists():
        return {
            "station": station,
            "year": year,
            "doy": doy,
            "status": "missing_snr_file",
            "snr_file": str(snr_file),
            "total_arcs": 0,
            "success_arcs": 0,
            "failed_arcs": 0,
            "pressure_hpa": np.nan,
            "temperature_c": np.nan,
            "rh_lsp_median": np.nan,
            "rh_ls_median": np.nan,
            "qc_flagged_arcs": 0,
            "resumed": False,
        }

    # 1. Read SNR file
    snr_data = read_snr_file(snr_file)

    # 2. Extract or read inventory
    if inv_file.exists():
        arc_inventory = pd.read_csv(inv_file)
    else:
        arc_inventory = extract_arc_inventory_consecutive(snr_data)
        arc_inventory.to_csv(inv_file, index=False)

    # 3. Retrieve GPT2 refraction parameters
    p_val, t_val = get_gpt2_refraction_parameters(station_config_path, year, doy)

    # 4. Process arcs
    day_results = []
    for _, row in arc_inventory.iterrows():
        try:
            res = process_gnss_ir_arc(
                snr_data=snr_data,
                arc_inventory=arc_inventory,
                satellite_number=int(row["satellite"]),
                arc_type=str(row["arc_type"]),
                arc_number=int(row["arc_number"]),
                station=station,
                year=year,
                doy=doy,
                pressure_hpa=p_val,
                temperature_c=t_val,
                refraction=True,
                minimum_height_m=0.5,
                maximum_height_m=8.0,
            )
            res_dict = res.to_dict()
            for col in [
                "duration_sec",
                "duration_min",
                "elevation_span_deg",
                "max_time_gap_sec",
                "mean_consistency_error_deg",
            ]:
                if col in row:
                    res_dict[f"inventory_{col}"] = row[col]
            day_results.append(res_dict)
        except Exception as exc:
            res_dict = {
                "station": station,
                "year": year,
                "doy": doy,
                "prn": int(row["satellite"]),
                "arc_number": int(row["arc_number"]),
                "arc_type": str(row["arc_type"]),
                "processing_status": "failed",
                "failure_reason": str(exc),
            }
            day_results.append(res_dict)

    df_day = pd.DataFrame(day_results)
    df_day = attach_qc_flags(df_day, config="A")
    df_day.to_csv(daily_features_path, index=False)

    succ_df = df_day[df_day["processing_status"] == "success"]
    n_succ = len(succ_df)
    n_fail = len(df_day) - n_succ
    n_flagged = int(df_day["qc_flagged"].sum())

    return {
        "station": station,
        "year": year,
        "doy": doy,
        "status": "processed",
        "snr_file": str(snr_file),
        "total_arcs": len(df_day),
        "success_arcs": n_succ,
        "failed_arcs": n_fail,
        "pressure_hpa": p_val,
        "temperature_c": t_val,
        "rh_lsp_median": float(succ_df["lsp_reflector_height_m"].median()) if n_succ > 0 else np.nan,
        "rh_ls_median": float(succ_df["ls_reflector_height_m"].median()) if n_succ > 0 else np.nan,
        "qc_flagged_arcs": n_flagged,
        "resumed": False,
    }


def aggregate_station_daily_features(
    station: str,
    year: int,
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate clean arcs into daily mean and median datasets."""
    station = station.lower()
    annual_csv = output_dir / f"{station}_{year}_annual_gnss_ir_features.csv"
    if not annual_csv.exists():
        raise FileNotFoundError(f"Annual features table not found: {annual_csv}")

    df_annual = pd.read_csv(annual_csv)
    clean_mask = df_annual["qc_flagged"] == False
    df_clean = df_annual[clean_mask].copy()

    daily_rows = []
    for doy in range(1, 367):
        date_str = str(pd.to_datetime(f"{year}-{doy:03d}", format="%Y-%j").date())
        sub_raw = df_annual[df_annual["doy"] == doy]
        sub_clean = df_clean[df_clean["doy"] == doy]

        n_raw = len(sub_raw)
        n_clean = len(sub_clean)

        if n_clean > 0:
            h_vals = sub_clean["ls_reflector_height_m"].to_numpy()
            amp_vals = sub_clean["ls_amplitude"].to_numpy()
            phase_vals = sub_clean["ls_phase"].to_numpy()

            rh_mean = float(np.mean(h_vals))
            amp_mean = float(np.mean(amp_vals))
            circ_mean, r_len = compute_circular_mean(phase_vals)

            rh_median = float(np.median(h_vals))
            amp_median = float(np.median(amp_vals))
            circ_median = compute_circular_median(phase_vals)

            daily_rows.append({
                "station": station,
                "date": date_str,
                "year": year,
                "doy": doy,
                "raw_arc_count": n_raw,
                "clean_arc_count": n_clean,
                "qc_flagged_count": n_raw - n_clean,
                "qc_pass_rate": float(n_clean / n_raw) if n_raw > 0 else 0.0,
                # Mean features
                "rh_mean": rh_mean,
                "amp_mean": amp_mean,
                "phase_circ_mean": circ_mean,
                "phase_cos_mean": float(np.cos(circ_mean)),
                "phase_sin_mean": float(np.sin(circ_mean)),
                # Median features
                "rh_median": rh_median,
                "amp_median": amp_median,
                "phase_circ_median": circ_median,
                "phase_cos_median": float(np.cos(circ_median)),
                "phase_sin_median": float(np.sin(circ_median)),
                "phase_resultant_length": r_len,
            })
        else:
            daily_rows.append({
                "station": station,
                "date": date_str,
                "year": year,
                "doy": doy,
                "raw_arc_count": n_raw,
                "clean_arc_count": 0,
                "qc_flagged_count": n_raw,
                "qc_pass_rate": 0.0,
                "rh_mean": np.nan,
                "amp_mean": np.nan,
                "phase_circ_mean": np.nan,
                "phase_cos_mean": np.nan,
                "phase_sin_mean": np.nan,
                "rh_median": np.nan,
                "amp_median": np.nan,
                "phase_circ_median": np.nan,
                "phase_cos_median": np.nan,
                "phase_sin_median": np.nan,
                "phase_resultant_length": np.nan,
            })

    df_agg = pd.DataFrame(daily_rows)

    # Split into mean and median datasets matching P041 conventions
    mean_df = df_agg[[
        "station", "date", "year", "doy", "raw_arc_count", "clean_arc_count",
        "qc_flagged_count", "qc_pass_rate", "rh_mean", "amp_mean",
        "phase_circ_mean", "phase_cos_mean", "phase_sin_mean", "phase_resultant_length",
    ]].rename(columns={
        "rh_mean": "rh",
        "amp_mean": "amp",
        "phase_circ_mean": "phase_circ",
        "phase_cos_mean": "phase_cos",
        "phase_sin_mean": "phase_sin",
    })

    median_df = df_agg[[
        "station", "date", "year", "doy", "raw_arc_count", "clean_arc_count",
        "qc_flagged_count", "qc_pass_rate", "rh_median", "amp_median",
        "phase_circ_median", "phase_cos_median", "phase_sin_median", "phase_resultant_length",
    ]].rename(columns={
        "rh_median": "rh",
        "amp_median": "amp",
        "phase_circ_median": "phase_circ",
        "phase_cos_median": "phase_cos",
        "phase_sin_median": "phase_sin",
    })

    mean_csv = output_dir / f"{station}_{year}_daily_mean_dataset.csv"
    median_csv = output_dir / f"{station}_{year}_daily_median_dataset.csv"
    mean_df.to_csv(mean_csv, index=False)
    median_df.to_csv(median_csv, index=False)

    print(f"[{station.upper()}] Exported daily mean features: {mean_csv.name} ({len(mean_df)} rows)")
    print(f"[{station.upper()}] Exported daily median features: {median_csv.name} ({len(median_df)} rows)")

    return mean_df, median_df


def process_station_batch(
    station: str,
    year: int = 2016,
    doy_start: int = 1,
    doy_end: int = 366,
    max_workers: int = 6,
) -> None:
    station = station.lower()
    print("=" * 80)
    print(f"STARTING BATCH FEATURE EXTRACTION: {station.upper()} (Year {year}, DOY {doy_start:03d} - {doy_end:03d})")
    print(f"Concurrency: {max_workers} processes")
    print("=" * 80)

    output_dir = PROJECT_ROOT / "data" / "processed" / station
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = PROJECT_ROOT / "data" / "raw" / "gnss" / "input" / station / f"{station}.json"

    t0 = time.time()
    summaries = []

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                process_single_station_day,
                station,
                year,
                doy,
                output_dir,
                cfg_path,
            ): doy
            for doy in range(doy_start, doy_end + 1)
        }

        completed = 0
        total = len(futures)
        for future in as_completed(futures):
            res = future.result()
            summaries.append(res)
            completed += 1
            if completed % 25 == 0 or completed == total:
                elapsed = time.time() - t0
                rate = completed / elapsed
                rem_sec = (total - completed) / rate if rate > 0 else 0
                print(
                    f"[{station.upper()} PROGRESS] {completed}/{total} days completed "
                    f"({completed/total*100:.1f}%) | {rate:.1f} days/sec | ETA: {rem_sec/60:.1f}m"
                )

    # Sort summaries by DOY
    summaries = sorted(summaries, key=lambda x: x["doy"])
    summary_df = pd.DataFrame(summaries)
    summary_path = output_dir / f"{station}_{year}_daily_processing_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"[{station.upper()}] Saved daily processing summary: {summary_path.name}")

    # Consolidate annual arc features
    print(f"[{station.upper()}] Consolidating annual arc features table...")
    all_arc_dfs = []
    for doy in range(doy_start, doy_end + 1):
        f_p = output_dir / f"{station}_doy{doy:03d}_gnss_ir_arc_features.csv"
        if f_p.exists() and f_p.stat().st_size > 1000:
            all_arc_dfs.append(pd.read_csv(f_p))

    if all_arc_dfs:
        annual_df = pd.concat(all_arc_dfs, ignore_index=True)
        annual_path = output_dir / f"{station}_{year}_annual_gnss_ir_features.csv"
        annual_df.to_csv(annual_path, index=False)
        print(f"[{station.upper()}] Saved annual features table: {annual_path.name} ({len(annual_df)} arcs)")

    # Aggregate daily features
    print(f"[{station.upper()}] Computing daily aggregated mean and median datasets...")
    aggregate_station_daily_features(station, year, output_dir)

    total_time = time.time() - t0
    print(f"[{station.upper()} COMPLETE] Processed {total} days in {total_time/60.0:.2f} minutes.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract GNSS-IR features for LOSO stations.")
    parser.add_argument("--station", required=True, choices=["p037", "p043"], help="Station ID")
    parser.add_argument("--year", type=int, default=2016, help="Target year")
    parser.add_argument("--doy-start", type=int, default=1, help="Start DOY")
    parser.add_argument("--doy-end", type=int, default=366, help="End DOY")
    parser.add_argument("--workers", type=int, default=6, help="Number of parallel processes")

    args = parser.parse_args()
    process_station_batch(
        station=args.station,
        year=args.year,
        doy_start=args.doy_start,
        doy_end=args.doy_end,
        max_workers=args.workers,
    )
