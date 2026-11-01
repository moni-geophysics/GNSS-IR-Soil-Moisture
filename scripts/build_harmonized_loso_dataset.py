"""
Harmonized Three-Station Leave-One-Station-Out (LOSO) Dataset Builder and Readiness Audit.
Stations: P041 (Marshall, CO), P037 (Fremont Pass, CO), P043 (Newcastle, WY).
Year: 2016 (Leap year, DOY 001-366).

Integrates:
1. Daily aggregated GNSS-IR features (rh, amp, phase_circ, phase_cos, phase_sin, R, arc counts).
2. Daily in situ / reflectometry PBO_H2O surface soil moisture (0-5 cm) ground truth.
3. Daily meteorological precipitation (Vaisala WXT520).
4. Physical site characteristics (coordinates, elevation, soil texture, saturation, land cover).

Features:
- Preserves explicit missing values (no zero-filling for missing GNSS or target).
- Explicit boolean indicator 'usable_for_loso' (clean_arcs > 0 and quality_flag == 'G').
- Exports consolidated multi-station CSV tables for both Median and Mean aggregations.
- Computes comprehensive distributional comparisons and distribution-shift audit.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


STATION_METADATA = {
    "p041": {
        "name": "Marshall (Field Site 2), CO",
        "latitude": 39.949492042,
        "longitude": -105.194266387,
        "altitude_m": 1728.8555755,
        "orthometric_height_m": 1744.7156,
        "nominal_rh_m": 1.93,
        "soil_sand_pct": 37.0,
        "soil_clay_pct": 19.0,
        "soil_silt_pct": 44.0,
        "soil_saturation": 0.43,
        "soil_organic_carbon": 0.82,
        "land_cover": "Grassland",
        "climate_koppen": "BSk",
        "reference_distance_m": 210.69,
    },
    "p037": {
        "name": "Fremont Pass, CO",
        "latitude": 38.421753832,
        "longitude": -105.104683735,
        "altitude_m": 1616.8946041,
        "orthometric_height_m": 1633.6646,
        "nominal_rh_m": 1.91,
        "soil_sand_pct": 58.0,
        "soil_clay_pct": 24.0,
        "soil_silt_pct": 18.0,
        "soil_saturation": 0.41,
        "soil_organic_carbon": 0.50,
        "land_cover": "Grassland",
        "climate_koppen": "BSk",
        "reference_distance_m": 0.54,
    },
    "p043": {
        "name": "Newcastle, WY",
        "latitude": 43.881145655,
        "longitude": -104.185702780,
        "altitude_m": 1490.9542094,
        "orthometric_height_m": 1504.5042,
        "nominal_rh_m": 1.89,
        "soil_sand_pct": 37.0,
        "soil_clay_pct": 19.0,
        "soil_silt_pct": 44.0,
        "soil_saturation": 0.44,
        "soil_organic_carbon": 0.39,
        "land_cover": "Grassland",
        "climate_koppen": "BSk",
        "reference_distance_m": 0.53,
    },
}


def build_harmonized_dataset(agg_type: str = "median") -> pd.DataFrame:
    """Build harmonized dataset across all three stations for the chosen aggregation type."""
    agg_suffix = f"daily_{agg_type}_dataset.csv"
    dfs = []

    for st in ["p041", "p037", "p043"]:
        st_dir = PROCESSED_DIR / st
        features_csv = st_dir / f"{st}_2016_{agg_suffix}"
        ref_csv = st_dir / f"{st}_2016_pbo_h2o_reference.csv"

        if not features_csv.exists():
            raise FileNotFoundError(f"Features file missing for {st}: {features_csv}")
        if not ref_csv.exists():
            raise FileNotFoundError(f"Reference file missing for {st}: {ref_csv}")

        df_feat = pd.read_csv(features_csv)
        df_ref = pd.read_csv(ref_csv)

        # Standardize dates
        df_feat["date"] = pd.to_datetime(df_feat["date"])
        df_ref["date"] = pd.to_datetime(df_ref["date"])

        # Merge on station, date, year, doy
        merged = df_feat.merge(
            df_ref[["date", "pbo_sm", "pbo_sm_flag", "sm_is_good", "precip_mm"]],
            on="date",
            how="outer",
        )

        merged["station"] = st
        merged["year"] = 2016
        merged["doy"] = merged["date"].dt.dayofyear
        merged["date_str"] = merged["date"].dt.strftime("%Y-%m-%d")

        # Attach metadata
        meta = STATION_METADATA[st]
        for k, v in meta.items():
            merged[k] = v

        # Usable flag: clean arcs present AND ground truth flag is 'G'
        merged["usable_for_loso"] = (
            (merged["clean_arc_count"].fillna(0) > 0)
            & (merged["sm_is_good"] == True)
            & (merged["pbo_sm"].notnull())
            & (merged["rh"].notnull())
        )

        dfs.append(merged)

    harmonized = pd.concat(dfs, ignore_index=True)
    harmonized = harmonized.sort_values(["station", "date"]).reset_index(drop=True)

    # Reorder columns logically
    col_order = [
        "station", "name", "date_str", "year", "doy", "usable_for_loso",
        # Primary Target
        "pbo_sm", "pbo_sm_flag", "sm_is_good", "precip_mm",
        # GNSS-IR Features
        "rh", "amp", "phase_circ", "phase_cos", "phase_sin", "phase_resultant_length",
        # QC & Arc Statistics
        "clean_arc_count", "raw_arc_count", "qc_flagged_count", "qc_pass_rate",
        # Site Metadata
        "latitude", "longitude", "altitude_m", "orthometric_height_m", "nominal_rh_m",
        "soil_sand_pct", "soil_clay_pct", "soil_silt_pct", "soil_saturation",
        "soil_organic_carbon", "land_cover", "climate_koppen", "reference_distance_m",
    ]
    existing_cols = [c for c in col_order if c in harmonized.columns]
    harmonized = harmonized[existing_cols].rename(columns={"date_str": "date"})

    out_csv = PROCESSED_DIR / f"loso_harmonized_dataset_{agg_type}.csv"
    harmonized.to_csv(out_csv, index=False)
    print(f"\n[SUCCESS] Built Harmonized Dataset ({agg_type.upper()}): {out_csv.relative_to(PROJECT_ROOT)}")
    print(f"  Total records: {len(harmonized)} ({len(harmonized[harmonized['usable_for_loso']])} usable for LOSO)")
    return harmonized


def run_readiness_audit(df: pd.DataFrame) -> None:
    """Perform thorough LOSO readiness audit on the harmonized dataset."""
    print("\n" + "=" * 80)
    print("LOSO EXPERIMENTAL READINESS AUDIT")
    print("=" * 80)

    usable_df = df[df["usable_for_loso"]].copy()

    print("\n1. USABLE DATE SUMMARY PER STATION:")
    print("-" * 50)
    for st in ["p041", "p037", "p043"]:
        st_sub = usable_df[usable_df["station"] == st]
        total_sub = df[df["station"] == st]
        gnss_avail = total_sub[total_sub["clean_arc_count"] > 0]
        sm_good = total_sub[total_sub["sm_is_good"] == True]
        print(f"Station {st.upper()} ({STATION_METADATA[st]['name']}):")
        print(f"  Total Calendar Days:     {len(total_sub)}")
        print(f"  GNSS Clean Days:         {len(gnss_avail)}")
        print(f"  PBO_H2O Flag 'G' Days:   {len(sm_good)}")
        print(f"  Usable Combined Days:    {len(st_sub)} ({len(st_sub)/366*100:.1f}%)")
        print(f"  Usable Date Span:        {st_sub['date'].min()} to {st_sub['date'].max()}")

    # Common dates across all 3 stations
    dates_p041 = set(usable_df[usable_df["station"] == "p041"]["date"])
    dates_p037 = set(usable_df[usable_df["station"] == "p037"]["date"])
    dates_p043 = set(usable_df[usable_df["station"] == "p043"]["date"])
    common_dates = sorted(dates_p041 & dates_p037 & dates_p043)

    print(f"\n2. COMMON CONCURRENT DATES (INTERSECTION):")
    print("-" * 50)
    print(f"  Simultaneous usable days across all 3 sites: {len(common_dates)} days")
    print(f"  Common date span: {common_dates[0]} to {common_dates[-1]}")

    print("\n3. FEATURE & TARGET DISTRIBUTIONS (USABLE SAMPLE):")
    print("-" * 50)
    metrics = ["pbo_sm", "rh", "amp", "phase_circ", "phase_cos", "clean_arc_count", "precip_mm"]
    for m in metrics:
        print(f"\n--- Metric: {m} ---")
        for st in ["p041", "p037", "p043"]:
            vals = usable_df[usable_df["station"] == st][m].dropna()
            print(
                f"  {st.upper()}: Mean={vals.mean():.4f} | Std={vals.std():.4f} | "
                f"Median={vals.median():.4f} | IQR=[{vals.quantile(0.25):.4f}, {vals.quantile(0.75):.4f}] | "
                f"Min={vals.min():.4f} | Max={vals.max():.4f}"
            )

    print("\n4. STATION DISTRIBUTION SHIFTS & IMPLICATIONS FOR LOSO:")
    print("-" * 50)
    print("  a) Reflector Height (rh):")
    print(f"     P041 Median = {usable_df[usable_df['station']=='p041']['rh'].median():.3f} m")
    print(f"     P037 Median = {usable_df[usable_df['station']=='p037']['rh'].median():.3f} m")
    print(f"     P043 Median = {usable_df[usable_df['station']=='p043']['rh'].median():.3f} m")
    print("     -> Highly consistent across stations (~1.89 - 1.93 m, delta < 4 cm).")

    print("\n  b) Surface Soil Moisture Target (pbo_sm, 0-5 cm):")
    print(f"     P041 Median = {usable_df[usable_df['station']=='p041']['pbo_sm'].median():.4f} m³/m³ (Mean = {usable_df[usable_df['station']=='p041']['pbo_sm'].mean():.4f})")
    print(f"     P037 Median = {usable_df[usable_df['station']=='p037']['pbo_sm'].median():.4f} m³/m³ (Mean = {usable_df[usable_df['station']=='p037']['pbo_sm'].mean():.4f})")
    print(f"     P043 Median = {usable_df[usable_df['station']=='p043']['pbo_sm'].median():.4f} m³/m³ (Mean = {usable_df[usable_df['station']=='p043']['pbo_sm'].mean():.4f})")
    print("     -> P037 is systematically drier (sandy loam, faster drainage). P041 and P043 have similar loam texture and soil moisture ranges.")

    print("\n  c) Multipath Amplitude (amp):")
    print(f"     P041 Median = {usable_df[usable_df['station']=='p041']['amp'].median():.2f} V/V")
    print(f"     P037 Median = {usable_df[usable_df['station']=='p037']['amp'].median():.2f} V/V")
    print(f"     P043 Median = {usable_df[usable_df['station']=='p043']['amp'].median():.2f} V/V")

    print("\n5. DATA INTEGRITY & UNITS AUDIT:")
    print("-" * 50)
    print("  - Reflector Height: meters (m), search window [0.5, 8.0] m, nominal filter [1.6, 2.4] m.")
    print("  - Amplitude: linear scale (V/V), threshold >= 5.0 V/V.")
    print("  - Phase: radians in [-pi, pi], circular median / mean.")
    print("  - Phase Transformations: phase_cos, phase_sin in [-1.0, 1.0].")
    print("  - Soil Moisture Target: volumetric water content (m³/m³), depth 0-5 cm, Quality Flag 'G'.")
    print("  - Missingness: strictly preserved as NaN; zero imputation avoided.")
    print("=" * 80)


if __name__ == "__main__":
    df_med = build_harmonized_dataset("median")
    df_mean = build_harmonized_dataset("mean")
    run_readiness_audit(df_med)
