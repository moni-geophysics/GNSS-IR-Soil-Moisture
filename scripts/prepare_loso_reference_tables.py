"""
PBO_H2O Reference Alignment and Quality Audit for Three-Station LOSO Experiment.
Processes and aligns ISMN PBO_H2O surface soil moisture (0-5 cm) and Vaisala precipitation
for stations P041 (MARSHALL2), P037 (FREEMONTAP), and P043 (NEWCASTLE-1) for Year 2016.

Features:
- Preserves raw ISMN data completely unchanged.
- Parses .stm files into standardized, verified daily tables.
- Implements strict quality-flag segregation (Flag 'G' for primary benchmark).
- Verifies common-date overlap across all 3 stations.
- Calculates distributional statistics and flag counts.
- Exports individual reference tables and a consolidated reference alignment table.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_ISMN_DIR = PROJECT_ROOT / "data" / "raw" / "ismn"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def parse_ismn_stm(file_path: Path) -> pd.DataFrame:
    """Parse an ISMN ASCII format .stm file."""
    if not file_path.exists():
        raise FileNotFoundError(f"ISMN file not found: {file_path}")

    records = []
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        header_line = f.readline().strip()
        header_parts = header_line.split()

        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4:
                # Format: YYYY/MM/DD HH:MM value quality_flag [provider_flag]
                dt_str = f"{parts[0]} {parts[1]}"
                val_str = parts[2]
                flag = parts[3]
                provider_flag = parts[4] if len(parts) > 4 else None

                try:
                    val = float(val_str)
                except ValueError:
                    val = np.nan

                records.append({
                    "datetime_utc": dt_str,
                    "date": parts[0].replace("/", "-"),
                    "value": val,
                    "quality_flag": flag,
                    "provider_flag": provider_flag,
                })

    df = pd.DataFrame(records)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
    return df


def build_station_reference(
    station_id: str,
    site_folder_name: str,
    station_subfolder: str,
) -> pd.DataFrame:
    """Load and align daily soil moisture and precipitation for one station."""
    st_dir = RAW_ISMN_DIR / site_folder_name / "PBO_H2O" / station_subfolder
    if not st_dir.exists():
        raise FileNotFoundError(f"Station directory not found: {st_dir}")

    sm_files = list(st_dir.glob("*sm*0.000000_0.050000*.stm"))
    p_files = list(st_dir.glob("*p_-2.000000_-2.000000*.stm"))

    if not sm_files:
        raise FileNotFoundError(f"Soil moisture .stm file missing in {st_dir}")
    if not p_files:
        raise FileNotFoundError(f"Precipitation .stm file missing in {st_dir}")

    df_sm = parse_ismn_stm(sm_files[0])
    df_p = parse_ismn_stm(p_files[0])

    # Filter to 2016
    df_sm_2016 = df_sm[(df_sm["date"].dt.year == 2016)].copy()
    df_p_2016 = df_p[(df_p["date"].dt.year == 2016)].copy()

    # Reindex to full 366-day calendar
    full_dates = pd.date_range("2016-01-01", "2016-12-31", freq="D")
    base_df = pd.DataFrame({
        "date": full_dates,
        "station": station_id.lower(),
        "year": 2016,
        "doy": full_dates.dayofyear,
    })

    # Merge SM
    df_sm_sub = df_sm_2016[["date", "value", "quality_flag"]].rename(
        columns={"value": "pbo_sm", "quality_flag": "pbo_sm_flag"}
    )
    merged = base_df.merge(df_sm_sub, on="date", how="left")

    # Merge Precip
    df_p_sub = df_p_2016[["date", "value", "quality_flag"]].rename(
        columns={"value": "precip_mm", "quality_flag": "precip_flag"}
    )
    merged = merged.merge(df_p_sub, on="date", how="left")

    merged["date_str"] = merged["date"].dt.strftime("%Y-%m-%d")

    # Flag indicator: good quality mask
    merged["sm_is_good"] = merged["pbo_sm_flag"] == "G"
    merged["sm_good_value"] = np.where(merged["sm_is_good"], merged["pbo_sm"], np.nan)

    return merged


def run_reference_audit() -> Dict[str, Any]:
    print("=" * 80)
    print("PBO_H2O REFERENCE DATA AUDIT AND ALIGNMENT (2016)")
    print("=" * 80)

    stations_meta = {
        "p041": {"folder": "p041", "sub": "MARSHALL2", "name": "Marshall (Field Site 2), CO"},
        "p037": {"folder": "p037", "sub": "FREEMONTAP", "name": "Fremont Pass, CO"},
        "p043": {"folder": "p043", "sub": "NEWCASTLE-1", "name": "Newcastle, WY"},
    }

    dfs = {}
    stats = {}

    for st, meta in stations_meta.items():
        df = build_station_reference(st, meta["folder"], meta["sub"])
        dfs[st] = df

        # Save station-specific table
        st_out_dir = PROCESSED_DIR / st
        st_out_dir.mkdir(parents=True, exist_ok=True)
        out_csv = st_out_dir / f"{st}_2016_pbo_h2o_reference.csv"
        df.to_csv(out_csv, index=False)
        print(f"\n[{st.upper()} - {meta['name']}]")
        print(f"  Saved reference table: {out_csv.relative_to(PROJECT_ROOT)}")

        # Audit metrics
        total_days = len(df)
        observed_days = df["pbo_sm"].notnull().sum()
        good_days = df["sm_is_good"].sum()
        missing_days = total_days - observed_days
        flag_dist = df["pbo_sm_flag"].value_counts(dropna=False).to_dict()

        good_vals = df[df["sm_is_good"]]["pbo_sm"]

        st_stats = {
            "station": st,
            "name": meta["name"],
            "total_days": int(total_days),
            "observed_days": int(observed_days),
            "missing_days": int(missing_days),
            "good_quality_days": int(good_days),
            "flags": flag_dist,
            "mean": float(good_vals.mean()) if not good_vals.empty else np.nan,
            "std": float(good_vals.std()) if not good_vals.empty else np.nan,
            "median": float(good_vals.median()) if not good_vals.empty else np.nan,
            "min": float(good_vals.min()) if not good_vals.empty else np.nan,
            "max": float(good_vals.max()) if not good_vals.empty else np.nan,
            "q25": float(good_vals.quantile(0.25)) if not good_vals.empty else np.nan,
            "q75": float(good_vals.quantile(0.75)) if not good_vals.empty else np.nan,
        }
        stats[st] = st_stats

        print(f"  Observed Days: {observed_days}/366 | Missing: {missing_days}")
        print(f"  Good Quality (Flag G): {good_days} days ({good_days / 366 * 100:.1f}%)")
        print(f"  Quality Flag Breakdown: {flag_dist}")
        print(f"  Good SM Distribution: Mean={st_stats['mean']:.4f}, Median={st_stats['median']:.4f}, Std={st_stats['std']:.4f}, Range=[{st_stats['min']:.4f}, {st_stats['max']:.4f}]")

    # Common dates intersection
    good_dates_p041 = set(dfs["p041"][dfs["p041"]["sm_is_good"]]["date_str"])
    good_dates_p037 = set(dfs["p037"][dfs["p037"]["sm_is_good"]]["date_str"])
    good_dates_p043 = set(dfs["p043"][dfs["p043"]["sm_is_good"]]["date_str"])

    common_3st = sorted(good_dates_p041 & good_dates_p037 & good_dates_p043)
    n_common = len(common_3st)

    print("\n" + "=" * 80)
    print("COMMON USABLE DATE INTERSECTION (FLAG 'G')")
    print("=" * 80)
    print(f"P041 Good Dates: {len(good_dates_p041)}")
    print(f"P037 Good Dates: {len(good_dates_p037)}")
    print(f"P043 Good Dates: {len(good_dates_p043)}")
    print(f"\nINTERSECTION (Common across all 3 stations): {n_common} dates")
    print(f"Date range of common dates: {common_3st[0]} to {common_3st[-1]}")

    # Build consolidated reference table across all three stations
    aligned_rows = []
    for d in pd.date_range("2016-01-01", "2016-12-31", freq="D"):
        d_str = d.strftime("%Y-%m-%d")
        doy = d.dayofyear

        row = {
            "date": d_str,
            "year": 2016,
            "doy": doy,
            "in_3station_common_set": d_str in common_3st,
        }

        for st in ["p041", "p037", "p043"]:
            st_sub = dfs[st][dfs[st]["date_str"] == d_str]
            if not st_sub.empty:
                r = st_sub.iloc[0]
                row[f"{st}_pbo_sm"] = r["pbo_sm"]
                row[f"{st}_sm_flag"] = r["pbo_sm_flag"]
                row[f"{st}_sm_is_good"] = r["sm_is_good"]
                row[f"{st}_precip_mm"] = r["precip_mm"]
            else:
                row[f"{st}_pbo_sm"] = np.nan
                row[f"{st}_sm_flag"] = np.nan
                row[f"{st}_sm_is_good"] = False
                row[f"{st}_precip_mm"] = np.nan

        aligned_rows.append(row)

    df_aligned = pd.DataFrame(aligned_rows)
    aligned_csv = PROCESSED_DIR / "loso_three_station_pbo_h2o_reference_aligned.csv"
    df_aligned.to_csv(aligned_csv, index=False)
    print(f"\nSaved consolidated three-station reference table: {aligned_csv.relative_to(PROJECT_ROOT)}")

    return {
        "stats": stats,
        "n_common": n_common,
        "common_dates": common_3st,
    }


if __name__ == "__main__":
    run_reference_audit()
