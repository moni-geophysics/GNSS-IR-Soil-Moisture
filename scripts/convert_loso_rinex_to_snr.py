"""
Convert downloaded 2016 RINEX observations (.16d.Z) to SNR66 format.

Stations targeted:
  - P037 (FreemontAPCO2004)
  - P043 (NewcastleWY2005)

Requirements:
  - Preserves existing files (e.g. data/raw/gnss/2016/snr/p037/p0370010.16.snr66).
  - Skips already converted days.
  - Converts Compact RINEX (.16d.Z) to RINEX (.16o) via CRX2RNX, then calls rinex2snr.
  - Matches established P041 conventions (DOY 1-131 precise gbm, DOY 132-366 broadcast nav/gps).
  - Generates comprehensive conversion manifest with MD5 and size metrics.
  - Keeps compressed RINEX and converted SNR files strictly separated.
  - Multi-worker concurrent processing for speed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_GNSS_DIR = PROJECT_ROOT / "data" / "raw" / "gnss"
CONVERSION_MANIFEST_PATH = RAW_GNSS_DIR / "2016" / "loso_conversion_manifest.json"
LOG_DIR = PROJECT_ROOT / "logs"
LOG_FILE = LOG_DIR / "convert_loso_snr_2016.log"


def setup_logger() -> logging.Logger:
    """Set up logger for SNR conversion."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("loso_converter")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh_fmt = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s")
    fh.setFormatter(fh_fmt)
    logger.addHandler(fh)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch_fmt = logging.Formatter("%(message)s")
    ch.setFormatter(ch_fmt)
    logger.addHandler(ch)

    return logger


def calculate_md5(file_path: Path) -> str:
    """Calculate MD5 checksum."""
    h = hashlib.md5()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def convert_day(
    station: str,
    year: int,
    doy: int,
    rinex_file: Path,
    snr_dir: Path,
    logger: logging.Logger,
) -> Dict[str, Any]:
    """Convert a single daily RINEX file to SNR66 format via WSL gnssrefl."""
    station = station.lower()
    yy = str(year)[-2:]
    doy_str = f"{doy:03d}"

    raw_snr_name = f"{station}{doy_str}0.{yy}.snr66"
    gz_snr_name = f"{station}{doy_str}0.{yy}.snr66.gz"

    raw_snr_path = snr_dir / raw_snr_name
    gz_snr_path = snr_dir / gz_snr_name

    record: Dict[str, Any] = {
        "station": station,
        "year": year,
        "doy": doy,
        "status": "pending",
        "output_file": None,
        "size_bytes": 0,
        "md5": None,
        "orbit_used": None,
        "error": None,
    }

    # 1. Preservation check: if uncompressed or gzipped SNR already exists
    if raw_snr_path.exists() and raw_snr_path.stat().st_size > 10000:
        record["status"] = "already_exists_raw"
        record["output_file"] = str(raw_snr_path.relative_to(PROJECT_ROOT))
        record["size_bytes"] = raw_snr_path.stat().st_size
        record["md5"] = calculate_md5(raw_snr_path)
        record["orbit_used"] = "preserved_original"
        logger.info(f"[{station.upper()} DOY {doy_str}] Preserved existing SNR: {raw_snr_name} ({record['size_bytes'] / 1024:.1f} KB)")
        return record

    if gz_snr_path.exists() and gz_snr_path.stat().st_size > 10000:
        record["status"] = "already_exists_gz"
        record["output_file"] = str(gz_snr_path.relative_to(PROJECT_ROOT))
        record["size_bytes"] = gz_snr_path.stat().st_size
        record["md5"] = calculate_md5(gz_snr_path)
        logger.info(f"[{station.upper()} DOY {doy_str}] Existing compressed SNR: {gz_snr_name} ({record['size_bytes'] / 1024:.1f} KB)")
        return record

    if not rinex_file.exists():
        record["status"] = "missing_input_rinex"
        record["error"] = f"RINEX file {rinex_file.name} not found"
        logger.warning(f"[{station.upper()} DOY {doy_str}] RINEX file missing: {rinex_file.name}")
        return record

    # Build WSL conversion paths
    win_rinex_path = str(rinex_file.resolve()).replace("\\", "/")
    drive_letter = win_rinex_path[0].lower()
    wsl_rinex_path = f"/mnt/{drive_letter}{win_rinex_path[2:]}"

    win_gnss_root = str(RAW_GNSS_DIR.resolve()).replace("\\", "/")
    wsl_gnss_root = f"/mnt/{drive_letter}{win_gnss_root[2:]}"

    tools_crx2rnx = str((PROJECT_ROOT / "tools" / "gnssrefl" / "CRX2RNX").resolve()).replace("\\", "/")
    wsl_crx2rnx = f"/mnt/{drive_letter}{tools_crx2rnx[2:]}"

    # Orbit strategy: DOY 1-131 check gbm sp3 existence, else gps; DOY 132-366 gps
    # Note: If DOY <= 131, try gbm first; if fails or not found, fall back to gps
    primary_orb = "gbm" if doy <= 131 else "gps"

    def run_conv(orb_opt: str) -> subprocess.CompletedProcess:
        bash_cmd = f"""
export PATH="/home/moni/gnssrefl-env/bin:/home/moni/gnssrefl-exe:$PATH"
export EXE=/home/moni/gnssrefl-exe
export REFL_CODE="{wsl_gnss_root}"
TMPDIR="/tmp/conv_{station}_{doy_str}_{orb_opt}"
mkdir -p "$TMPDIR"
cd "$TMPDIR"
cp "{wsl_rinex_path}" ./
gzip -dc "{rinex_file.name}" > "{station}{doy_str}0.{yy}d" 2>/dev/null
"{wsl_crx2rnx}" -f "{station}{doy_str}0.{yy}d" >/dev/null 2>&1
rinex2snr -input_file "{station}{doy_str}0.{yy}o" -orb {orb_opt} >/dev/null 2>&1
cd /tmp
rm -rf "$TMPDIR"
"""
        return subprocess.run(
            ["wsl", "bash", "-c", bash_cmd],
            capture_output=True,
            text=True,
            timeout=180,
        )

    try:
        try:
            proc = run_conv(primary_orb)
            used_orb = primary_orb
        except subprocess.TimeoutExpired:
            logger.warning(f"[{station.upper()} DOY {doy_str}] {primary_orb} timed out. Falling back to gps nav...")
            proc = run_conv("gps")
            used_orb = "gps_fallback"

        # If primary (gbm) failed to produce file, fall back to gps
        if not (gz_snr_path.exists() and gz_snr_path.stat().st_size > 10000) and not (raw_snr_path.exists() and raw_snr_path.stat().st_size > 10000):
            if primary_orb != "gps":
                logger.info(f"[{station.upper()} DOY {doy_str}] gbm orbit produced no output. Falling back to gps nav...")
                proc = run_conv("gps")
                used_orb = "gps_fallback"

        if gz_snr_path.exists() and gz_snr_path.stat().st_size > 10000:
            record["status"] = "converted_gz"
            record["output_file"] = str(gz_snr_path.relative_to(PROJECT_ROOT))
            record["size_bytes"] = gz_snr_path.stat().st_size
            record["md5"] = calculate_md5(gz_snr_path)
            record["orbit_used"] = used_orb
            logger.info(f"[{station.upper()} DOY {doy_str}] OK converted ({used_orb}): {gz_snr_name} ({record['size_bytes'] / 1024:.1f} KB)")
            return record

        if raw_snr_path.exists() and raw_snr_path.stat().st_size > 10000:
            record["status"] = "converted_raw"
            record["output_file"] = str(raw_snr_path.relative_to(PROJECT_ROOT))
            record["size_bytes"] = raw_snr_path.stat().st_size
            record["md5"] = calculate_md5(raw_snr_path)
            record["orbit_used"] = used_orb
            logger.info(f"[{station.upper()} DOY {doy_str}] OK converted ({used_orb}): {raw_snr_name} ({record['size_bytes'] / 1024:.1f} KB)")
            return record

        record["status"] = "conversion_failed"
        record["error"] = f"Return code {proc.returncode}. Stderr: {proc.stderr[:200]}"
        logger.error(f"[{station.upper()} DOY {doy_str}] FAILED conversion: {record['error']}")

    except Exception as e:
        record["status"] = "conversion_exception"
        record["error"] = f"{type(e).__name__}: {str(e)}"
        logger.error(f"[{station.upper()} DOY {doy_str}] Exception during conversion: {record['error']}")

    return record


def run_batch_conversion(
    stations: List[str],
    year: int = 2016,
    doy_start: int = 1,
    doy_end: int = 366,
    max_workers: int = 4,
) -> Dict[str, Any]:
    """Execute concurrent conversion across stations and days of year."""
    logger = setup_logger()
    logger.info("=================================================================")
    logger.info(f"Starting RINEX -> SNR66 Conversion: Year {year} (DOY {doy_start:03d} to {doy_end:03d})")
    logger.info(f"Target Stations: {[s.upper() for s in stations]} | Concurrency: {max_workers} workers")
    logger.info("=================================================================")

    manifest_entries: List[Dict[str, Any]] = []

    for station in stations:
        station_rinex_dir = RAW_GNSS_DIR / str(year) / "rinex" / station.lower()
        station_snr_dir = RAW_GNSS_DIR / str(year) / "snr" / station.lower()
        station_snr_dir.mkdir(parents=True, exist_ok=True)

        yy = str(year)[-2:]
        tasks = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for doy in range(doy_start, doy_end + 1):
                doy_str = f"{doy:03d}"
                rinex_file = station_rinex_dir / f"{station.lower()}{doy_str}0.{yy}d.Z"
                tasks.append(
                    executor.submit(
                        convert_day,
                        station,
                        year,
                        doy,
                        rinex_file,
                        station_snr_dir,
                        logger,
                    )
                )

            for future in as_completed(tasks):
                res = future.result()
                manifest_entries.append(res)

    CONVERSION_MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary_report: Dict[str, Any] = {
        "year": year,
        "stations": stations,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "total_processed": len(manifest_entries),
        "results": manifest_entries,
    }

    with open(CONVERSION_MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(summary_report, f, indent=2)

    logger.info(f"Conversion manifest written to {CONVERSION_MANIFEST_PATH.relative_to(PROJECT_ROOT)}")
    return summary_report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert LOSO GNSS RINEX files to SNR66.")
    parser.add_argument("--stations", nargs="+", default=["p037", "p043"], help="Station IDs to convert")
    parser.add_argument("--year", type=int, default=2016, help="Target year")
    parser.add_argument("--doy-start", type=int, default=1, help="Start DOY (1)")
    parser.add_argument("--doy-end", type=int, default=366, help="End DOY (366)")
    parser.add_argument("--workers", type=int, default=4, help="Concurrent workers")

    args = parser.parse_args()
    run_batch_conversion(
        stations=args.stations,
        year=args.year,
        doy_start=args.doy_start,
        doy_end=args.doy_end,
        max_workers=args.workers,
    )
