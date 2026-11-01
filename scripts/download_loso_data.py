"""
Download and manage 2016 GNSS observation files for LOSO experiment stations.

Stations targeted:
  - P037 (FreemontAPCO2004, Fremont County Airport, CO)
  - P043 (NewcastleWY2005, Newcastle, Weston County, WY)

Features:
  - Secure EarthScope SSO OAuth2 authentication (no credentials leaked in logs).
  - Resumable: skips files already downloaded and valid (> 10 KB).
  - Thread-pool concurrency for robust and rapid downloading.
  - Generates comprehensive manifest recording MD5, file sizes, and status.
  - Preserves existing data (e.g., P037 DOY 001 SNR file).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
from earthscope_sdk.auth.device_code_flow import DeviceCodeFlowSimple


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_GNSS_DIR = PROJECT_ROOT / "data" / "raw" / "gnss"
TOKENS_PATH = RAW_GNSS_DIR / "sso_tokens.json"
MANIFEST_PATH = RAW_GNSS_DIR / "2016" / "loso_download_manifest.json"
LOG_DIR = PROJECT_ROOT / "logs"
LOG_FILE = LOG_DIR / "download_loso_2016.log"

EARTHSCOPE_BASE_URL = "https://data.earthscope.org/archive/gnss/rinex/obs"


def setup_logger() -> logging.Logger:
    """Set up file and console logger without exposing sensitive tokens."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("loso_downloader")
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
    """Calculate MD5 checksum of a file."""
    h = hashlib.md5()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


class EarthScopeAuthManager:
    """Thread-safe manager for refreshing and providing EarthScope bearer tokens."""

    def __init__(self, token_dir: Path, logger: logging.Logger):
        self.token_dir = token_dir
        self.logger = logger
        self.flow = DeviceCodeFlowSimple(self.token_dir)
        self._refresh_token()

    def _refresh_token(self) -> None:
        try:
            self.flow.load_tokens()
            self.flow.refresh()
            self.logger.info("EarthScope authentication token refreshed successfully.")
        except Exception as e:
            self.logger.error(f"Failed to refresh EarthScope tokens: {type(e).__name__}: {e}")
            raise

    def get_token(self) -> str:
        # Refresh if nearing expiration or expired
        ttl_seconds = self.flow.ttl.total_seconds() if hasattr(self.flow.ttl, "total_seconds") else float(self.flow.ttl)
        if ttl_seconds < 300:
            self._refresh_token()
        return self.flow.access_token

    def force_refresh(self) -> str:
        self._refresh_token()
        return self.flow.access_token


def download_single_day(
    station: str,
    year: int,
    doy: int,
    auth_mgr: EarthScopeAuthManager,
    dest_dir: Path,
    logger: logging.Logger,
    max_retries: int = 3,
) -> Dict[str, Any]:
    """Download a single daily RINEX file (.16d.Z) for a station and day of year."""
    station = station.lower()
    yy = str(year)[-2:]
    doy_str = f"{doy:03d}"
    filename = f"{station}{doy_str}0.{yy}d.Z"
    target_path = dest_dir / filename
    temp_path = dest_dir / f"{filename}.tmp"

    record: Dict[str, Any] = {
        "station": station,
        "year": year,
        "doy": doy,
        "filename": filename,
        "path": str(target_path.relative_to(PROJECT_ROOT)),
        "status": "pending",
        "http_code": None,
        "size_bytes": 0,
        "md5": None,
        "retries": 0,
        "error": None,
    }

    # 1. Check if valid file already exists
    if target_path.exists() and target_path.stat().st_size > 10000:
        record["status"] = "already_exists"
        record["size_bytes"] = target_path.stat().st_size
        record["md5"] = calculate_md5(target_path)
        record["http_code"] = 200
        return record

    url = f"{EARTHSCOPE_BASE_URL}/{year}/{doy_str}/{filename}"

    for attempt in range(1, max_retries + 1):
        record["retries"] = attempt - 1
        try:
            token = auth_mgr.get_token()
            headers = {"Authorization": f"Bearer {token}"}

            with requests.get(url, headers=headers, stream=True, timeout=30) as r:
                record["http_code"] = r.status_code

                if r.status_code == 401:
                    logger.warning(f"[{station.upper()} DOY {doy_str}] 401 Unauthorized. Refreshing token...")
                    auth_mgr.force_refresh()
                    continue

                if r.status_code == 404:
                    record["status"] = "not_found"
                    record["error"] = "HTTP 404 Not Found"
                    logger.warning(f"[{station.upper()} DOY {doy_str}] File not found on EarthScope (404).")
                    return record

                if r.status_code == 200:
                    with open(temp_path, "wb") as f:
                        for chunk in r.iter_content(chunk_size=32768):
                            if chunk:
                                f.write(chunk)

                    file_size = temp_path.stat().st_size
                    if file_size < 10000:
                        temp_path.unlink(missing_ok=True)
                        raise ValueError(f"Downloaded file suspiciously small ({file_size} bytes)")

                    # Atomically move temp file to target
                    temp_path.replace(target_path)
                    record["status"] = "downloaded"
                    record["size_bytes"] = target_path.stat().st_size
                    record["md5"] = calculate_md5(target_path)
                    logger.info(f"[{station.upper()} DOY {doy_str}] OK: {target_path.name} ({record['size_bytes'] / 1024:.1f} KB)")
                    return record

                logger.warning(f"[{station.upper()} DOY {doy_str}] Attempt {attempt}/{max_retries} returned HTTP {r.status_code}")

        except Exception as e:
            record["error"] = f"{type(e).__name__}: {str(e)}"
            logger.warning(f"[{station.upper()} DOY {doy_str}] Attempt {attempt}/{max_retries} error: {record['error']}")
            if temp_path.exists():
                temp_path.unlink(missing_ok=True)

        if attempt < max_retries:
            time.sleep(2.0 * attempt)

    record["status"] = "failed"
    logger.error(f"[{station.upper()} DOY {doy_str}] FAILED after {max_retries} attempts.")
    return record


def run_batch_download(
    stations: List[str],
    year: int = 2016,
    doy_start: int = 1,
    doy_end: int = 366,
    max_workers: int = 4,
) -> Dict[str, Any]:
    """Execute concurrent batch download across stations and days of year."""
    logger = setup_logger()
    logger.info("=================================================================")
    logger.info(f"Starting LOSO GNSS Observation Download: Year {year} (DOY {doy_start:03d} to {doy_end:03d})")
    logger.info(f"Target Stations: {[s.upper() for s in stations]}")
    logger.info("=================================================================")

    auth_mgr = EarthScopeAuthManager(RAW_GNSS_DIR, logger)

    manifest_entries: List[Dict[str, Any]] = []

    for station in stations:
        station_rinex_dir = RAW_GNSS_DIR / str(year) / "rinex" / station.lower()
        station_rinex_dir.mkdir(parents=True, exist_ok=True)

        tasks = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for doy in range(doy_start, doy_end + 1):
                tasks.append(
                    executor.submit(
                        download_single_day,
                        station,
                        year,
                        doy,
                        auth_mgr,
                        station_rinex_dir,
                        logger,
                    )
                )

            for future in as_completed(tasks):
                res = future.result()
                manifest_entries.append(res)

    # Save manifest
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary_report: Dict[str, Any] = {
        "year": year,
        "stations": stations,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "total_requested": len(manifest_entries),
        "results": manifest_entries,
    }

    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(summary_report, f, indent=2)

    logger.info(f"Manifest written to {MANIFEST_PATH.relative_to(PROJECT_ROOT)}")
    return summary_report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download LOSO GNSS observations from EarthScope.")
    parser.add_argument("--stations", nargs="+", default=["p037", "p043"], help="Station IDs to download")
    parser.add_argument("--year", type=int, default=2016, help="Target year")
    parser.add_argument("--doy-start", type=int, default=1, help="Start DOY (1)")
    parser.add_argument("--doy-end", type=int, default=366, help="End DOY (366 for leap year 2016)")
    parser.add_argument("--workers", type=int, default=4, help="Concurrent workers")

    args = parser.parse_args()
    run_batch_download(
        stations=args.stations,
        year=args.year,
        doy_start=args.doy_start,
        doy_end=args.doy_end,
        max_workers=args.workers,
    )
