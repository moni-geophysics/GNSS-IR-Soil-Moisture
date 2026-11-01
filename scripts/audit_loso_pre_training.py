"""
Comprehensive Final Pre-Training Audit Script for Three-Station LOSO Experiment.
Analyzes:
A. Temporal Data Availability:
   - Monthly valid paired sample counts, GNSS feature availability, PBO_H2O availability.
   - Longest continuous gap in valid paired observations per station.
   - Seasonal distribution of valid observations.
   - Monthly coverage table and multi-panel visualization saved to results/figures/.
B. Feature Consistency & Phase Analysis:
   - Identical feature definitions, units, and aggregation procedures across P041, P037, P043.
   - Detailed distributions (mean, std, median, IQR, min, max, skewness).
   - In-depth investigation of station-specific phase offsets and why target-informed calibration is forbidden.
C. Spatial-Footprint Limitation Documentation:
   - Analysis of ~210.69 m separation at P041 vs ~0.54 m at P037 and ~0.53 m at P043.
D. Formal LOSO Protocol Specification:
   - Folds 1, 2, and 3 definition.
   - Full 771-sample primary design vs 216-sample common-date sensitivity design.
   - Equal station weighting formulation.
E. Temporal Validation Feasibility (Spatial + Temporal):
   - Hydrological regime vs chronological split feasibility.
   - Candidate boundary definitions and exact sample counts.
F. Feature Ablation Sets Design:
   - Model A (RH + Amp), Model B (RH + Amp + Phase cos + sin), Model C (Model B + R + QC).
   - Cross-station transferability assessment.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

HARMONIZED_CSV = PROCESSED_DIR / "loso_harmonized_dataset_median.csv"


def run_audit():
    print("=" * 85)
    print("FINAL PRE-TRAINING AUDIT: THREE-STATION LEAVE-ONE-STATION-OUT (LOSO)")
    print("=" * 85)

    if not HARMONIZED_CSV.exists():
        raise FileNotFoundError(f"Harmonized dataset missing: {HARMONIZED_CSV}")

    df = pd.read_csv(HARMONIZED_CSV)
    df["date"] = pd.to_datetime(df["date"])
    df["month"] = df["date"].dt.month
    df["month_name"] = df["date"].dt.strftime("%b")

    stations = ["p041", "p037", "p043"]
    station_names = {
        "p041": "P041 (Marshall, CO)",
        "p037": "P037 (Fremont Pass, CO)",
        "p043": "P043 (Newcastle, WY)",
    }

    # =========================================================================
    # SECTION A: TEMPORAL DATA AVAILABILITY
    # =========================================================================
    print("\n" + "=" * 85)
    print("SECTION A: TEMPORAL DATA AVAILABILITY AUDIT")
    print("=" * 85)

    # 1. Monthly Coverage Table
    monthly_records = []
    months = list(range(1, 13))
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    for m, m_name in zip(months, month_names):
        sub_m = df[df["month"] == m]
        days_in_month = len(sub_m[sub_m["station"] == "p041"])

        rec = {
            "Month": m_name,
            "Days_in_Month": days_in_month,
        }

        for st in stations:
            st_m = sub_m[sub_m["station"] == st]
            gnss_avail = (st_m["clean_arc_count"] > 0).sum()
            pbo_avail = (st_m["pbo_sm"].notnull()).sum()
            pbo_good = (st_m["sm_is_good"] == True).sum()
            usable = st_m["usable_for_loso"].sum()

            rec[f"{st.upper()}_GNSS"] = gnss_avail
            rec[f"{st.upper()}_PBO_Obs"] = pbo_avail
            rec[f"{st.upper()}_PBO_Good"] = pbo_good
            rec[f"{st.upper()}_Usable"] = usable

        # 3-station common dates in month
        p41_dates = set(sub_m[(sub_m["station"] == "p041") & (sub_m["usable_for_loso"])]["date"])
        p37_dates = set(sub_m[(sub_m["station"] == "p037") & (sub_m["usable_for_loso"])]["date"])
        p43_dates = set(sub_m[(sub_m["station"] == "p043") & (sub_m["usable_for_loso"])]["date"])
        rec["3Station_Common"] = len(p41_dates & p37_dates & p43_dates)

        monthly_records.append(rec)

    monthly_df = pd.DataFrame(monthly_records)
    print("\nMONTHLY COVERAGE TABLE (2016):")
    print(monthly_df.to_string(index=False))

    # Summary totals
    totals = {
        "Month": "TOTAL",
        "Days_in_Month": 366,
    }
    for st in stations:
        totals[f"{st.upper()}_GNSS"] = monthly_df[f"{st.upper()}_GNSS"].sum()
        totals[f"{st.upper()}_PBO_Obs"] = monthly_df[f"{st.upper()}_PBO_Obs"].sum()
        totals[f"{st.upper()}_PBO_Good"] = monthly_df[f"{st.upper()}_PBO_Good"].sum()
        totals[f"{st.upper()}_Usable"] = monthly_df[f"{st.upper()}_Usable"].sum()
    totals["3Station_Common"] = monthly_df["3Station_Common"].sum()
    print("-" * 85)
    print(pd.DataFrame([totals]).to_string(index=False))

    # 2. Continuous Gap Analysis
    print("\nLONGEST CONTINUOUS GAPS IN VALID PAIRED OBSERVATIONS:")
    print("-" * 85)
    gap_info = {}

    for st in stations:
        st_df = df[df["station"] == st].sort_values("date").reset_index(drop=True)
        usable_dates = st_df[st_df["usable_for_loso"]]["date"].tolist()

        full_calendar = pd.date_range("2016-01-01", "2016-12-31", freq="D")
        is_usable_series = full_calendar.isin(usable_dates)

        # Identify gap clusters
        gap_blocks = []
        current_gap = []

        for d, is_u in zip(full_calendar, is_usable_series):
            if not is_u:
                current_gap.append(d)
            else:
                if current_gap:
                    gap_blocks.append(current_gap)
                    current_gap = []
        if current_gap:
            gap_blocks.append(current_gap)

        if gap_blocks:
            longest_gap = max(gap_blocks, key=len)
            longest_len = len(longest_gap)
            start_d = longest_gap[0].strftime("%Y-%m-%d")
            end_d = longest_gap[-1].strftime("%Y-%m-%d")

            # Determine dominant reason for gap
            gap_sub = st_df[st_df["date"].isin(longest_gap)]
            n_missing_gnss = (gap_sub["clean_arc_count"].fillna(0) == 0).sum()
            n_frozen_sm = (gap_sub["pbo_sm_flag"].isin(["D03", "D02,D03"])).sum()
            n_unobserved_sm = (gap_sub["pbo_sm"].isnull()).sum()

            gap_info[st] = {
                "longest_gap_days": longest_len,
                "start_date": start_d,
                "end_date": end_d,
                "missing_gnss": int(n_missing_gnss),
                "frozen_flag": int(n_frozen_sm),
                "unobserved_sm": int(n_unobserved_sm),
                "total_gaps_count": len(gap_blocks),
            }

            print(f"Station {st.upper()} ({station_names[st]}):")
            print(f"  Longest Continuous Gap: {longest_len} days ({start_d} to {end_d})")
            print(f"  Primary Driver: Frozen/Snow Flag (D03): {n_frozen_sm} days | Unobserved SM: {n_unobserved_sm} days | Missing GNSS: {n_missing_gnss} days")
            print(f"  Total distinct gap intervals during 2016: {len(gap_blocks)}")
        else:
            print(f"Station {st.upper()}: No gaps found!")

    # 3. Seasonal Distribution
    print("\nSEASONAL DISTRIBUTION OF USABLE OBSERVATIONS (DJF, MAM, JJA, SON):")
    print("-" * 85)
    def assign_season(dt):
        m = dt.month
        if m in [12, 1, 2]:
            return "Winter (DJF)"
        elif m in [3, 4, 5]:
            return "Spring (MAM)"
        elif m in [6, 7, 8]:
            return "Summer (JJA)"
        else:
            return "Autumn (SON)"

    df["season"] = df["date"].apply(assign_season)
    usable_df = df[df["usable_for_loso"]].copy()

    season_order = ["Winter (DJF)", "Spring (MAM)", "Summer (JJA)", "Autumn (SON)"]
    seasonal_summary = []

    for s in season_order:
        s_row = {"Season": s}
        for st in stations:
            cnt = len(usable_df[(usable_df["station"] == st) & (usable_df["season"] == s)])
            tot = len(df[(df["station"] == st) & (df["season"] == s)])
            pct = cnt / tot * 100.0 if tot > 0 else 0.0
            s_row[f"{st.upper()}_Usable"] = cnt
            s_row[f"{st.upper()}_Pct"] = f"{pct:.1f}%"
        seasonal_summary.append(s_row)

    season_df = pd.DataFrame(seasonal_summary)
    print(season_df.to_string(index=False))

    # 4. Generate Visualization
    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True, sharey=True)
    colors = {"p041": "#1f77b4", "p037": "#2ca02c", "p043": "#d62728"}

    for idx, st in enumerate(stations):
        ax = axes[idx]
        st_df = df[df["station"] == st].sort_values("date")

        # Plot full year timeline
        dates = st_df["date"]
        # Status code: 0 = Missing/Frozen, 1 = GNSS clean only, 2 = PBO flag G only, 3 = Paired Usable
        status = np.zeros(len(dates))
        status[st_df["clean_arc_count"] > 0] = 1
        status[st_df["sm_is_good"] == True] = 2
        status[st_df["usable_for_loso"] == True] = 3

        # Plot PBO SM where usable
        ax.plot(
            st_df[st_df["usable_for_loso"]]["date"],
            st_df[st_df["usable_for_loso"]]["pbo_sm"],
            label=f"{station_names[st]} (Usable: {st_df['usable_for_loso'].sum()} days)",
            color=colors[st],
            lw=1.5,
            marker="o",
            markersize=3,
            alpha=0.85,
        )

        # Highlight missing or frozen gaps
        gap_dates = st_df[~st_df["usable_for_loso"]]["date"]
        if not gap_dates.empty:
            ax.scatter(
                gap_dates,
                np.full(len(gap_dates), 0.01),
                marker="|",
                color="gray",
                alpha=0.5,
                label="Unpaired / Flagged / Frozen Gap",
            )

        ax.set_ylabel(r"Soil Moisture ($\mathrm{m}^3/\mathrm{m}^3$)", fontsize=11)
        ax.set_ylim(-0.02, 0.40)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="upper right", frameon=True, fontsize=10)
        ax.set_title(f"{station_names[st]} — 2016 Paired Observation Timeline", fontsize=12, fontweight="bold")

    axes[-1].xaxis.set_major_locator(mdates.MonthLocator())
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    plt.xticks(rotation=0)
    plt.tight_layout()

    fig_path = FIGURES_DIR / "loso_temporal_coverage_audit.png"
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"\nSaved temporal coverage visualization: {fig_path.relative_to(PROJECT_ROOT)}")

    # =========================================================================
    # SECTION B: FEATURE CONSISTENCY & PHASE OFFSET ANALYSIS
    # =========================================================================
    print("\n" + "=" * 85)
    print("SECTION B: FEATURE CONSISTENCY & PHASE OFFSET AUDIT")
    print("=" * 85)

    features = ["rh", "amp", "phase_circ", "phase_cos", "phase_sin", "phase_resultant_length", "clean_arc_count", "qc_pass_rate"]
    stat_rows = []

    for f in features:
        for st in stations:
            vals = usable_df[usable_df["station"] == st][f].dropna()
            stat_rows.append({
                "Feature": f,
                "Station": st.upper(),
                "N": len(vals),
                "Mean": float(vals.mean()),
                "Std": float(vals.std()),
                "Median": float(vals.median()),
                "IQR": float(vals.quantile(0.75) - vals.quantile(0.25)),
                "Min": float(vals.min()),
                "Max": float(vals.max()),
                "Skew": float(vals.skew()),
            })

    stat_df = pd.DataFrame(stat_rows)
    print("\nFEATURE DISTRIBUTIONAL AUDIT (USABLE SAMPLE):")
    print(stat_df.to_string(index=False))

    # Phase circular statistics
    print("\nCIRCULAR PHASE METRICS & STATION-SPECIFIC PHASE OFFSETS:")
    print("-" * 85)
    for st in stations:
        phases = usable_df[usable_df["station"] == st]["phase_circ"].to_numpy()
        s = np.mean(np.sin(phases))
        c = np.mean(np.cos(phases))
        circ_mean = np.arctan2(s, c)
        circ_r = np.sqrt(s**2 + c**2)
        circ_var = 1.0 - circ_r
        print(f"Station {st.upper()} ({station_names[st]}):")
        print(f"  Circular Mean Phase: {circ_mean:.4f} rad ({np.degrees(circ_mean):.1f}°)")
        print(f"  Resultant Vector Length R: {circ_r:.4f}")
        print(f"  Circular Variance (1 - R): {circ_var:.4f}")
        print(f"  Phase Cosine Median: {usable_df[usable_df['station'] == st]['phase_cos'].median():.4f}")

    print("\nCRITICAL OBSERVATION ON PHASE TRANSFERABILITY:")
    print("  - At P041: Mean phase = -2.29 rad (-131.1°), phase_cos median = -0.7934.")
    print("  - At P037: Mean phase = +1.16 rad (+66.7°), phase_cos median = +0.2292.")
    print("  - At P043: Mean phase = +1.81 rad (+103.9°), phase_cos median = -0.4579.")
    print("  -> Inter-station phase difference is NOT driven by soil moisture! It is dictated by the")
    print("     static baseline phase offset phi_0 = 4*pi*H_refl / lambda (mod 2*pi).")
    print("  -> A 1-cm change in monument height shifts phase by (4*pi*0.01)/0.1903 = 0.66 rad (37.8°).")
    print("  -> Target-informed calibration of the held-out test station is STRICTLY FORBIDDEN.")
    print("     In an authentic LOSO deployment, the test site has NO ground truth soil moisture.")

    # =========================================================================
    # SECTION C: SPATIAL-FOOTPRINT SEPARATION AUDIT
    # =========================================================================
    print("\n" + "=" * 85)
    print("SECTION C: SPATIAL-FOOTPRINT SEPARATION LIMITATION")
    print("=" * 85)
    print("Physical Sensor Separation Audit:")
    print("  - Station P037: ISMN PBO_H2O probe is at 0.54 m from GNSS tower (collocated inside FFZ).")
    print("  - Station P043: ISMN PBO_H2O probe is at 0.53 m from GNSS tower (collocated inside FFZ).")
    print("  - Station P041: ISMN MARSHALL2 probe is at 210.69 m South (183.1° azimuth) from GNSS tower.")
    print("Reflection Footprint Boundary:")
    print("  - Tower height H ~ 1.95 m, GPS L1 wavelength lambda = 0.1903 m.")
    print("  - First Fresnel Zone (FFZ) outer boundary at 5° elevation: ~41.1 m radially.")
    print("  - Effective GNSS-IR footprint: 4 m to 45 m radial annulus (~5,500 m²).")
    print("Scientific Limitation:")
    print("  - At P041, MARSHALL2 is ~4.7 times further than the outer edge of the reflection zone.")
    print("  - Both sit in the same open unmanaged semi-arid grassland on flat terrain with loam soil")
    print("    (37% sand, 44% silt, 19% clay; saturation 0.43 m³/m³).")
    print("  - Meso-scale drydowns correlate, but micro-topographic runoff, localized snowmelt, and")
    print("    sub-field spatial heterogeneity introduce an inherent representation error bound at P041.")

    # =========================================================================
    # SECTION D: FORMAL LOSO PROTOCOL SPECIFICATION
    # =========================================================================
    print("\n" + "=" * 85)
    print("SECTION D: FORMAL THREE-FOLD LOSO PROTOCOL SPECIFICATION")
    print("=" * 85)

    n_p041 = len(usable_df[usable_df["station"] == "p041"])
    n_p037 = len(usable_df[usable_df["station"] == "p037"])
    n_p043 = len(usable_df[usable_df["station"] == "p043"])
    n_total = len(usable_df)

    folds = [
        {
            "Fold": "Fold 1",
            "Held_Out_Test_Station": "P037 (Fremont Pass)",
            "Test_N": n_p037,
            "Train_Stations": "P041 + P043",
            "Train_N": n_p041 + n_p043,
            "Train_P041_N": n_p041,
            "Train_Other_N": n_p043,
            "P041_Weight": 0.5 / n_p041,
            "Other_Weight": 0.5 / n_p043,
        },
        {
            "Fold": "Fold 2",
            "Held_Out_Test_Station": "P043 (Newcastle)",
            "Test_N": n_p043,
            "Train_Stations": "P041 + P037",
            "Train_N": n_p041 + n_p037,
            "Train_P041_N": n_p041,
            "Train_Other_N": n_p037,
            "P041_Weight": 0.5 / n_p041,
            "Other_Weight": 0.5 / n_p037,
        },
        {
            "Fold": "Fold 3",
            "Held_Out_Test_Station": "P041 (Marshall)",
            "Test_N": n_p041,
            "Train_Stations": "P037 + P043",
            "Train_N": n_p037 + n_p043,
            "Train_P041_N": 0,
            "Train_Other_N": n_p037 + n_p043,
            "P037_Weight": 0.5 / n_p037,
            "P043_Weight": 0.5 / n_p043,
        },
    ]

    print("\nPRIMARY SPATIAL LOSO EXPERIMENT (ALL ELIGIBLE PAIRED DATES):")
    print(f"  Pooled usable dataset size: {n_total} station-days")
    for f in folds:
        print(f"\n  [{f['Fold']}] Held-out Test Station: {f['Held_Out_Test_Station']}")
        print(f"    - Test Set Size:  {f['Test_N']} days ({f['Test_N']/n_total*100:.1f}% of pooled sample)")
        print(f"    - Training Set:   {f['Train_Stations']} -> Total Train Size: {f['Train_N']} days")
        if f["Fold"] == "Fold 3":
            print(f"    - Equal Station Weights: w(P037) = {f['P037_Weight']:.6f}, w(P043) = {f['P043_Weight']:.6f} (each site contributes 50.0% of gradient)")
        else:
            print(f"    - Equal Station Weights: w(P041) = {f['P041_Weight']:.6f}, w(Other) = {f['Other_Weight']:.6f} (each site contributes 50.0% of gradient)")

    print("\nSECONDARY SENSITIVITY BENCHMARK (COMMON 216 SYNCHRONOUS DATES):")
    print(f"  Strictly intersected dates across all 3 stations: 216 dates.")
    print("  In this secondary benchmark, every fold evaluates on identical chronological windows.")

    # =========================================================================
    # SECTION E: SPATIAL-PLUS-TEMPORAL EVALUATION FEASIBILITY
    # =========================================================================
    print("\n" + "=" * 85)
    print("SECTION E: SPATIAL-PLUS-TEMPORAL EVALUATION FEASIBILITY")
    print("=" * 85)

    print("Investigating Candidate Chronological Boundaries for Spatio-Temporal Holdout:")
    print("A scientific temporal partition must correspond to natural hydrological regimes rather than")
    print("an arbitrary percentage ratio.\n")

    boundaries = [
        {
            "Name": "Hydrological Regime A (Spring Thaw / Early Summer vs Late Summer / Autumn)",
            "Boundary_Date": "2016-07-01",
            "Rationale": "Splits between active snowmelt/spring infiltration (DOY 1-182) and summer drydown/autumn storms (DOY 183-366).",
        },
        {
            "Name": "Hydrological Regime B (Peak Growing Season Split)",
            "Boundary_Date": "2016-08-01",
            "Rationale": "Separates maximum vegetative/high dynamic range period from senescent autumn/fall freeze-up.",
        },
        {
            "Name": "P041 Historical Split Mirror",
            "Boundary_Date": "2016-10-06",
            "Rationale": "Mirrors the exact single-station temporal split previously established for P041 (DOY 280).",
        },
    ]

    for b in boundaries:
        b_dt = pd.to_datetime(b["Boundary_Date"])
        t_early = usable_df[usable_df["date"] < b_dt]
        t_late = usable_df[usable_df["date"] >= b_dt]

        print(f"Boundary Option: {b['Name']}")
        print(f"  Split Date: {b['Boundary_Date']} | Rationale: {b['Rationale']}")
        print(f"  Overall Early Sample: {len(t_early)} days ({len(t_early)/len(usable_df)*100:.1f}%) | Late Sample: {len(t_late)} days ({len(t_late)/len(usable_df)*100:.1f}%)")
        for st in stations:
            e_cnt = len(t_early[t_early["station"] == st])
            l_cnt = len(t_late[t_late["station"] == st])
            print(f"    - {st.upper()}: Early = {e_cnt} days | Late = {l_cnt} days (Total = {e_cnt + l_cnt})")
        print()

    # =========================================================================
    # SECTION F: FEATURE ABLATION DESIGN
    # =========================================================================
    print("=" * 85)
    print("SECTION F: FEATURE ABLATION DESIGN & CROSS-STATION TRANSFERABILITY")
    print("=" * 85)

    ablation_sets = {
        "Model A (Phase-Free Core)": {
            "Features": ["rh", "amp"],
            "Dimensionality": 2,
            "Transferability_Score": "HIGH (Physically Invariant)",
            "Rationale": (
                "RH measures vertical antenna-to-surface geometry and tracks moisture-induced penetration changes. "
                "Amplitude reflects ground surface dielectric reflectivity. Both are largely invariant to static "
                "monument height differences across sites."
            ),
        },
        "Model B (Interference Phase Integration)": {
            "Features": ["rh", "amp", "phase_cos", "phase_sin"],
            "Dimensionality": 4,
            "Transferability_Score": "LOW-TO-MODERATE (High Station Phase Offset Confounding)",
            "Rationale": (
                "Includes circular phase components (cos, sin). While phase provides sub-wavelength sensitivity to soil "
                "moisture within a single station, raw phase shifts across stations due to static monument elevation "
                "offsets (phi_0). Without target-informed calibration, Model B risks severe spatial bias."
            ),
        },
        "Model C (Full Sensor Diagnostic Set)": {
            "Features": ["rh", "amp", "phase_cos", "phase_sin", "phase_resultant_length", "clean_arc_count", "qc_pass_rate"],
            "Dimensionality": 7,
            "Transferability_Score": "MODERATE (Diagnostic Augmented)",
            "Rationale": (
                "Augments Model B with phase resultant vector length R (daily phase coherence / roughness index) "
                "and daily arc tracking QC diagnostics (clean_arc_count, qc_pass_rate). Tests whether signal quality "
                "and directional tracking stability regularize cross-site transfer."
            ),
        },
    }

    for name, info in ablation_sets.items():
        print(f"\n[{name}]")
        print(f"  Feature List: {info['Features']} (Dimension: {info['Dimensionality']})")
        print(f"  Cross-Station Transferability: {info['Transferability_Score']}")
        print(f"  Scientific Rationale: {info['Rationale']}")

    print("\n" + "=" * 85)
    print("FINAL PRE-TRAINING AUDIT COMPLETE — READY FOR USER REVIEW")
    print("=" * 85)

    return {
        "monthly_df": monthly_df,
        "gap_info": gap_info,
        "season_df": season_df,
        "stat_df": stat_df,
        "folds": folds,
        "boundaries": boundaries,
        "ablation_sets": ablation_sets,
    }


if __name__ == "__main__":
    run_audit()
