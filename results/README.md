# Results Directory: Four-Method GNSS-IR Soil Moisture Benchmark

This directory contains the consolidated evaluation results, machine-readable configurations, and publication figures for the four implemented soil moisture retrieval methods at station P041 (Marshall, Colorado) for the year 2016.

## Directory Structure

```
results/
├── BENCHMARK_REPORT.md     # Detailed scientific report explaining all metrics, models, and findings
├── README.md               # Quick-reference overview (this file)
├── configs/                # Machine-readable JSON configurations for all models
│   ├── all_models_config.json
│   ├── climatology_config.json
│   ├── precipitation_ols_config.json
│   ├── random_forest_config.json
│   └── xgboost_config.json
└── figures/                # Publication-quality benchmark figures
    ├── benchmark_performance_all_methods.png
    └── benchmark_timeseries_all_methods.png
```

## Primary Benchmark Summary (Official 63 Dates: Oct 7 – Dec 31, 2016)

- **Mean Climatology Baseline:** RMSE = 0.049958 m³/m³, Bias = +0.046387 m³/m³, ubRMSE = 0.018550 m³/m³
- **Precipitation OLS Baseline:** RMSE = 0.051139 m³/m³, Bias = +0.047658 m³/m³, ubRMSE = 0.018545 m³/m³, r = 0.0368
- **Random Forest (Primary Median):** RMSE = 0.033697 m³/m³, Bias = +0.022714 m³/m³, ubRMSE = 0.024891 m³/m³, r = 0.4249
- **XGBoost (Optimal Tuned Grid_092):** RMSE = 0.034783 m³/m³, Bias = +0.028665 m³/m³, ubRMSE = 0.019703 m³/m³, r = 0.2253

Refer to `BENCHMARK_REPORT.md` for full error decomposition, parameter tables, and diagnostic sensitivity analyses.
