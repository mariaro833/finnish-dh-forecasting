# Probabilistic Forecasting of Finnish CHP-Based District Heating
### Temporal Fusion Transformer with Residual-from-Persistence Target Reformulation

[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

> **Paper:** "Probabilistic Forecasting of Cogenerated District Heating in Finland Under Structural Fleet Transition: A Temporal Fusion Transformer Approach"  
> **Author:** Maria Rohnonen
> **Preprint:** Zenodo DOI: 10.5281/zenodo.20118487

---

## Overview

This repository contains the full data pipeline, model training scripts, baselines, and figure reproduction code for a probabilistic 24-hour forecasting model of national Finnish CHP-based district heating, trained on nine years (2016–2024) of hourly Fingrid open data.

The key contribution is a **residual-from-persistence target reformulation** that handles non-stationarity caused by structural fleet transition (CHP plant decommissioning). The TFT-resid v3 model achieves:

| Metric | Value |
|---|---|
| MAE | 87.4 MWh/h |
| sMAPE | 21.2% |
| PI80 coverage | 49.9% |
| PI80 width | 101.5 MWh/h |

This work forms the research foundation of the **ET-Design Lab** (Aurinkolab Community), where the forecasting methodology is applied in the construction of an AI-driven Uusimaa Heating Digital Twin — a real student-built tool at the intersection of cutting-edge energy research and hands-on engineering education for teenagers.

---

## Repository Structure

```
├── data/
│   ├── fetch_fingrid.py          # Download district heating data from Fingrid API
│   ├── fetch_weather.py          # Download weather data from Open-Meteo archive API
│   ├── fetch_prices2.py          # Download Finnish electricity spot prices from ENTSO-E
│   └── build_dataset.py          # Merge and engineer features into final dataset
│
├── models/
│   ├── train_tft_csc_v3.py       # TFT-resid v3 — final model (residual target)
│   ├── train_job_v3.sh           # SLURM job for TFT-resid v3 (A100, 8h)
│   ├── train_tft_csc.py          # TFT-abs v2 — absolute target version
│   ├── train_job_v2.sh           # SLURM job for TFT-abs v2
│   └── eval_job.sh               # SLURM job for evaluation
│
├── baselines/
│   ├── xgb_v3.py                 # XGBoost v3 — graded residual target (final baseline)
│   ├── xgb_v3_job.sh             # SLURM job for XGBoost v3
│   ├── xgb_v1.py                 # XGBoost v1 — absolute target
│   ├── xgb_v1_job.sh             # SLURM job for XGBoost v1
│   └── naive_baselines.py        # Persistence and seasonal naive baselines
│
├── figures/
│   ├── fig1_horizon_mae.py       # Figure 1: Per-horizon MAE comparison
│   ├── fig2_training_curves.py   # Figure 2: Training curves (TFT-abs vs TFT-resid)
│   └── fig3_calibration.py       # Figure 3: PI80 calibration bar chart
│
├── results/
│   ├── results_v3.json           # TFT-resid v3 test results
│   ├── results_v2.json           # TFT-abs v2 test results
│   ├── results_xgb_v3.json       # XGBoost v3 test results
│   ├── results_naive.json        # Naive baseline results
│   └── training_curves.json      # Training/validation loss curves
│
└── README.md
```

---

## Requirements

```bash
pip install pandas numpy matplotlib requests pytorch-forecasting pytorch-lightning xgboost
```

Tested on:
- Python 3.10
- PyTorch 2.4
- CSC Mahti HPC cluster (A100 GPU, SLURM)

---

## Data Sources

| Source | Description | Access |
|---|---|---|
| [Fingrid Open Data](https://data.fingrid.fi) | District heating CHP production (Dataset 201) | Free API key required |
| [Open-Meteo Archive](https://open-meteo.com) | Hourly weather — Helsinki, Espoo, Vantaa | Free, no key required |
| [ENTSO-E Transparency](https://transparency.entsoe.eu) | Finnish electricity spot prices | Free API key required |

---


## Usage

### 0. Set up environment variables

Create a `.env` file in the project root:

Then install python-dotenv:

```bash
pip install python-dotenv
```

> **Important:** 
Add `.env` to your `.gitignore` — never commit API keys to GitHub.

### 1. Fetch data

```bash
python data/fetch_fingrid.py
python data/fetch_weather.py
python data/fetch_prices2.py
```

python data/fetch_fingrid.py
python data/fetch_weather.py
python data/fetch_prices2.py
```

### 2. Build dataset

```bash
python data/build_dataset.py
# Output: data_full_real.csv (~78,000 rows, 2016–2024)
```

### 3. Train TFT-resid v3 (CSC Mahti)

```bash
sbatch models/train_job_v3.sh
# Requires: A100 GPU, ~6h walltime
```

### 4. Run XGBoost v3 baseline

```bash
sbatch baselines/xgb_v3_job.sh
# CPU only, ~1h walltime
```

### 5. Run naive baselines

```bash
python baselines/naive_baselines.py
```

### 6. Reproduce figures

```bash
# Requires results JSON files in results/
python figures/fig1_horizon_mae.py
python figures/fig2_training_curves.py
python figures/fig3_calibration.py
```

---

## Model Versions

| Model | Target | MAE (MWh/h) | sMAPE % | PI80 cov % |
|---|---|---|---|---|
| TFT-abs v2 | Absolute heat load | 115.5 | 37.0 | 19.4 |
| TFT-resid v3 | Residual from persistence | **87.4** | **21.2** | 49.9 |
| XGBoost v1 | Absolute heat load | 189.3 | 61.5 | 43.9 |
| XGBoost v3 | Graded residual | 143.5 | 35.9 | 74.0 |
| Naive persistence | — | 175.5 | — | — |
| Seasonal naive 24h | — | 183.7 | — | — |

---

## Citation

If you use this code or data pipeline, please cite:

```
Rohnonen, M. (2025). Probabilistic Forecasting of Cogenerated District Heating in Finland
Under Structural Fleet Transition: A Temporal Fusion Transformer Approach.
[Zenodo DOI — add after upload]
```

---

## License

Code: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)  
Data: Subject to Fingrid, Open-Meteo, and ENTSO-E terms of use.

---

## Acknowledgements

This research was carried out at Arcada University of Applied Sciences (Helsinki) as part of the Big Data Analytics post-master specialisation programme, using the CSC Mahti HPC cluster (project_2001220).
