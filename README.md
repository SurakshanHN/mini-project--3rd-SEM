# Forecasting Household Appliance Energy Use

Mini Project (BCP308 Community Project) | **Team B-7**  
**SDG Alignment:** SDG 7 — Affordable and Clean Energy (**Target 7.3:** Double the global rate of improvement in energy efficiency; Indicator 7.3.1)

---

## 1. Project Overview

Households cannot shift or manage electricity load if they cannot anticipate when demand will peak. Without forward visibility, time-of-use tariffs and voluntary demand-management practices remain difficult to adopt. This project develops an anti-leakage short-term forecasting pipeline and retrospective demonstration app for whole-house appliance energy consumption (`Appliances`, Wh).

---

## 2. Problem Statement

Given historical energy consumption, calendar/time signals, indoor room temperatures and humidities, and outdoor weather telemetry available at forecast origin $t$, predict future appliance electricity consumption $\hat{y}(t+h)$ at two horizons:

- **$h = 1$ (10 minutes ahead):** Immediate short-term load momentum.
- **$h = 6$ (1 hour ahead):** Actionable scheduling horizon for shifting flexible domestic appliances.

---

## 3. Dataset Description

- **Source:** [UCI Appliances Energy Prediction Dataset (Dataset 374)](https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction) (Candanedo et al., 2017)
- **Local File:** `data/energydata_complete.csv`
- **Size:** 19,735 rows, 29 attributes, 0 missing values, 0 duplicates
- **Resolution & Span:** Strictly 10-minute intervals from `2016-01-11 17:00:00` to `2016-05-27 18:00:00` (~137 days, Stambruges, Belgium)
- **Attribute Groups:**
  - `Appliances`: Whole-house appliance energy use (Wh) — **Target**
  - `lights`: Sub-metered lighting energy (Wh) — **Excluded** (unrealistic in standard households)
  - `T1`–`T5`, `T7`–`T9` and `RH_1`–`RH_5`, `RH_7`–`RH_9`: Indoor room temperatures (°C) and humidities (%)
  - `T6`, `RH_6`: Exterior north-facade temperature (°C) and humidity (%)
  - `T_out`, `Press_mm_hg`, `RH_out`, `Windspeed`, `Visibility`, `Tdewpoint`: Weather station observations
  - `rv1`, `rv2`: Identical uniform random noise columns (`r = -0.0111` with target) — **Excluded from production**, used only for negative-control diagnostics

---

## 4. Forecasting Methodology

- **Anti-Leakage Rule:** Features at origin $t$ strictly use observations at or before $t$. Future sensor values at $t+h$ never enter the feature matrix.
- **Chronological Split (70 / 15 / 15):**
  - **Train:** `13,814` rows (`2016-01-11 17:00` → `2016-04-16 15:10`; `13,669 - h + 1` usable after initial 144-step lookback)
  - **Validation:** `2,960` rows (`2016-04-16 15:20` → `2016-05-07 04:30`; `2,959` rows for $h=1$, `2,954` for $h=6$)
  - **Test:** `2,961` rows (`2016-05-07 04:40` → `2016-05-27 18:00`; `2,960` rows for $h=1$, `2,955` for $h=6$)
- **Target Alignment:** `Appliances.shift(-h)` computed independently inside each split so no target crosses split boundaries.
- **Validation-Only Selection:** Hyperparameters and target transforms (`raw` vs `log1p`) are chosen exclusively on validation MAE.

---

## 5. Feature Engineering (`src/features.py`)

All training, evaluation, and Streamlit inference import the same `src/features.py` module (43 production features):

- **Time features (5):** `hour`, `hour_sin`, `hour_cos`, `day_of_week`, `is_weekend`
- **Consumption lags (6):** `appliance_lag_0` ($t$), `lag_1`, `lag_2`, `lag_3`, `lag_6` (1h), `lag_144` (24h)
- **Shifted rolling means (3):** `appliance_roll_mean_1h` (6 steps), `3h` (18 steps), `6h` (36 steps) computed via `shift(1).rolling(w).mean()` over $[t-w \dots t-1]$
- **Environmental context (29):** `indoor_temp_mean` (mean of `T1..T5, T7..T9`), `indoor_rh_mean`, `temp_diff_in_out`, `temp_diff_in_t6`, `rh_diff_in_out`, plus all 8 indoor temperatures, 8 indoor humidities, `T6`, `RH_6`, and 6 weather station readings
- **Leakage Verification:** Automated `pytest` suite (`tests/test_features.py`) verifies future perturbations leave features at $t$ unchanged and single-row app inference matches batch features identically.

---

## 6. Models

1. **Ridge Regression (validation-selected `alpha`):** `StandardScaler` + `Ridge` pipeline (fit on train only; `alpha` selected from `[0.1, 1.0, 10.0, 50.0, 100.0, 500.0, 1000.0]` via validation MAE even in the initial comparison).
2. **Random Forest Regressor:** 150 trees, `max_depth=12`, `min_samples_leaf=5`, `max_features="sqrt"`.
3. **LightGBM Regressor:** Gradient-boosted trees with validation early stopping, compared across `raw` and `log1p` targets (`expm1` inversion clipped at 0 before Wh evaluation).
4. **Tuned Winner:** Compact validation-only grid search over the strongest initial family (`LightGBM`), selecting L1 (`regression_l1`) objective for both horizons (`log1p` at $h=1$, `raw` at $h=6$).

---

## 7. Evaluation Methodology

Models are evaluated on original Wh using **MAE**, **RMSE**, **$R^2$**, and **Persistence Skill Score**:
$$\text{Skill Score} = 1 - \frac{\text{MAE}_{\text{model}}}{\text{MAE}_{\text{persistence}}}$$

Every model is benchmarked against all four Review 1 baselines (`Persistence`, `Mean by (hour, is_weekend)`, `Same Time Yesterday`, `Train-Mean`).

---

## 8. Actual Results

### 10-Minute Ahead ($h = 1$) — `outputs/model_results.csv`

| Model | Transform | Val MAE | Val RMSE | Val $R^2$ | Test MAE | Test RMSE | Test $R^2$ | Test Skill vs Pers. |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **LightGBM (Tuned)** | `log1p` | **22.23** | **58.47** | **0.5982** | **22.62** | **58.76** | **0.5820** | **+0.1543** |
| LightGBM | `log1p` | 23.51 | 59.64 | 0.5821 | 24.05 | 59.26 | 0.5748 | +0.1010 |
| Persistence (Baseline) | `none` | 26.16 | 66.44 | 0.4813 | 26.75 | 66.85 | 0.4590 | 0.0000 |
| Ridge | `raw` | 26.47 | 59.91 | 0.5782 | 27.08 | 60.47 | 0.5574 | -0.0123 |
| Random Forest | `log1p` | 27.16 | 66.83 | 0.4752 | 29.07 | 64.55 | 0.4955 | -0.0867 |
| LightGBM | `raw` | 26.24 | 57.86 | 0.6066 | 29.35 | 60.58 | 0.5558 | -0.0970 |
| Mean by (hour, is_weekend) | `none` | 48.51 | 87.23 | 0.1060 | 47.90 | 85.75 | 0.1098 | -0.7907 |

### 1-Hour Ahead ($h = 6$) — `outputs/model_results.csv`

| Model | Transform | Val MAE | Val RMSE | Val $R^2$ | Test MAE | Test RMSE | Test $R^2$ | Test Skill vs Pers. |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **LightGBM (Tuned)** | `raw` | **34.53** | 87.63 | 0.0969 | **33.89** | 84.73 | 0.1320 | **+0.2692** |
| LightGBM | `log1p` | 36.15 | 83.95 | 0.1713 | 35.57 | 80.04 | 0.2254 | +0.2328 |
| Ridge | `log1p` | 37.73 | 86.91 | 0.1117 | 37.41 | 81.43 | 0.1984 | +0.1933 |
| Ridge | `raw` | 42.93 | 83.94 | 0.1715 | 41.38 | 79.67 | 0.2326 | +0.1076 |
| Random Forest | `log1p` | 38.99 | 83.69 | 0.1764 | 41.72 | **78.94** | **0.2466** | +0.1003 |
| LightGBM | `raw` | 44.22 | **82.62** | **0.1973** | 45.22 | 79.20 | 0.2416 | +0.0249 |
| Persistence (Baseline) | `none` | 51.58 | 111.45 | -0.4609 | 46.37 | 101.76 | -0.2520 | 0.0000 |
| Mean by (hour, is_weekend) | `none` | 48.47 | 87.21 | 0.1057 | 47.97 | 85.82 | 0.1095 | -0.0345 |

### Hourly-Aggregated Forecasting (Complete 6-Step Hours, `outputs/hourly_results.csv`)
- **Hourly Persistence:** Test MAE `187.65 Wh`, RMSE `357.52 Wh`, $R^2 = 0.2914$
- **Hourly Mean (target hour, weekend):** Test MAE `249.33 Wh`, RMSE `385.21 Wh`, $R^2 = 0.1774$
- **Hourly Ridge (val-alpha):** Test MAE `195.70 Wh`, RMSE `317.03 Wh`, $R^2 = 0.4428$
- **Hourly LightGBM:** Test MAE `166.16 Wh`, RMSE `332.15 Wh`, $R^2 = 0.3884$

---

## 9. Installation Instructions

Requires **Python 3.12+** (tested on Python `3.14.10`).

```bash
git clone git@github.com:SurakshanHN/mini-project--3rd-SEM.git
cd mini-project--3rd-SEM

python3 -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

Run anti-leakage tests and model parity verification:

```bash
.venv/bin/pytest -v tests/test_features.py
.venv/bin/python scripts/verify_models.py
```

---

## 10. Notebook Execution Instructions

- **Review 1 (EDA & Baselines):** Open or execute `01_eda_baseline.ipynb`.
- **Review 2 (Features, ML Models, Evaluation & Model Export):**
  ```bash
  .venv/bin/jupyter nbconvert --to notebook --execute --inplace 02_features_models_eval.ipynb
  ```
All file paths resolve relative to the repository root.

---

## 11. Streamlit Application Instructions

Launch the retrospective test-period dashboard locally:

```bash
.venv/bin/streamlit run app.py
```

In the app sidebar:
1. Choose forecast horizon (**10 minutes** or **1 hour**).
2. Select a test date (`2016-05-07` to `2016-05-27`) and forecast origin time `t`.
3. Inspect the **ML Forecast**, **Persistence Forecast**, **Actual Consumption** (retrospective evaluation only), **High-Usage Window Warning** (based on training 90th percentile `210.0 Wh` / diagnostic spike `300.0 Wh`), and 12-hour history chart.

---

## 12. Deployment Instructions (Streamlit Community Cloud)

The repository is configured for Streamlit Community Cloud (`.streamlit/config.toml`, pinned `requirements.txt`, local `data/energydata_complete.csv`, and serialized `models/model_h*.joblib`):

1. Push all committed changes to GitHub (`git push origin main`).
2. Visit [share.streamlit.io](https://share.streamlit.io/) and click **New app**.
3. Select repository `SurakshanHN/mini-project--3rd-SEM`, branch `main`, and main file path `app.py`.
4. Under **Advanced settings**, select **Python 3.12** (or **3.13/3.14**; required because `numpy==2.5.5` requires Python `>=3.12`) and click **Deploy**.

---

## 13. Limitations

1. **Single House in Belgium:** Data reflects one low-energy house in Stambruges, Belgium (2016); appliance mixes and schedules differ from typical Indian households.
2. **Short Seasonal Span:** ~4.5 months (Jan–May 2016), excluding peak summer cooling loads.
3. **Whole-House Aggregate Target:** Sub-appliances are not metered separately, and sudden occupant-driven spikes (`>= 300 Wh`) remain difficult to predict 1 hour ahead (`F1 = 0.0000` at $h=6$ under point-median regression).
4. **No Tariff or Occupancy Logs:** No electricity pricing slabs or ground-truth occupancy sensors exist in the dataset.
5. **Community Interaction Pending:** Real household interviews in `field/interaction_log.csv` have not yet been logged.

---

## 14. SDG 7 Alignment

This project aligns with **SDG 7 (Target 7.3 / Indicator 7.3.1)** by demonstrating short-term household load visibility and descriptive peak-window awareness. No measured electricity or financial savings are claimed, as physical savings require real household intervention and long-term metering.
