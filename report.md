# Project Report: Forecasting Household Appliance Energy Use

**Course:** BCP308 Community Project (Mini Project) | **Team:** B-7  
**SDG Alignment:** SDG 7 — Affordable and Clean Energy (Target 7.3, Indicator 7.3.1)  
**Repository:** `mini-project--3rd-SEM`

---

## 1. Abstract

Households cannot shift discretionary electrical loads if they cannot anticipate when consumption will rise, leaving time-of-use tariffs and voluntary demand-response measures underutilized. This project builds and evaluates an anti-leakage short-term forecasting pipeline for whole-house appliance electricity consumption (`Appliances`, Wh) using historical consumption lags, cyclical time encodings, indoor room telemetry, and outdoor weather observations from the UCI Appliances Energy Prediction dataset (19,735 intervals at 10-minute resolution). Models are evaluated chronologically across two horizons—10 minutes ahead ($h=1$) and 1 hour ahead ($h=6$)—against four transparent baselines including persistence. Tuned LightGBM models beat persistence at both horizons on held-out test MAE (`22.62 Wh` vs `26.75 Wh`, skill `+15.43%` at $h=1$; `33.89 Wh` vs `46.37 Wh`, skill `+26.92%` at $h=6$), while honest error and spike diagnostics confirm that sudden high-load spikes ($\ge 300\text{ Wh}$) remain inherently stochastic 1 hour ahead. The validated pipeline is packaged with `joblib` and served through a retrospective Streamlit demonstration dashboard (`app.py`).

---

## 2. Introduction

Residential electricity demand is characterized by low baseline loads punctuated by sharp spikes from cooking, laundry, heating, and water-heating appliances. Short-term demand forecasting provides forward visibility so occupants can stagger high-draw tasks away from peak windows. To remain credible for academic and real-world deployment, forecasting systems must prevent temporal data leakage and prove they outperform simple heuristics such as persistence ($\hat{y}(t+h) = y(t)$).

---

## 3. Problem Statement

At forecast origin $t$, predict whole-house appliance electricity consumption `Appliances` (Wh) at $t + h$ using **only** information available at or before time $t$:
$$\hat{y}(t+h) = f(\mathcal{F}_{\le t}), \quad h \in \{1\text{ (10 min)}, 6\text{ (1 hour)}\}$$
Future sensor values, weather observations, or target values at $t+1 \dots t+h$ must never enter the predictor matrix.

---

## 4. Community Rationale

Electricity consumers often receive only a single monthly bill with little visibility into which hours drive high demand. Without clear advance warnings, households cannot easily reschedule flexible appliances (such as washing machines, irons, or water pumps). Community outreach templates (`field/outreach_message.txt`, `field/round1_questions.md`) target local households to determine whether numeric Wh values or plain-language high-usage window warnings are more actionable.

---

## 5. Dataset

- **Dataset:** UCI Appliances Energy Prediction (Dataset 374; Candanedo et al., 2017), stored locally at `data/energydata_complete.csv`.
- **Dimensions:** 19,735 rows $\times$ 29 columns; 0 missing values; 0 duplicate rows or timestamps.
- **Temporal Coverage:** Strictly uniform 10-minute grid from `2016-01-11 17:00:00` to `2016-05-27 18:00:00` (137 days, 1 hour) from one residential house in Stambruges, Belgium.
- **Sensors & Exclusions:**
  - `T1`–`T5`, `T7`–`T9` and `RH_1`–`RH_5`, `RH_7`–`RH_9`: 8 indoor room temperature (°C) and humidity (%) zones.
  - `T6`, `RH_6`: Exterior north-facade temperature and humidity (`r(T6, T_out) = 0.9748`).
  - `T_out`, `Press_mm_hg`, `RH_out`, `Windspeed`, `Visibility`, `Tdewpoint`: Chievres weather station readings.
  - `lights` ($r = 0.1973$ with `Appliances`): Excluded from production features because standard residential homes lack sub-metered lighting circuits.
  - `rv1`, `rv2` (identical uniform random noise, $r = -0.0111$): Excluded from production features and retained solely for a negative-control diagnostic.

---

## 6. Exploratory Data Analysis

From `01_eda_baseline.ipynb`:
1. **Heavy Right Skewness:** Mean `97.69 Wh`, median `60.00 Wh`, 75th percentile `100.00 Wh`, max `1,080.00 Wh`; only `1.46%` (`288 / 19,735`) of intervals exceed `500 Wh`.
2. **Diurnal Routines:** Weekday consumption peaks at `194.89 Wh` (hour 18) with an overnight low of `48.10 Wh` (hour 3); weekend consumption stays elevated midday (`108.60`–`160.22 Wh` across hours 10–16) and peaks at `178.55 Wh` (hour 18).
3. **Weak Linear Weather Correlations:** All individual sensor correlations with `Appliances` have $|r| < 0.20$ (strongest: `RH_out` $r = -0.152$, `T2` $r = 0.120$, `T6` $r = 0.118$), motivating autoregressive lags and non-linear gradient boosting.

---

## 7. Methodology

1. **Chronological Splitting (70 / 15 / 15):**
   - **Train:** first `13,814` rows (`2016-01-11 17:00` → `2016-04-16 15:10`).
   - **Validation:** next `2,960` rows (`2016-04-16 15:20` → `2016-05-07 04:30`).
   - **Test:** final `2,961` rows (`2016-05-07 04:40` → `2016-05-27 18:00`).
2. **Leakage-Free Target Construction:** Inside each split separately, `target_h = Appliances.shift(-h)` drops the final $h$ rows so no target crosses a split boundary (`2,959` / `2,954` val rows and `2,960` / `2,955` test rows for $h=1$ / $h=6$).
3. **Train-Only Preprocessing:** `StandardScaler` (inside Ridge `Pipeline`) and all hyperparameters are fitted/selected strictly on training and validation sets; the test set is touched once for final evaluation.

---

## 8. Feature Engineering

Implemented once in `src/features.py` (43 production features) and shared identically across `02_features_models_eval.ipynb` and `app.py`:
- **Time (5):** `hour`, `hour_sin` ($\sin(2\pi\cdot\text{hour}/24)$), `hour_cos` ($\cos(2\pi\cdot\text{hour}/24)$), `day_of_week`, `is_weekend`.
- **Autoregressive Lags (6):** `appliance_lag_0` ($t$), `lag_1`, `lag_2`, `lag_3`, `lag_6` (1 hour), `lag_144` (24 hours).
- **Shifted Rolling Means (3):** `appliance_roll_mean_1h` (6 steps), `3h` (18 steps), `6h` (36 steps) computed via `Appliances.shift(1).rolling(w).mean()` over $[t-w \dots t-1]$.
- **Environmental Context (29):** `indoor_temp_mean` (average of `T1..T5, T7..T9`), `indoor_rh_mean`, `temp_diff_in_out` (`indoor_temp_mean - T_out`), `temp_diff_in_t6` (`indoor_temp_mean - T6`), `rh_diff_in_out`, plus all 8 indoor temperatures, 8 indoor humidities, `T6`, `RH_6`, and 6 weather station features.
- **Automated Verification (`tests/test_features.py`):** 4/4 `pytest` checks confirm forbidden columns are absent, perturbing future rows `> t` leaves features at `t` identical, single-timestamp inference matches batch features within `1e-12`, and split sizes match Review 1.

---

## 9. Models

- **Model A — Ridge Regression (validation-selected `alpha`):** `StandardScaler` + `Ridge`, selecting `alpha` from `[0.1, 1.0, 10.0, 50.0, 100.0, 500.0, 1000.0]` via validation MAE during initial model comparison (`h=1`: `alpha=100.0` raw / `1000.0` log1p; `h=6`: `alpha=1000.0` raw / `1000.0` log1p).
- **Model B — Random Forest Regressor:** Conservative fixed settings (`n_estimators=150`, `max_depth=12`, `min_samples_leaf=5`, `max_features="sqrt"`).
- **Model C — LightGBM Regressor:** Conservative defaults (`n_estimators=1000`, `learning_rate=0.03`, `num_leaves=31`, `min_child_samples=20`) with validation early stopping (`50` rounds).
- **Target Transform Comparison:** Every model was trained on both `raw` Wh and `log1p(Appliances)` (inverted via `expm1` and clipped at `0.0` before Wh evaluation).
- **Compact Validation-Only Tuning:** Tuned the strongest initial family (`LightGBM`) across 16 configurations (`num_leaves` $\in \{15, 31\}$, `min_child_samples` $\in \{20, 50\}$, `learning_rate` $\in \{0.02, 0.05\}$, `objective` $\in \{\text{regression}, \text{regression\_l1}\}$) across `raw` and `log1p`.

---

## 10. Experimental Setup

- **Runtime:** Python `3.14.10`, `pandas 3.1.1`, `numpy 2.5.5`, `scikit-learn 1.9.1`, `lightgbm 4.7.0`, `joblib 1.6.0`, `streamlit 1.67.0`.
- **Metrics:** MAE (Wh), RMSE (Wh), $R^2$, and Persistence Skill Score $= 1 - \text{MAE}_{\text{model}} / \text{MAE}_{\text{persistence}}$.
- **Additional Diagnostics:** Permutation feature importance (10 repeats on validation), `rv1`/`rv2` negative control, `>= 300 Wh` spike classification diagnostics, and complete-hour (`6` steps/hour, `3,289` hours) hourly-aggregated forecasting.

---

## 11. Results and Discussion

### A. Main Horizon Results (`outputs/model_results.csv`)

| Horizon | Model | Transform | Val MAE | Val RMSE | Val $R^2$ | Test MAE | Test RMSE | Test $R^2$ | Test Skill vs Pers. |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| `h=1` | **LightGBM (Tuned)** | `log1p` | **22.23** | **58.47** | **0.5982** | **22.62** | **58.76** | **0.5820** | **+0.1543** |
| `h=1` | LightGBM | `log1p` | 23.51 | 59.64 | 0.5821 | 24.05 | 59.26 | 0.5748 | +0.1010 |
| `h=1` | Persistence (Baseline) | `none` | 26.16 | 66.44 | 0.4813 | 26.75 | 66.85 | 0.4590 | 0.0000 |
| `h=1` | Ridge | `raw` | 26.47 | 59.91 | 0.5782 | 27.08 | 60.47 | 0.5574 | -0.0123 |
| `h=1` | Random Forest | `log1p` | 27.16 | 66.83 | 0.4752 | 29.07 | 64.55 | 0.4955 | -0.0867 |
| `h=1` | LightGBM | `raw` | 26.24 | 57.86 | 0.6066 | 29.35 | 60.58 | 0.5558 | -0.0970 |
| `h=6` | **LightGBM (Tuned)** | `raw` | **34.53** | 87.63 | 0.0969 | **33.89** | 84.73 | 0.1320 | **+0.2692** |
| `h=6` | LightGBM | `log1p` | 36.15 | 83.95 | 0.1713 | 35.57 | 80.04 | 0.2254 | +0.2328 |
| `h=6` | Ridge | `log1p` | 37.73 | 86.91 | 0.1117 | 37.41 | 81.43 | 0.1984 | +0.1933 |
| `h=6` | Ridge | `raw` | 42.93 | 83.94 | 0.1715 | 41.38 | 79.67 | 0.2326 | +0.1076 |
| `h=6` | Random Forest | `log1p` | 38.99 | 83.69 | 0.1764 | 41.72 | **78.94** | **0.2466** | +0.1003 |
| `h=6` | LightGBM | `raw` | 44.22 | **82.62** | **0.1973** | 45.22 | 79.20 | 0.2416 | +0.0249 |
| `h=6` | Persistence (Baseline) | `none` | 51.58 | 111.45 | -0.4609 | 46.37 | 101.76 | -0.2520 | 0.0000 |
| `h=6` | Mean by (hour, is_weekend) | `none` | 48.47 | 87.21 | 0.1057 | 47.97 | 85.82 | 0.1095 | -0.0345 |

### B. Key Analytical Findings
1. **Persistence Is Strong at 10 Minutes, Beaten Only by `log1p` / L1 LightGBM:** At $h=1$, Ridge with validation-selected `alpha` (`raw` and `log1p`), Random Forest (`raw` and `log1p`), and initial `raw` L2 LightGBM all failed to beat persistence (`26.75 Wh`). Only initial `log1p` LightGBM (`24.05 Wh`) and tuned L1 `log1p` LightGBM (`22.62 Wh`, `+15.43%` skill) outperformed persistence.
2. **At 1 Hour ($h=6$), Combining Autoregressive Lags and Diurnal Signals Excels:** Tuned LightGBM (`regression_l1`, `raw`) cuts test MAE from `46.37 Wh` (persistence) and `47.97 Wh` (hour/weekend mean) down to `33.89 Wh` (`+26.92%` skill) while raising $R^2$ from `-0.2520` to `+0.1320`.
3. **Target Transformation (`log1p`):** `log1p` consistently helped the initial models under L2 loss (`h=1` LightGBM: `26.24 -> 23.51` val MAE; `h=6` LightGBM: `44.22 -> 36.15`, Ridge with val-alpha: `42.93 -> 37.73`, RF: `52.26 -> 38.99`), whereas validation tuning with `regression_l1` directly minimized MAE at `h=6` (`34.53` val MAE).
4. **Feature Importance & Joint `rv1`/`rv2` Negative Control:** Validation permutation importance ranks `appliance_lag_0`, `appliance_lag_1`, and `hour` highest at $h=1$, and `appliance_lag_0`, `hour_cos`, `hour_sin`, and `hour` highest at $h=6$. Exact equality check confirms `(rv1 == rv2).all() == True` (`max(|rv1 - rv2|) = 0.0`). Because permuting one duplicate leaves the identical copy available, we ran both individual and **joint** permutation (10 repeats): at $h=1$, individual `rv1` (`+0.0133 Wh`), `rv2` (`-0.0070 Wh`), and **joint `rv1+rv2` (`-0.0022 ± 0.0131 Wh`)** are negligible compared to `appliance_lag_0` (`+31.2616 Wh`); at $h=6$, individual `rv1` (`+0.0009 Wh`), `rv2` (`0.0000 Wh`), and **joint `rv1+rv2` (`-0.0075 ± 0.0047 Wh`)** are negligible compared to `hour_cos` (`+4.0690 Wh`). This sanity check does not prove the absence of temporal leakage.
5. **Spike Analysis ($\ge 300\text{ Wh}$, 135 test spikes):** At $h=1$, tuned LightGBM achieved higher precision (`0.6538` vs `0.4851` for persistence) with lower recall (`0.3778` vs `0.4815`, F1 `0.4789` vs `0.4833`). At $h=6$, L1 point-median regression predicted `0` events $\ge 300\text{ Wh}$ (`F1 = 0.0000`), honestly confirming that rare 1-hour-ahead spikes cannot be reliably captured by point regression alone.
6. **Hourly-Aggregated Forecasting (`outputs/hourly_results.csv`):** Summing complete 6-step hours (`3,289` hours; row timestamp = start of completed hour, forecast origin = end of completed hour = start of target hour, calendar features & baselines aligned strictly to `target_hour_start` across train/val/test): `Hourly LightGBM` achieved Val MAE `173.13 Wh` / Test MAE `166.16 Wh` ($R^2 = 0.3884$) vs `Hourly Persistence` (`223.43` / `187.65 Wh`, $R^2 = 0.2914$), `Hourly Ridge (val-alpha=100.0)` (`198.19` / `195.70 Wh`, $R^2 = 0.4428$), and `Hourly Mean (target hour, weekend)` (`252.66` / `249.33 Wh`, $R^2 = 0.1774`).

---

## 12. Community Interaction

**Status: PENDING REAL HOUSEHOLD INTERVIEWS.**  
Preparation materials (`field/outreach_message.txt`, `field/round1_questions.md`, `field/round2_guide_and_template.md`) and logging schema (`field/interaction_log.csv`) are complete, but **0 real interviews** have been logged yet. No participant data or quotations have been fabricated. Findings and resulting project modifications will be added once Round 1 and Round 2 household conversations are completed.

---

## 13. Application and Deployment

- **Artifacts:** `models/model_h1.joblib` (1.45 MB) and `models/model_h6.joblib` (294 KB) encapsulate the trained LightGBM estimators, exact 43-feature ordering, transform flags, and training thresholds (`90th percentile = 210.0 Wh`), verified for `1e-12` prediction parity via `scripts/verify_models.py`.
- **Streamlit App (`app.py`):** Provides horizon selection (`10 min` / `1 hour`), chronological test-period timestamp selection, ML vs Persistence vs Retrospective Actual metric cards, a 12-hour history & point-forecast plot, and a descriptive high-usage window banner. Verified locally via `streamlit.testing.v1.AppTest` and headless server startup.
- **Deployment Readiness:** Configured with `.streamlit/config.toml` and pinned `requirements.txt` for Streamlit Community Cloud (`SurakshanHN/mini-project--3rd-SEM`, `app.py`). Public cloud deployment remains to be triggered on `share.streamlit.io` after pushing to GitHub.

---

## 14. Limitations

- Single Belgian residence (2016) rather than local Indian households.
- 4.5-month late-winter/spring span; summer cooling demand is absent.
- Whole-house aggregate target (`Appliances`) without sub-appliance labels or tariff schedules.
- Unscheduled 1-hour-ahead spikes ($\ge 300\text{ Wh}$) remain unpredictable under point regression.
- Field validation logs are still pending.

---

## 15. SDG 7 Alignment

Supports **SDG 7 (Target 7.3 / Indicator 7.3.1)** conceptually by demonstrating short-term consumption forecasting and descriptive peak-window awareness tools that help households recognize elevated demand periods. No measured energy or bill savings are claimed.

---

## 16. Conclusion

An end-to-end, leakage-audited household energy forecasting pipeline was built and verified. Tuned LightGBM models meaningfully improve upon persistence MAE at both 10-minute (`+15.43%`) and 1-hour (`+26.92%`) horizons, as well as on hourly-aggregated loads (`166.16 Wh` vs `187.65 Wh`), while honestly documenting the difficulty of predicting sudden 1-hour spikes.

---

## 17. Future Scope

- Complete Round 1 and Round 2 local household interviews and refine `app.py` alert wording/thresholds accordingly.
- Explore quantile regression (e.g., 80th/90th percentile LightGBM) for spike-probability alerts instead of single point forecasts.
- Evaluate on multi-home Indian smart-meter datasets with real time-of-use tariff structures.
