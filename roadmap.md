# Roadmap: Forecasting Household Appliance Energy Use

**Course:** BCP308 Community Project (Mini Project) | **Team:** B-7 | **Guide:** RS | **Co-Guide:** ES | **Venue:** Room 503
**Dataset:** UCI 374 Appliances Energy Prediction (19,735 records, 29 features)
**SDG:** 7.3 / Indicator 7.3.1

---

## Timeline

| When | Goal |
|---|---|
| Sun 22 Nov | Load data, run EDA, compute persistence baseline, prepare Review 1 points, prepare field outreach |
| Mon 23 Nov | **Review 1** (completed), then start feature work |
| Tue 24 - Wed 25 Nov | Features, models, evaluation, get ready for Review 2 |
| Thu 26 Nov | **Review 2**, plus first round of household conversations |
| Fri 27 - Sat 28 Nov | Build & deploy the app, second round of conversations |
| Sun 29 Nov | Report, notebook cleanup, demo rehearsal |
| Mon 30 Nov | **Review 3** |

---

## Step 1: Setup

- [x] Create a GitHub repo (`SurakshanHN/mini-project--3rd-SEM`) and local workspace.
- [x] Download `energydata_complete.csv` from the UCI page.
- [x] Save a local copy in `data/energydata_complete.csv` so the demo never depends on UCI being reachable.

## Step 2: Understand the data (Review 1)

- [x] Check missing values (`0`) and duplicates (`0`).
- [x] Plot the target over time (`figures/01_appliances_time_series.png`).
- [x] Average the target by hour of day (`figures/03_mean_appliances_by_hour.png`).
- [x] Plot correlations (`figures/04_correlation_heatmap.png`, `figures/05_correlation_with_target.png`).
- [x] Lock forecasting framing (`y(t+h)` for `h=1` and `h=6`), exclude `lights` and `rv1`/`rv2` from main models.

## Step 3: Baselines (Review 1)

- [x] **Persistence:** `y(t+h) = y(t)` (`Test MAE: 26.75 Wh` at `h=1`, `46.37 Wh` at `h=6`).
- [x] Same time yesterday, Train-Mean, and Mean by `(hour, is_weekend)`.
- [x] Report Val/Test MAE, RMSE and R² (`outputs/baseline_results.csv`).

## Step 4: Features (`src/features.py`)

- [x] Time features: `hour`, `hour_sin`, `hour_cos`, `day_of_week`, `is_weekend`.
- [x] Lags: `Appliances` at `t`, `t-1`, `t-2`, `t-3`, `t-6` (1h), `t-144` (24h).
- [x] Shifted rolling means: past `1h`, `3h`, `6h` computed over `[t-w .. t-1]`.
- [x] Context: `indoor_temp_mean - T_out`, `indoor_temp_mean - T6`, indoor/outdoor humidity & weather.
- [x] Chronological 70/15/15 split and automated anti-leakage tests (`tests/test_features.py` — 4/4 passed).

## Step 5: Models (`02_features_models_eval.ipynb`)

- [x] Ridge Regression (train-only `StandardScaler`, validation-tuned `alpha`).
- [x] Random Forest Regressor.
- [x] LightGBM Regressor (with validation early stopping).
- [x] Compact validation-only tuned LightGBM winner for both horizons.
- [x] Compare `raw` vs `log1p` target training (inverted with `expm1` before Wh evaluation).
- [x] Report Persistence Skill Score (`1 - MAE_model / MAE_persistence`).

## Step 6: Evaluation and honest findings (Review 2)

- [x] Unified table of all models vs all 4 baselines at both horizons (`outputs/model_results.csv`).
- [x] Plot predicted vs actual for a typical test week (`07`, `08`).
- [x] Plot ML vs persistence comparison (`09`) and error by hour of day (`10`).
- [x] Permutation importance (`11`) and `rv1`/`rv2` negative-control check (`13`).
- [x] Spike prediction analysis (`12`) and hourly-aggregated forecasting (`14`, `outputs/hourly_results.csv`).

## Step 7: Field interaction (off the laptop)

- [x] Prepare outreach message (`field/outreach_message.txt`), Round 1 questionnaire (`field/round1_questions.md`), Round 2 template (`field/round2_guide_and_template.md`), and log schema (`field/interaction_log.csv`).
- [ ] **Round 1:** Interview 6–8 real households and log responses in `field/interaction_log.csv`.
- [ ] **Round 2:** Show working `app.py` prototype, ask whether they would shift appliance use, and log what changed in the project.

> Everything in `field/interaction_log.csv` must come from real conversations.

## Step 8: Deployment (Review 3)

- [x] Save production models with `joblib` (`models/model_h1.joblib`, `models/model_h6.joblib`) and verify parity (`scripts/verify_models.py`).
- [x] Build Streamlit app (`app.py`) reusing `src/features.py` and test locally (`AppTest` + headless server).
- [x] Pin `requirements.txt`, fix `.gitignore`, and add `.streamlit/config.toml`.
- [ ] Push final commits to GitHub (`git push origin main`).
- [ ] Deploy free on Streamlit Community Cloud (`share.streamlit.io`) and verify public URL.

## Step 9: Report

- [x] Draft complete technical report in `report.md` and update `README.md` with verified Review 2/3 results.
- [ ] Update Section 12 of `report.md` once real Round 1 & Round 2 household logs are collected.
- [x] Make sure `02_features_models_eval.ipynb` runs top-to-bottom cleanly (`nbconvert --execute`).

## Step 10: Demo rehearsal

- [ ] Rehearse 10-minute presentation twice (Problem 1m, Data 1m, Baseline vs ML 1m, App demo 1m, Field findings 1m, Q&A).

---

## Deliverables checklist

- [x] Synopsis (`project_synopsis.md`)
- [x] Working ML notebook with verified evaluation results (`02_features_models_eval.ipynb`)
- [x] Working Streamlit demo app (`app.py`) & saved `joblib` artifacts (`models/`)
- [ ] Field interaction record (who was consulted and what changed — `field/interaction_log.csv`)
- [x] Draft project report (`report.md` — pending final field section)
- [ ] Final demonstration
