# Roadmap: Forecasting Household Appliance Energy Use

**Course:** BCP308 Community Project (Mini Project) | **Team:** B-7 | **Guide:** RS | **Co-Guide:** ES | **Venue:** Room 503
**Dataset:** UCI 374 Appliances Energy Prediction (19,735 records, 29 features)
**SDG:** 7.3 / Indicator 7.3.1

---

## Timeline

| When | Goal |
|---|---|
| Sun 22 Nov (today) | Load data, run EDA, compute persistence baseline, prepare Review 1 points, message 6-8 households |
| Mon 23 Nov | **Review 1** (3:30pm), then start feature work |
| Tue 24 - Wed 25 Nov | Features, models, evaluation, get ready for Review 2 |
| Thu 26 Nov | **Review 2**, plus first round of household conversations |
| Fri 27 - Sat 28 Nov | Deploy the app, second round of conversations |
| Sun 29 Nov | Report, notebook cleanup, demo rehearsal |
| Mon 30 Nov | **Review 3** |

---

## Step 1: Setup (today, ~1 hour)

- [ ] Create a GitHub repo and a Colab notebook; mount Google Drive.
- [ ] Download `energydata_complete.csv` from the UCI page.
- [ ] Save a copy in Drive and in the repo so the demo never depends on UCI being reachable.

## Step 2: Understand the data (today)

**Facts about the data**
- 10-minute resolution, about 4.5 months in early 2016, from one house in Belgium.
- Target is `Appliances` (Wh): heavily right-skewed with rare large spikes.
- Indoor sensors: `T1`-`T9`, `RH_1`-`RH_9`.
- Outdoor weather: `T_out`, `RH_out`, `Press_mm_hg`, `Windspeed`, `Visibility`, `Tdewpoint`.
- `rv1` and `rv2` are random noise columns. Keep them only as a sanity check; a good model should ignore them.

**Tasks**
- [ ] Check missing values and duplicates.
- [ ] Plot the target over time.
- [ ] Average the target by hour of day.
- [ ] Plot correlations.

**Decide your framing now**
- Your abstract says *forecast*, so predict `y(t+h)` using only information available at time `t`.
- Pick horizons of **10 minutes** and **1 hour**.
- Do not use sensor values from `t+h` as inputs. That leaks the future and inflates results.
- Leave `lights` and `rv1`/`rv2` out of the main model. `lights` is also a consumption reading and may not exist in a real household setup.

## Step 3: Baselines (Review 1 material)

- [ ] **Persistence:** predict `y(t+h) = y(t)`.
- [ ] Optional but cheap: same time yesterday.
- [ ] Report MAE, RMSE and R² for both.

This is your benchmark. Everything after is judged against it.

## Step 4: Features

| Type | Details |
|---|---|
| Time | Hour (sin/cos encoded), day of week, weekend flag |
| Lags | Appliances at t-1, t-2, t-3, t-6 (1 hour), t-144 (24 hours) |
| Rolling means | Past usage over 1h, 3h, 6h. Always shift them so the current target never leaks in |
| Context | Indoor-minus-outdoor temperature, current humidity and weather |

**Splitting**
- Split **chronologically, never randomly**, e.g. 70% train / 15% validation / 15% test in time order.
- Fit any scaler on the training portion only.

## Step 5: Models (keep it to four)

1. Ridge or linear regression
2. Random Forest
3. LightGBM or XGBoost (likely your best)
4. Optional: a small tuned version of the winner

Skip LSTMs and deep learning. With limited time, and on this dataset, they rarely beat gradient boosting by enough to matter.

Because the target is skewed:
- Compare MAE alongside RMSE.
- Try a `log1p` transform of the target.
- Report skill score vs persistence: `1 - MAE_model / MAE_persistence`.

## Step 6: Evaluation and honest findings (Review 2 material)

- [ ] Table of all models vs both baselines, at both horizons.
- [ ] Plot predicted vs actual for a typical week.
- [ ] Plot error by hour of day.
- [ ] Permutation importance; check that `rv1`/`rv2` rank near the bottom.
- [ ] Test hourly-aggregated or peak-window predictions, which are usually more predictable and more useful to a household.

Expect modest scores. Published results on this dataset are not high, and your abstract already says the honest finding is that the target is hard to predict. If your model barely beats persistence at 10 minutes, say so plainly.

## Step 7: Field interaction (off the laptop)

Use two rounds with real households (family, neighbours, hostel or PG residents).

**Round 1 (before Review 2)**
- Ask about tariffs.
- Ask whether they know their usage pattern.
- Ask whether they would change habits.
- Ask what form of alert they would actually use: an hourly number, a "high-usage window" warning, or a daily summary.

**Round 2 (after the app works)**
- Show them the output.
- Ask whether they would shift appliance use because of it.

**Interaction record**
Log every interaction with: date, household (with consent), what you showed, what they said, **and what you changed in the project because of it**. The deliverable explicitly asks for "what changed", so let their feedback shape your output format.

> Everything in this record must come from real conversations. Evaluators will probe it.

## Step 8: Deployment (27-28 Nov)

- [ ] Save the model with `joblib`.
- [ ] Build a small Streamlit app (`app.py`): pick a timestamp from the test period, then show the forecast, the persistence forecast and the actual, with a plot and a simple "high-usage window ahead" message.
- [ ] Add `requirements.txt` with pinned versions.
- [ ] Push to GitHub.
- [ ] Deploy free on Streamlit Community Cloud or Hugging Face Spaces.

**Backups for demo day**
- A Gradio interface in Colab with `share=True` (temporary public link).
- A short screen recording of the app working.

**Notebook**
- [ ] Make sure the Colab notebook runs top to bottom on "Restart and run all".
- [ ] Share the link with view access.

## Step 9: Report (29 Nov)

Suggested structure:

1. Problem and community rationale
2. Dataset
3. Method
4. Results (including baseline comparison)
5. Field interaction
6. Limitations
7. SDG 7 link

**SDG 7.3:** say the project supports load awareness and efficiency. Do not claim measured savings, since none were measured.

**Limitations to state openly**
- Single house in Belgium
- Short time span
- No tariff data

## Step 10: Demo rehearsal

Run through it twice for the 3:30-3:40 slot. Ten minutes is short:

| Segment | Time |
|---|---|
| Problem | 1 min |
| Data | 1 min |
| Baseline vs model result | 1 min |
| App demo | 1 min |
| Field findings | 1 min |
| Questions | Remaining time |

---

## Deliverables checklist

- [ ] Synopsis
- [ ] Working ML model in Google Colab with evaluation results
- [ ] Field interaction record (who was consulted and what changed)
- [ ] Project report
- [ ] Final demonstration
