# Review 1 Preparation & Viva Defense Notes
**Course:** BCP308 Community Project (Mini Project) | **Team:** B-7  
**Project:** Forecasting Household Appliance Energy Use  
**Dataset:** UCI Appliances Energy Prediction (19,735 records, 10-minute resolution, Jan–May 2016)

---

## Problem in Plain Language

Households cannot shift or manage their electricity load if they cannot anticipate when their demand will peak. Without forward visibility, time-of-use tariffs and voluntary demand-reduction efforts remain difficult for consumers to act upon. This project develops and evaluates a short-term forecasting workflow to anticipate impending aggregate household electricity demand from ambient environmental telemetry.

---

## 1. Project Overview & Forecasting Ground Rules

- **Objective:** Forecast short-term whole-house electricity consumption (`Appliances`, Wh) at horizon $t+h$ using only information available up to time $t$.
- **Forecast Horizons:**
  - $h = 1$ step (10 minutes ahead)
  - $h = 6$ steps (1 hour ahead)
- **Target Variable:** Total whole-house appliance electricity draw (Wh). It is an aggregate household reading, not sub-metered for individual appliances.
- **Strict Anti-Leakage Constraints:**
  - Predictors at time $t$ never access telemetry, weather, or consumption from $t+h$.
  - Target is constructed independently within each split using `shift(-h)`, dropping the final $h$ rows of each partition so that no target pair spans across a split boundary.
  - Chronological split only: 70% Train (13,814 rows), 15% Validation (2,960 rows), 15% Test (2,961 rows) by temporal row order. Zero random shuffling.

---

## 2. Sensor Clarifications & Feature Exclusions

- **Indoor Sensors:**
  - Temperatures: `T1`–`T5`, `T7`–`T9` (8 interior zones: kitchen, living room, laundry, office, bathroom, ironing room, teenager room, parents room).
  - Relative Humidities: `RH_1`–`RH_5`, `RH_7`–`RH_9` (8 interior zones).
- **Outdoor Sensors:**
  - `T6` (temperature) and `RH_6` (humidity) are outdoor sensors mounted on the north facade of the house.
  - They correlate strongly with local weather station observations over the full dataset:
    - **Pearson $r(T_6, T_{\text{out}}) = 0.9748$**
    - **Pearson $r(\text{RH}_6, \text{RH}_{\text{out}}) = 0.7186$**
  - Weather station readings: `T_out`, `Press_mm_hg`, `RH_out`, `Windspeed`, `Visibility`, `Tdewpoint`.
- **Reason for Excluding `lights`:**
  - `lights` is excluded because a standard consumer household does not have sub-metered lighting circuits. Using `lights` at time $t$ is not future leakage (the reading is known at time $t$), but assuming its availability requires sub-metering infrastructure that is unrealistic for normal deployments.
- **Role of `rv1` and `rv2`:**
  - Synthetic random noise columns introduced by the original dataset authors. They are identical columns ($r = -0.0111$ with target) and act as a single negative noise control. Any properly regularized machine learning model in Review 2 should assign them near-zero feature importance.

---

## 3. Consolidated Baseline Results Table

All figures are quoted directly from [baseline_results.csv](file:///home/unhealthycoder/Borderland/miniproject-3rdSEM/appliance-forecast/outputs/baseline_results.csv) (training target mean $\bar{y}_{\text{train}} = 98.78$ Wh):

| Horizon | Baseline Model | Val MAE (Wh) | Val RMSE (Wh) | Val R² | Test MAE (Wh) | Test RMSE (Wh) | Test R² |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **$h=1$ (10-min)** | **Persistence** | **26.16** | **66.44** | **0.4813** | **26.75** | **66.85** | **0.4590** |
| $h=1$ (10-min) | Same Time Yesterday | 53.32 | 115.40 | -0.5647 | 52.51 | 109.49 | -0.4513 |
| $h=1$ (10-min) | Train-Mean | 53.98 | 92.41 | -0.0034 | 52.83 | 90.90 | -0.0004 |
| $h=1$ (10-min) | Mean by (hour, is_weekend) | 48.51 | 87.23 | 0.1060 | 47.90 | 85.75 | 0.1098 |
| **$h=6$ (1-hour)** | **Persistence** | **51.58** | **111.45** | **-0.4609** | **46.37** | **101.76** | **-0.2520** |
| $h=6$ (1-hour) | Same Time Yesterday | 53.24 | 115.38 | -0.5657 | 52.59 | 109.58 | -0.4519 |
| $h=6$ (1-hour) | Train-Mean | 53.94 | 92.38 | -0.0036 | 52.85 | 90.96 | -0.0004 |
| $h=6$ (1-hour) | Mean by (hour, is_weekend) | 48.47 | 87.21 | 0.1057 | 47.97 | 85.82 | 0.1095 |

### Side-by-Side Comparison at Horizon $h=6$ (1-Hour Ahead):

| Model Baseline | Val MAE (Wh) | Val RMSE (Wh) | Val R² | Test MAE (Wh) | Test RMSE (Wh) | Test R² |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Persistence** | 51.58 | 111.45 | -0.4609 | **46.37** | 101.76 | -0.2520 |
| **Mean by (hour, is_weekend)** | **48.47** | **87.21** | **0.1057** | 47.97 | **85.82** | **0.1095** |

**Metric Trade-Off at 1 Hour:**  
At $h=6$, persistence has lower test MAE than the hour-of-day baseline (**46.37 Wh** vs. **47.97 Wh**), but higher validation MAE (**51.58 Wh** vs. **48.47 Wh**). Meanwhile, the hour-of-day baseline achieves lower RMSE (**87.21 Wh** vs. **111.45 Wh** on val; **85.82 Wh** vs. **101.76 Wh** on test) and positive $R^2$ on both splits (**0.1057** on val; **0.1095** on test), whereas persistence produces negative $R^2$ (**-0.4609** on val; **-0.2520** on test). Therefore, **there is no single best baseline at $h=6$**, and models in Review 2 must be evaluated against both.

---

## 4. Key Takeaways & Review Findings

1. **Persistence Dominates at 10 Minutes ($h=1$):**  
   At $h=1$, persistence achieves a Test MAE of 26.75 Wh and an $R^2$ of 0.4590, cutting the test MAE of the naive train mean (52.83 Wh) nearly in half. Total whole-house consumption exhibits short-term temporal autocorrelation because household activities span multiple consecutive 10-minute intervals. Beating persistence at $h=1$ provides a stringent benchmark for Review 2 models.
2. **Empirical Verification of the "Spike Carried Forward" Phenomenon ($h=6$ vs. $h=1$):**  
   - At $h=1$, for test rows where $\text{Appliances}(t) \ge 300$ Wh (134 rows), consumption 10 minutes later remains elevated with a **median of 290.00 Wh** and a **mean of 329.55 Wh**. Only **2.99% (4 / 134)** drop below 100 Wh, and only **17 spikes** are missed from baseline.  
   - In contrast, at $h=6$, when a spike ($\ge 300$ Wh, 133 rows) is carried forward, median consumption 1 hour later falls sharply to **130.00 Wh** (mean: **186.24 Wh**), with **21.05% (28 / 133)** of rows dropping below 100 Wh back toward baseline. Furthermore, persistence completely misses **50 spikes** where usage jumped from $< 100$ Wh at time $t$ to $\ge 300$ Wh at $t+6$.  
   - Because squared error squares these large individual discrepancies ($> 150$ Wh residuals), carrying spikes forward and missing newly ignited spikes heavily inflates RMSE to **101.76 Wh** and drives $R^2$ negative (**-0.2520**), even while test MAE (**46.37 Wh**) beats the train mean (**52.85 Wh**).
3. **Consistency of Same-Time-Yesterday Across Horizons:**  
   The same-time-yesterday baseline produces near-identical results at $h=1$ (Test MAE: 52.51 Wh, RMSE: 109.49 Wh) and $h=6$ (Test MAE: 52.59 Wh, RMSE: 109.58 Wh). For any given target timestamp $T$, the prediction is identically the observed reading from 24 hours earlier ($T - 24\text{h}$). The slight numerical variation is solely due to dropping 5 additional boundary rows in the $h=6$ evaluation split (2,960 rows for $h=1$ vs. 2,955 rows for $h=6$).
4. **Baseline Trade-Off at 1 Hour ($h=6$):**  
   At $h=6$, persistence has lower test MAE than the hour-of-day baseline (46.37 Wh vs. 47.97 Wh) but higher val MAE (51.58 Wh vs. 48.47 Wh); the hour-of-day baseline achieves lower RMSE and higher $R^2$ on both splits. There is no single best baseline at $h=6$.

---

## 5. Plan for Review 2

- **Feature Engineering Pipeline:**
  - *Calendar & Diurnal:* Hour-of-day cyclical encoding ($\sin(2\pi \cdot \text{hour}/24)$, $\cos(2\pi \cdot \text{hour}/24)$), day of week (0–6), and weekend binary flag.
  - *Autoregressive Lags:* Past consumption lags at $t-1, t-2, t-3, t-6$ (1 hour prior), and $t-144$ (24 hours prior).
  - *Rolling Statistics:* Shifted rolling means of consumption over 1h, 3h, and 6h windows, computed strictly from historical values (using `shift(1)`) to avoid data leakage.
  - *Thermal & Weather Gradients:* Indoor-minus-outdoor temperature differences (e.g., average indoor temperature minus outdoor $T_6$ or $T_{\text{out}}$) and weather telemetry gradients.
- **Model Exploration (Four Candidates):**
  1. *Ridge Regression:* Regularized linear model serving as an interpretable parametric baseline.
  2. *Random Forest Regressor:* Non-linear bagging ensemble capable of handling feature interactions.
  3. *Gradient Boosted Decision Trees (LightGBM or XGBoost):* Efficient, high-performance boosting on tabular time-series features.
  4. *Small Tuned Winner:* A compact, hyperparameter-tuned model chosen from the top candidate, optimized for generalization without overfitting.
- **Target Transformation:**
  - Train models on $\log(1 + y)$ (`log1p(Appliances)`) to stabilize the right-skewed distribution, and invert with $\text{expm1}$ for evaluation on the original Wh scale.
- **Evaluation Metrics & Benchmarking:**
  - Standard metrics: MAE (Wh), RMSE (Wh), and $R^2$ reported separately on validation and test splits.
  - *Skill Score Formulation:*
    $$\text{Skill Score} = 1 - \frac{\text{MAE}_{\text{model}}}{\text{MAE}_{\text{baseline}}}$$
    Reported against **both** reference baselines:
    1. Persistence baseline
    2. Mean by (hour, is_weekend) baseline
- **Interpretability & Robustness Testing:**
  - *Permutation Importance:* Feature importance evaluated against the synthetic noise control `rv1` (and `rv2`). Features with importance below or comparable to `rv1` will be flagged as noise.
  - *Aggregated & Peak-Window Test:* Evaluation on hourly-aggregated consumption and peak-window detection accuracy to assess whether predictions offer practical utility for household demand management beyond 10-minute noise.

---

## 6. Field Interaction Plan

- **Objective:** Ground the forecasting project in real-world community energy management and gather qualitative user feedback on notification formats.
- **Participants:** Two planned interview rounds with local household contacts (family, neighbours, hostel/PG residents).
- **Round 1 Topics:**
  - Awareness of electricity tariffs and slab rates.
  - Knowledge of when household electricity usage typically peaks during the day.
  - Willingness to shift flexible energy usage if alerted in advance.
  - User preference among alert formats: an hourly predicted number, a high-usage window warning, or a daily summary.
- **Round 2 Topics:**
  - Presentation of prototype forecast alert mockups.
  - Evaluation of clarity, perceived usefulness, and willingness to act upon notifications.
- **Logging & Data Management:**
  - Interactions are recorded in [interaction_log.csv](file:///home/unhealthycoder/Borderland/miniproject-3rdSEM/appliance-forecast/field/interaction_log.csv) with columns: `date,household_label,consent_given,what_i_showed,what_they_said,what_i_changed_in_project`.
- **Current Status:**
  - **No field findings exist yet. None are invented.**

---

## 7. Limitations

1. **Single House in Belgium:** Data is collected from a single detached residence in Stambruges, Belgium. Building construction, climate, appliance stock, and lifestyle patterns differ considerably from Indian households.
2. **Limited Seasonal Window:** The dataset spans roughly 4.5 months (January 11 to May 27, 2016), covering winter into late spring. Summer and autumn dynamics (such as seasonal cooling loads) are not captured.
3. **Whole-House Aggregate Consumption Only:** The target is total household electricity draw (`Appliances`, Wh). Individual appliances or sub-circuits are not metered separately.
4. **Absence of Tariff Data:** The dataset contains no electricity pricing, time-of-use tariffs, or demand charges.
5. **Absence of Occupancy Logs:** No ground-truth occupancy counts, room-by-room presence indicators, or occupant activity logs are available.

---

## 8. Examiner / Viva Defense Questions

### Question 1: Why does $R^2$ go negative for persistence at 1 hour ($h=6$)?
**Answer:**  
$R^2$ compares the model's sum of squared residuals ($\text{SS}_{\text{res}}$) against the total variance around the test set mean ($\text{SS}_{\text{tot}}$):
$$R^2 = 1 - \frac{\text{SS}_{\text{res}}}{\text{SS}_{\text{tot}}} = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$$
Predicting the constant mean achieves $R^2 \approx 0$. When a model's sum of squared residuals exceeds the variance of the data ($\text{SS}_{\text{res}} > \text{SS}_{\text{tot}}$), $R^2$ becomes negative.  
Empirical analysis of the test set reveals that when an elevated consumption spike ($\ge 300$ Wh) occurs at time $t$, persistence predicts that same elevated level 1 hour later at $t+6$. However, 60 minutes later, median consumption drops sharply to **130.00 Wh** (mean: **186.24 Wh**), and **21.05% (28 / 133)** of rows have fallen below 100 Wh back toward baseline. Furthermore, persistence completely misses **50 spikes** where consumption jumped from $< 100$ Wh to $\ge 300$ Wh. Because squared-error loss squares these large residuals ($> 150$ Wh), carrying spikes forward and missing newly ignited spikes heavily inflates $\text{SS}_{\text{res}}$, driving $R^2$ to **-0.2520** and RMSE to **101.76 Wh** (worse than the train mean's **90.96 Wh**). Meanwhile, MAE evaluates errors linearly, so persistence still beats the train mean on MAE (**46.37 Wh** vs. **52.85 Wh**).

### Question 2: Why does "same-time-yesterday" score so poorly ($R^2 \approx -0.45$)?
**Answer:**  
Although human daily schedules exhibit general diurnal rhythms, predicting from a single point exactly 24 hours prior ($T - 144$ steps) fails for three likely reasons:
1. **High Temporal Resolution (10-minute intervals):** Household activities vary in exact start time day-to-day. If an activity begins at 18:10 on Tuesday but 18:40 on Wednesday, the model suffers a double penalty: predicting high when actual use is low, and predicting low when actual use is high.
2. **Stochastic Day-to-Day Operation:** High-energy household tasks do not recur on an identical 24-hour cycle; many happen intermittently throughout the week.
3. **Weekday vs. Weekend Shifts:** Comparing Monday to Sunday or Saturday to Friday introduces large schedule mismatches.  
Conditioning on the *average* over the training set for each (hour, weekend flag) smooths out this single-point noise, improving test MAE to **47.90 Wh** and lifting $R^2$ from **-0.45** to **+0.11**.

### Question 3: Why must chronological splitting be used instead of random K-Fold cross-validation?
**Answer:**  
Time-series telemetry violates the independent and identically distributed (i.i.d.) assumption. In an autocorrelation-rich dataset with 10-minute sampling, random shuffling places points from $t-1$ and $t+1$ into the training set while evaluating on time $t$. This results in severe data leakage, artificially inflating validation accuracy. Chronological splitting preserves temporal causality.

### Question 4: Why are $T_6$ and $RH_6$ separated from indoor sensors?
**Answer:**  
Per the UCI variable table, $T_6$ and $RH_6$ are sensors mounted on the exterior north facade of the building. Indoor sensors comprise $T_1$–$T_5$, $T_7$–$T_9$ and $RH_1$–$RH_5$, $RH_7$–$RH_9$. Over the full dataset, $T_6$ and $RH_6$ correlate strongly with the weather station readings ($r = 0.9748$ for $T_6$ vs. $T_{\text{out}}$; $r = 0.7186$ for $\text{RH}_6$ vs. $\text{RH}_{\text{out}}$).

### Question 5: Why is `lights` excluded if its value at time $t$ is already known?
**Answer:**  
Excluding `lights` is not an anti-leakage requirement; at time $t$, `lights(t)` is historical data. Rather, it is an operational constraint: `lights` is an active electricity consumption stream. Standard households do not possess sub-metered lighting circuits. Relying on sub-metered lighting inputs would render the forecasting system inapplicable to standard residential smart-meter deployments.

### Question 6: What is the purpose of `rv1` and `rv2`?
**Answer:**  
Columns `rv1` and `rv2` are identical uniform random variables generated as benchmark controls. Because they have negligible correlation with target consumption ($r = -0.0111$), they serve as a negative control: in Review 2, feature importance analysis must demonstrate that models assign them near-zero predictive weight.

### Question 7: Why is persistence considered the primary baseline in short-term forecasting?
**Answer:**  
In high-frequency time series (such as 10-minute sampling), electrical loads display strong immediate temporal autocorrelation. A persistence model costs zero training time, has no parameters, and frequently outperforms naive statistical aggregates at short horizons. It sets a minimum performance bar: any machine learning model that cannot beat persistence at $h=1$ has failed to extract meaningful patterns from telemetry.

### Question 8: Why not use an LSTM or Deep Learning architecture?
**Answer:**  
Deep sequential architectures like LSTMs require substantial training data, extensive hyperparameter tuning, and significant computational overhead to prevent overfitting. With approximately 19,735 records spanning only 4.5 months, tabular gradient-boosted trees (e.g., LightGBM) and regularized linear regression are widely demonstrated to achieve competitive or superior performance on tabular telemetry while remaining fast, lightweight, and easily interpretable.

### Question 9: Why can $R^2$ be low even for a well-trained forecasting model on this dataset?
**Answer:**  
Household electricity consumption contains significant stochastic variance due to unscheduled human activities. Because $R^2$ evaluates squared errors relative to the mean, occasional sudden spikes heavily penalize the metric even when the model accurately captures the baseline load and provides practically usable MAE values. In residential load forecasting, modest $R^2$ values are typical.

### Question 10: What if the machine learning model barely beats persistence?
**Answer:**  
We will report it plainly and objectively. In empirical data science, identifying that a simple heuristic matches complex models is a valid and informative finding. Furthermore, this community project includes field interaction studies, user feedback analysis, and alert format design, which provide practical deliverables regardless of whether the model margin over persistence is large or modest.

### Question 11: Why use Belgian dataset for an Indian community project?
**Answer:**  
The UCI dataset serves as a rigorous, peer-reviewed open benchmark to build and validate the forecasting methodology and pipeline. The community field interactions do not evaluate Belgian consumption figures; instead, they evaluate the *practical usefulness, clarity, and actionability of the resulting alert formats* (such as peak-window warnings) among local residents.

### Question 12: How does this project relate to SDG 7.3 and Indicator 7.3.1?
**Answer:**  
SDG Target 7.3 aims to double the global rate of improvement in energy efficiency, measured by Indicator 7.3.1 (energy intensity measured in terms of primary energy and GDP). This project aligns with the intent of SDG 7.3 by promoting consumer energy awareness and facilitating voluntary peak load management. However, we claim no measured energy intensity reductions or actual measured energy savings, as verifying physical savings would require sub-metering infrastructure, tariff incentives, and long-term post-intervention monitoring.
