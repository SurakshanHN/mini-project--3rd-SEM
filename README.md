# Forecasting Household Appliance Energy Use

Mini Project (BCP308 Community Project) | Team B-7  
**SDG 7:** Affordable and Clean Energy (Target 7.3: Double the global rate of improvement in energy efficiency)

---

## 1. Project Overview

Households cannot shift or manage energy load if they cannot anticipate it. Without accurate forward visibility, demand-response programs and dynamic time-of-use tariffs remain underutilized.

This project investigates short-term forecasting of household appliance energy consumption (`Appliances`, measured in Wh) using indoor environmental telemetry (temperatures `T1`–`T9`, relative humidities `RH_1`–`RH_9`) and local outdoor weather metrics from a single low-energy house in Belgium.

### Forecasting Framing & Horizons
- **Target Variable:** `Appliances` energy consumption (Wh) at time $t + h$.
- **Forecast Horizons:**
  - **10-minute horizon ($h = 1$ step):** Immediate short-term adjustment.
  - **1-hour horizon ($h = 6$ steps):** Actionable appliance scheduling window for household load shifting.
- **Strict Anti-Leakage Ground Rules:**
  - Strictly use data available at or before time $t$ to forecast time $t + h$.
  - **Never** use sensor, weather, or target values from $t + h$ as input features.
  - **Chronological Split Only:** 70% Train, 15% Validation, 15% Test in temporal order (never random splits).
  - Benchmark against an honest **Persistence Baseline** ($y(t+h) = y(t)$).
  - Every finding and metric is reproducible directly from code execution.

---

## 2. Dataset

- **Source:** [UCI Appliances Energy Prediction Dataset](https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction) (Candanedo et al., 2017)
- **File:** `data/energydata_complete.csv`
- **Volume:** 19,735 records, 29 attributes
- **Frequency:** 10-minute intervals
- **Period:** January 11, 2016 – May 27, 2016 (~4.5 months)
- **Location:** Stambruges, Belgium (single residential house)
- **Features:**
  - `date`: Timestamp (`YYYY-MM-DD HH:MM:SS`)
  - `Appliances`: Total energy use of appliances (Wh) — **Primary Target**
  - `lights`: Energy use of light fixtures in the house (Wh)
  - `T1`–`T5`, `T7`–`T9`: Indoor temperatures in 8 distinct rooms (°C)
  - `T6`: Outside temperature on north side of building (°C)
  - `RH_1`–`RH_5`, `RH_7`–`RH_9`: Indoor humidity in 8 distinct rooms (%)
  - `RH_6`: Outside humidity on north side of building (%)
  - `T_out`, `Press_mm_hg`, `RH_out`, `Windspeed`, `Visibility`, `Tdewpoint`: Chievres weather station observations
  - `rv1`, `rv2`: Identical synthetic random variables (acting as one noise control)

---

## 3. Project Structure

```text
appliance-forecast/
├── data/
│   └── energydata_complete.csv   # Raw UCI dataset (10-min resolution)
├── figures/                      # Visualizations and EDA plots
├── outputs/                      # Saved metrics, baseline outputs, and model artifacts
├── 01_eda_baseline.ipynb         # Data verification, exploratory analysis, persistence baselines
├── project_synopsis.md           # Formal project abstract and community rationale
├── roadmap.md                    # Project roadmap and milestone timeline
├── requirements.txt              # Python environment dependencies
└── README.md                     # Project documentation
```

---

## 4. Setup & Quickstart

### Prerequisites
- Python 3.10+ (tested with Python 3.14)
- VS Code with Jupyter extension

### Environment Setup
```bash
# 1. Clone repository & navigate to root
cd appliance-forecast

# 2. Create and activate a local virtual environment
python -m venv .venv
source .venv/bin/activate       # On Linux / macOS
# .venv\Scripts\activate        # On Windows

# 3. Install dependencies
pip install -r requirements.txt
```

### Running the Notebook
Open `01_eda_baseline.ipynb` in VS Code, select the `.venv` Python kernel, and run all cells. All paths are resolved relative to the project root via `pathlib.Path`.
