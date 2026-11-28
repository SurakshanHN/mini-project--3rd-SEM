"""Shared feature engineering and anti-leakage splitting utilities.

Used identically by the Review 2 training/evaluation notebook and the
Review 3 Streamlit application so training and inference never diverge.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# Sensor groupings verified against UCI Appliances Energy Prediction (Dataset 374):
# T6 and RH_6 are mounted on the exterior north facade of the building.
INDOOR_TEMP_COLS: Tuple[str, ...] = ("T1", "T2", "T3", "T4", "T5", "T7", "T8", "T9")
INDOOR_RH_COLS: Tuple[str, ...] = (
    "RH_1",
    "RH_2",
    "RH_3",
    "RH_4",
    "RH_5",
    "RH_7",
    "RH_8",
    "RH_9",
)
OUTDOOR_SENSOR_COLS: Tuple[str, ...] = ("T6", "RH_6")
WEATHER_COLS: Tuple[str, ...] = (
    "T_out",
    "Press_mm_hg",
    "RH_out",
    "Windspeed",
    "Visibility",
    "Tdewpoint",
)

# Columns excluded from all production models:
# - 'lights': sub-metered consumption unavailable in typical households
# - 'rv1', 'rv2': synthetic uniform random noise (negative control only)
FORBIDDEN_PRODUCTION_COLS: Tuple[str, ...] = ("lights", "rv1", "rv2")
RANDOM_CONTROL_COLS: Tuple[str, ...] = ("rv1", "rv2")

# Autoregressive lags (in 10-minute steps) and rolling window sizes
LAG_STEPS: Tuple[int, ...] = (0, 1, 2, 3, 6, 144)
ROLLING_WINDOWS_STEPS: Tuple[Tuple[str, int], ...] = (
    ("1h", 6),
    ("3h", 18),
    ("6h", 36),
)
MAX_LOOKBACK_STEPS: int = 144

# Chronological split ratios matching Review 1 (70% / 15% / 15%)
SPLIT_RATIOS: Tuple[float, float, float] = (0.70, 0.15, 0.15)


def load_and_validate_data(csv_path: str | Path) -> pd.DataFrame:
    """Load dataset, parse timestamps, sort chronologically, and validate schema."""
    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    required_cols = (
        {"date", "Appliances", "lights"}
        | set(INDOOR_TEMP_COLS)
        | set(INDOOR_RH_COLS)
        | set(OUTDOOR_SENSOR_COLS)
        | set(WEATHER_COLS)
        | set(RANDOM_CONTROL_COLS)
    )
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"Dataset missing required columns: {sorted(missing_cols)}")

    if df.isna().any().any():
        raise ValueError("Dataset contains unexpected missing values.")
    if df["date"].duplicated().any():
        raise ValueError("Dataset contains duplicate timestamps.")

    diffs = df["date"].diff().dropna()
    if not (diffs == pd.Timedelta(minutes=10)).all():
        raise ValueError("Dataset timestamps are not strictly spaced at 10-minute intervals.")

    return df


def chronological_split_indices(n_total: int) -> Tuple[int, int, int]:
    """Return (n_train, n_val, n_test) matching Review 1 70/15/15 split."""
    n_train = int(SPLIT_RATIOS[0] * n_total)
    n_val = int(SPLIT_RATIOS[1] * n_total)
    n_test = n_total - n_train - n_val
    return n_train, n_val, n_test


def get_feature_columns(include_random_controls: bool = False) -> List[str]:
    """Return deterministic ordered list of engineered model input columns."""
    cols: List[str] = [
        # Time features
        "hour",
        "hour_sin",
        "hour_cos",
        "day_of_week",
        "is_weekend",
        # Consumption lag features (backward-looking at origin t)
        *[f"appliance_lag_{lag}" for lag in LAG_STEPS],
        # Shifted rolling means (strictly past windows [t-w .. t-1])
        *[f"appliance_roll_mean_{label}" for label, _ in ROLLING_WINDOWS_STEPS],
        # Engineered environmental context
        "indoor_temp_mean",
        "indoor_rh_mean",
        "temp_diff_in_out",
        "temp_diff_in_t6",
        "rh_diff_in_out",
        # Raw indoor temperature & humidity telemetry (excluding T6/RH_6)
        *INDOOR_TEMP_COLS,
        *INDOOR_RH_COLS,
        # Outdoor facade sensors & weather station readings
        *OUTDOOR_SENSOR_COLS,
        *WEATHER_COLS,
    ]
    if include_random_controls:
        cols.extend(RANDOM_CONTROL_COLS)
    return cols


def build_features(
    df: pd.DataFrame,
    include_random_controls: bool = False,
) -> pd.DataFrame:
    """Build backward-looking feature matrix from chronologically ordered data.

    Every feature at row index t depends strictly on observations at <= t:
    - Time features come solely from df['date'] at t.
    - Consumption lags use shift(k) for k >= 0 (Appliances at t-k).
    - Rolling means use shift(1).rolling(w).mean(), averaging strictly past
      observations [t-w .. t-1] so current/future targets never leak in.
    - Environmental & weather features come strictly from timestamp t.
    - 'lights' is always excluded; 'rv1'/'rv2' are excluded unless
      include_random_controls=True (reserved for negative-control diagnostics).
    """
    if not df["date"].is_monotonic_increasing:
        raise ValueError("Input DataFrame must be sorted chronologically by 'date'.")

    out = pd.DataFrame(index=df.index)

    # 1. Time features (cyclical hour + calendar flags)
    dt = df["date"].dt
    hour = dt.hour.astype(float)
    out["hour"] = hour
    out["hour_sin"] = np.sin(2.0 * np.pi * hour / 24.0)
    out["hour_cos"] = np.cos(2.0 * np.pi * hour / 24.0)
    out["day_of_week"] = dt.day_of_week.astype(float)
    out["is_weekend"] = (dt.day_of_week >= 5).astype(float)

    # 2. Consumption lag features (strictly backward-looking)
    appliances = df["Appliances"].astype(float)
    for lag in LAG_STEPS:
        out[f"appliance_lag_{lag}"] = appliances.shift(lag)

    # 3. Shifted backward-looking rolling mean features ([t-w .. t-1])
    shifted_appliances = appliances.shift(1)
    for label, window in ROLLING_WINDOWS_STEPS:
        out[f"appliance_roll_mean_{label}"] = shifted_appliances.rolling(
            window=window, min_periods=window
        ).mean()

    # 4. Environmental context (indoor means excluding outdoor T6/RH_6)
    indoor_temp_mean = df[list(INDOOR_TEMP_COLS)].mean(axis=1)
    indoor_rh_mean = df[list(INDOOR_RH_COLS)].mean(axis=1)
    out["indoor_temp_mean"] = indoor_temp_mean
    out["indoor_rh_mean"] = indoor_rh_mean
    out["temp_diff_in_out"] = indoor_temp_mean - df["T_out"]
    out["temp_diff_in_t6"] = indoor_temp_mean - df["T6"]
    out["rh_diff_in_out"] = indoor_rh_mean - df["RH_out"]

    for col in (*INDOOR_TEMP_COLS, *INDOOR_RH_COLS, *OUTDOOR_SENSOR_COLS, *WEATHER_COLS):
        out[col] = df[col].astype(float)

    if include_random_controls:
        for col in RANDOM_CONTROL_COLS:
            out[col] = df[col].astype(float)

    ordered_cols = get_feature_columns(include_random_controls=include_random_controls)
    if not include_random_controls:
        forbidden = set(FORBIDDEN_PRODUCTION_COLS) & set(ordered_cols)
        assert not forbidden, f"Forbidden columns found in production features: {forbidden}"

    return out[ordered_cols]


def prepare_forecast_datasets(
    df: pd.DataFrame,
    horizon: int,
    include_random_controls: bool = False,
) -> Dict[str, object]:
    """Construct leakage-free train/val/test feature matrices and aligned targets.

    - Backward-only features are computed on the continuous series so validation
      and test retain historical lag/rolling context at split boundaries.
    - Splits are sliced strictly by chronological 70/15/15 row indices.
    - Targets y(t+h) = Appliances.shift(-horizon) are built INSIDE each split
      independently, dropping the final `horizon` rows so no target spans a split.
    - Initial MAX_LOOKBACK_STEPS rows with NaN lags are dropped from train only,
      preserving exact Review 1 row counts for validation (2960-h) and test (2961-h).
    """
    if horizon not in (1, 6):
        raise ValueError(f"Expected horizon 1 or 6, got {horizon}.")

    X_all = build_features(df, include_random_controls=include_random_controls)
    n_train, n_val, n_test = chronological_split_indices(len(df))

    splits_raw = {
        "train": (0, n_train),
        "val": (n_train, n_train + n_val),
        "test": (n_train + n_val, len(df)),
    }

    result: Dict[str, object] = {
        "feature_cols": list(X_all.columns),
        "horizon": horizon,
    }

    for split_name, (start_idx, end_idx) in splits_raw.items():
        df_slice = df.iloc[start_idx:end_idx].copy()
        X_slice = X_all.iloc[start_idx:end_idx].copy()

        # Construct target strictly inside this split and drop final `horizon` rows
        target = df_slice["Appliances"].astype(float).shift(-horizon)
        valid_target_mask = target.notna()

        # Drop rows lacking full historical lookback (affects train start only)
        valid_feature_mask = X_slice.notna().all(axis=1)
        keep_mask = valid_target_mask & valid_feature_mask

        X_split = X_slice.loc[keep_mask].reset_index(drop=True)
        y_split = target.loc[keep_mask].reset_index(drop=True)
        meta_split = pd.DataFrame(
            {
                "date": df_slice.loc[keep_mask, "date"].values,
                "target_date": (
                    df_slice.loc[keep_mask, "date"]
                    + pd.Timedelta(minutes=10 * horizon)
                ).values,
                "Appliances_t": df_slice.loc[keep_mask, "Appliances"].astype(float).values,
                "target": y_split.values,
            }
        )

        result[f"X_{split_name}"] = X_split
        result[f"y_{split_name}"] = y_split
        result[f"meta_{split_name}"] = meta_split

    return result


def build_single_timestamp_features(
    df: pd.DataFrame,
    origin_idx: int,
    feature_cols: List[str],
) -> pd.DataFrame:
    """Compute features for a single forecast origin `origin_idx` without future access.

    Slices df strictly up to `origin_idx` (inclusive), using at most the
    required historical lookback window, so future rows after `origin_idx`
    never enter the feature computation.
    """
    if origin_idx < MAX_LOOKBACK_STEPS:
        raise ValueError(
            f"origin_idx={origin_idx} has insufficient history "
            f"(requires >= {MAX_LOOKBACK_STEPS} prior steps)."
        )
    if origin_idx >= len(df):
        raise IndexError(f"origin_idx={origin_idx} out of bounds for len(df)={len(df)}.")

    start_idx = max(0, origin_idx - MAX_LOOKBACK_STEPS - 10)
    history_slice = df.iloc[start_idx : origin_idx + 1].reset_index(drop=True)
    features_df = build_features(history_slice, include_random_controls=False)
    row = features_df.iloc[[-1]][feature_cols].copy()
    if row.isna().any().any():
        raise ValueError("Computed feature row contains NaNs.")
    return row
