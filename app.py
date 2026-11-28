"""Streamlit dashboard for Household Appliance Energy Forecasting (Review 3).

Retrospective demonstration over the chronological test split (May 7 - May 27, 2016).
Uses the exact same feature-engineering logic (`src/features.py`) and saved
`joblib` artifacts (`models/model_h1.joblib`, `models/model_h6.joblib`) produced
by the Review 2 pipeline.
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features import (
    MAX_LOOKBACK_STEPS,
    build_single_timestamp_features,
    chronological_split_indices,
    load_and_validate_data,
)

DATA_PATH = PROJECT_ROOT / "data" / "energydata_complete.csv"
MODEL_PATHS = {
    1: PROJECT_ROOT / "models" / "model_h1.joblib",
    6: PROJECT_ROOT / "models" / "model_h6.joblib",
}


@st.cache_data(show_spinner=False)
def get_dataset() -> pd.DataFrame:
    return load_and_validate_data(DATA_PATH)


@st.cache_resource(show_spinner=False)
def load_model_artifact(horizon: int) -> dict:
    path = MODEL_PATHS[horizon]
    if not path.exists():
        raise FileNotFoundError(
            f"Saved model artifact not found at {path}. Run 02_features_models_eval.ipynb first."
        )
    return joblib.load(path)


def invert_prediction(raw_pred: float, target_transform: str) -> float:
    val = float(np.expm1(raw_pred)) if target_transform == "log1p" else float(raw_pred)
    return max(0.0, val)


def main() -> None:
    st.set_page_config(
        page_title="Household Appliance Energy Forecasting",
        page_icon="⚡",
        layout="wide",
    )

    st.title("Household Appliance Energy Forecasting")
    st.caption(
        "Predicting short-term household appliance consumption using historical energy and environmental data."
    )
    st.info(
        "**Retrospective Test-Period Demonstration (May 7 – May 27, 2016):** "
        "At forecast origin **t**, the model uses strictly observations at or before **t**. "
        "The actual consumption at **t + h** is shown below solely for retrospective evaluation "
        "and is never available at prediction time."
    )

    df = get_dataset()
    n_train, n_val, n_test = chronological_split_indices(len(df))
    test_start_idx = n_train + n_val

    # Sidebar Controls
    st.sidebar.header("Forecast Configuration")
    horizon_choice = st.sidebar.radio(
        "Forecast Horizon",
        options=[1, 6],
        format_func=lambda h: "10 minutes ahead (h = 1)" if h == 1 else "1 hour ahead (h = 6)",
        index=1,
    )

    artifact = load_model_artifact(horizon_choice)
    feature_cols = artifact["feature_cols"]
    model = artifact["model"]
    target_transform = artifact["target_transform"]
    thresholds = artifact.get("thresholds", {})
    high_usage_threshold = float(thresholds.get("high_usage_wh", 200.0))
    spike_threshold = float(thresholds.get("spike_diagnostic_wh", 300.0))

    # Valid test-period forecast origins (need horizon steps remaining in test split)
    valid_end_idx = len(df) - horizon_choice - 1
    valid_indices = list(range(max(test_start_idx, MAX_LOOKBACK_STEPS), valid_end_idx + 1))
    test_dates_series = df.loc[valid_indices, "date"]

    unique_days = sorted(test_dates_series.dt.date.unique())
    default_day_idx = min(2, len(unique_days) - 1)  # default to May 9, 2016 if present
    selected_day = st.sidebar.selectbox(
        "Select Test Date",
        options=unique_days,
        index=default_day_idx,
        format_func=lambda d: d.strftime("%a, %d %b %Y"),
    )

    day_mask = test_dates_series.dt.date == selected_day
    day_indices = [idx for idx, m in zip(valid_indices, day_mask) if m]
    day_timestamps = df.loc[day_indices, "date"].tolist()

    # Default near evening peak (~18:00) if available on that day
    default_time_idx = 0
    for i, ts in enumerate(day_timestamps):
        if ts.hour == 18 and ts.minute == 0:
            default_time_idx = i
            break

    selected_origin_idx = st.sidebar.selectbox(
        "Select Forecast Origin Time (t)",
        options=day_indices,
        index=default_time_idx,
        format_func=lambda idx: df.loc[idx, "date"].strftime("%H:%M (%Y-%m-%d)"),
    )

    origin_time = df.loc[selected_origin_idx, "date"]
    target_time = origin_time + pd.Timedelta(minutes=10 * horizon_choice)
    target_idx = selected_origin_idx + horizon_choice

    # Build features strictly using observations <= selected_origin_idx
    X_row = build_single_timestamp_features(df, selected_origin_idx, feature_cols)
    raw_pred = float(model.predict(X_row)[0])
    ml_forecast_wh = invert_prediction(raw_pred, target_transform)

    persistence_wh = float(df.loc[selected_origin_idx, "Appliances"])
    actual_wh = float(df.loc[target_idx, "Appliances"])

    ml_abs_err = abs(actual_wh - ml_forecast_wh)
    pers_abs_err = abs(actual_wh - persistence_wh)

    # Sidebar Model Summary
    st.sidebar.markdown("---")
    st.sidebar.subheader("Loaded Model Artifact")
    st.sidebar.write(f"**Model:** {artifact['model_name']}")
    st.sidebar.write(f"**Target Transform:** `{target_transform}`")
    st.sidebar.write(f"**Features:** {len(feature_cols)}")
    st.sidebar.write(
        f"**Test MAE:** {artifact['test_metrics']['MAE']:.2f} Wh "
        f"(Persistence: {artifact['persistence_test_mae']:.2f} Wh)"
    )
    st.sidebar.write(f"**Test Skill vs Persistence:** {artifact['test_skill_vs_persistence']:+.4f}")

    # Forecast Header & Status Banner
    st.markdown(
        f"**Forecast Origin (`t`):** `{origin_time:%Y-%m-%d %H:%M}` &nbsp;|&nbsp; "
        f"**Target Timestamp (`t + {horizon_choice}`):** `{target_time:%Y-%m-%d %H:%M}` &nbsp;|&nbsp; "
        f"**Horizon:** `{horizon_choice * 10} minutes`"
    )

    if ml_forecast_wh >= spike_threshold:
        st.warning(
            f"**High-Usage Window Warning (Elevated Spike Range):** The ML forecast is "
            f"**{ml_forecast_wh:.1f} Wh**, exceeding the diagnostic high-consumption threshold "
            f"({spike_threshold:.0f} Wh). Households may consider shifting discretionary high-draw "
            f"appliances if schedule flexibility permits (descriptive load-awareness indicator only; "
            f"no measured savings or tariff amounts are claimed)."
        )
    elif ml_forecast_wh >= high_usage_threshold:
        st.warning(
            f"**High-Usage Window Warning:** The ML forecast is **{ml_forecast_wh:.1f} Wh**, "
            f"above the training 90th-percentile threshold ({high_usage_threshold:.1f} Wh). "
            f"Consider staggering flexible appliance runs if practical (descriptive indicator only)."
        )
    else:
        st.success(
            f"**Normal Usage Window:** The ML forecast is **{ml_forecast_wh:.1f} Wh**, "
            f"below the training 90th-percentile high-usage threshold ({high_usage_threshold:.1f} Wh)."
        )

    # Metric Cards
    col1, col2, col3, col4 = st.columns(4)
    col1.metric(
        label="ML Forecast (Wh)",
        value=f"{ml_forecast_wh:.1f} Wh",
        delta=f"{ml_forecast_wh - persistence_wh:+.1f} vs Persistence",
        delta_color="off",
    )
    col2.metric(
        label="Persistence Forecast y(t)",
        value=f"{persistence_wh:.1f} Wh",
    )
    col3.metric(
        label="Actual at t + h (Retrospective)",
        value=f"{actual_wh:.1f} Wh",
    )
    col4.metric(
        label="Point Absolute Error (ML vs Pers.)",
        value=f"{ml_abs_err:.1f} Wh",
        delta=f"{ml_abs_err - pers_abs_err:+.1f} Wh vs Pers. ({pers_abs_err:.1f} Wh)",
        delta_color="inverse",
    )

    # Comparison Chart: Observed History (up to t) + Point Forecasts at t+h
    st.subheader("Recent Observed History & Forecast Comparison")
    history_window_steps = 72  # past 12 hours up to origin t
    hist_start_idx = max(0, selected_origin_idx - history_window_steps)
    hist_df = df.iloc[hist_start_idx : selected_origin_idx + 1].copy()

    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(
        hist_df["date"],
        hist_df["Appliances"],
        color="#1f77b4",
        linewidth=1.8,
        marker="o",
        markersize=2.5,
        label="Observed History (<= t)",
    )
    ax.axvline(origin_time, color="#6b7280", linestyle=":", linewidth=1.3, label="Forecast Origin (t)")
    ax.axhline(
        high_usage_threshold,
        color="#f59e0b",
        linestyle="--",
        linewidth=1.1,
        alpha=0.85,
        label=f"90th-Pct High-Usage Threshold ({high_usage_threshold:.0f} Wh)",
    )

    # Connect origin t to forecast points at t+h
    ax.plot([origin_time, target_time], [persistence_wh, ml_forecast_wh], color="#2ca02c", linestyle="-", linewidth=1.4, alpha=0.7)
    ax.plot([origin_time, target_time], [persistence_wh, persistence_wh], color="#ff7f0e", linestyle="--", linewidth=1.2, alpha=0.7)

    ax.scatter([target_time], [ml_forecast_wh], color="#2ca02c", s=110, zorder=5, label=f"ML Forecast ({ml_forecast_wh:.1f} Wh)")
    ax.scatter([target_time], [persistence_wh], color="#ff7f0e", marker="s", s=95, zorder=5, label=f"Persistence ({persistence_wh:.1f} Wh)")
    ax.scatter([target_time], [actual_wh], color="#dc2626", marker="D", s=95, zorder=6, label=f"Actual Retrospective ({actual_wh:.1f} Wh)")

    ax.set_title(
        f"Appliances Consumption History & {horizon_choice * 10}-Min Forecast at {target_time:%Y-%m-%d %H:%M}",
        fontweight="bold",
    )
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Appliances Energy (Wh)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="upper left", fontsize=9, ncol=2)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # Context & Explanation Expander
    with st.expander("Input Context at Forecast Origin (t) & Methodology Notes"):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Key Features at Origin `t`**")
            summary_feats = {
                "Appliances at t (Wh)": float(X_row["appliance_lag_0"].iloc[0]),
                "Appliances at t-1 (Wh)": float(X_row["appliance_lag_1"].iloc[0]),
                "Past 1h Rolling Mean (Wh)": round(float(X_row["appliance_roll_mean_1h"].iloc[0]), 2),
                "Past 6h Rolling Mean (Wh)": round(float(X_row["appliance_roll_mean_6h"].iloc[0]), 2),
                "24h Ago Lag t-144 (Wh)": float(X_row["appliance_lag_144"].iloc[0]),
                "Mean Indoor Temp (°C)": round(float(X_row["indoor_temp_mean"].iloc[0]), 2),
                "Outdoor Temp T_out (°C)": round(float(X_row["T_out"].iloc[0]), 2),
                "Indoor - Outdoor Temp Diff (°C)": round(float(X_row["temp_diff_in_out"].iloc[0]), 2),
            }
            st.json(summary_feats)
        with c2:
            st.markdown("**Anti-Leakage & Scope Notes**")
            st.markdown(
                "- Features are generated via `src/features.py` using strictly observations at or before `t`.\n"
                "- `lights`, `rv1`, and `rv2` are excluded from the production model.\n"
                "- `Actual at t + h` is shown purely for retrospective evaluation on the held-out UCI test period.\n"
                "- Dataset covers one house in Stambruges, Belgium (Jan–May 2016) with no tariff pricing."
            )


if __name__ == "__main__":
    main()
