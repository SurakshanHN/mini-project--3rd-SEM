"""Anti-leakage and feature engineering verification tests."""

from pathlib import Path

import numpy as np
import pandas as pd

from src.features import (
    FORBIDDEN_PRODUCTION_COLS,
    MAX_LOOKBACK_STEPS,
    build_features,
    build_single_timestamp_features,
    chronological_split_indices,
    get_feature_columns,
    load_and_validate_data,
    prepare_forecast_datasets,
)

DATA_PATH = Path("data/energydata_complete.csv")


def test_production_features_exclude_forbidden_columns():
    cols = get_feature_columns(include_random_controls=False)
    for forbidden in FORBIDDEN_PRODUCTION_COLS:
        assert forbidden not in cols
    assert len(cols) == len(set(cols))


def test_no_future_leakage_under_perturbation():
    """Perturbing all rows after t must leave features at t completely unchanged."""
    df = load_and_validate_data(DATA_PATH)
    probe_idx = 15000  # inside validation/test region

    baseline_feats = build_features(df, include_random_controls=False).iloc[probe_idx]

    df_perturbed = df.copy()
    numeric_cols = [c for c in df_perturbed.columns if c != "date"]
    df_perturbed.loc[probe_idx + 1 :, numeric_cols] = 99999.0

    perturbed_feats = build_features(df_perturbed, include_random_controls=False).iloc[probe_idx]
    pd.testing.assert_series_equal(baseline_feats, perturbed_feats)


def test_single_timestamp_matches_batch_features():
    """App single-row inference features must match batch training/eval features exactly."""
    df = load_and_validate_data(DATA_PATH)
    feature_cols = get_feature_columns(include_random_controls=False)
    batch = build_features(df, include_random_controls=False)

    for idx in (MAX_LOOKBACK_STEPS, 13814, 16774, 18000):
        single = build_single_timestamp_features(df, idx, feature_cols)
        np.testing.assert_allclose(single.iloc[0].values, batch.iloc[idx].values, rtol=1e-12, atol=1e-12)


def test_chronological_splits_match_review1_sizes():
    df = load_and_validate_data(DATA_PATH)
    n_train, n_val, n_test = chronological_split_indices(len(df))
    assert (n_train, n_val, n_test) == (13814, 2960, 2961)

    for h in (1, 6):
        ds = prepare_forecast_datasets(df, horizon=h, include_random_controls=False)
        assert len(ds["X_train"]) == n_train - MAX_LOOKBACK_STEPS - h
        assert len(ds["X_val"]) == n_val - h
        assert len(ds["X_test"]) == n_test - h
        assert not ds["X_train"].isna().any().any()
        assert not ds["X_val"].isna().any().any()
        assert not ds["X_test"].isna().any().any()
        # Ensure strict chronological separation (no date overlap across splits)
        assert ds["meta_train"]["target_date"].max() <= ds["meta_val"]["date"].min()
        assert ds["meta_val"]["target_date"].max() <= ds["meta_test"]["date"].min()
