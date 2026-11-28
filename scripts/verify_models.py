"""Verify saved joblib model artifacts load cleanly and reproduce exact test metrics."""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features import (
    build_single_timestamp_features,
    load_and_validate_data,
    prepare_forecast_datasets,
)


def invert_predictions(preds: np.ndarray, target_transform: str) -> np.ndarray:
    arr = np.asarray(preds, dtype=float)
    if target_transform == "log1p":
        arr = np.expm1(arr)
    return np.clip(arr, 0.0, None)


def main() -> None:
    df = load_and_validate_data(PROJECT_ROOT / "data" / "energydata_complete.csv")

    for h in (1, 6):
        artifact_path = PROJECT_ROOT / "models" / f"model_h{h}.joblib"
        if not artifact_path.exists():
            raise FileNotFoundError(f"Missing model artifact: {artifact_path}")

        artifact = joblib.load(artifact_path)
        assert artifact["horizon"] == h
        feature_cols = artifact["feature_cols"]
        model = artifact["model"]
        t_form = artifact["target_transform"]

        ds = prepare_forecast_datasets(df, horizon=h, include_random_controls=False)
        assert list(ds["X_test"].columns) == feature_cols, "Feature order mismatch!"

        preds = invert_predictions(model.predict(ds["X_test"]), t_form)
        mae = float(mean_absolute_error(ds["y_test"], preds))
        rmse = float(np.sqrt(mean_squared_error(ds["y_test"], preds)))
        r2 = float(r2_score(ds["y_test"], preds))

        saved_test = artifact["test_metrics"]
        assert np.isclose(mae, saved_test["MAE"], atol=1e-9), f"MAE mismatch: {mae} vs {saved_test['MAE']}"
        assert np.isclose(rmse, saved_test["RMSE"], atol=1e-9), f"RMSE mismatch: {rmse} vs {saved_test['RMSE']}"
        assert np.isclose(r2, saved_test["R2"], atol=1e-9), f"R2 mismatch: {r2} vs {saved_test['R2']}"

        # Also check single-timestamp inference parity on test rows
        n_train_val = len(df) - (len(ds["X_test"]) + h)
        for offset in (0, 500, len(ds["X_test"]) - 1):
            origin_idx = n_train_val + offset
            single_x = build_single_timestamp_features(df, origin_idx, feature_cols)
            single_pred = invert_predictions(model.predict(single_x), t_form)[0]
            assert np.isclose(single_pred, preds[offset], rtol=1e-12, atol=1e-12)

        print(
            f"[OK] model_h{h}.joblib ({artifact['model_name']}, {t_form}): "
            f"Test MAE={mae:.2f} Wh | RMSE={rmse:.2f} Wh | R2={r2:.4f} | Single-row parity PASSED"
        )


if __name__ == "__main__":
    main()
