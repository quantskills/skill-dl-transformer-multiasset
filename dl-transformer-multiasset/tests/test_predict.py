"""Tests for predict.py feature building functions."""
import os
import tempfile
from pathlib import Path

import pandas as pd
import pytest


def test_build_predict_features_returns_dataframe(monkeypatch):
    """Verify build_predict_features returns DataFrame with date/symbol columns."""
    from scripts.predict import build_predict_features

    # Set environment variables
    monkeypatch.setenv("PANDA_DATA_PREDICT_START", "20240101")
    monkeypatch.setenv("PANDA_DATA_PREDICT_END", "20240110")
    monkeypatch.setenv("PANDA_DATA_USERNAME", "test_user")
    monkeypatch.setenv("PANDA_DATA_PASSWORD", "test_pass")

    # Mock panda_data.init_token
    def mock_init_token(**kwargs):
        pass

    monkeypatch.setattr("panda_data.init_token", mock_init_token)

    # Mock list_commodity_symbols
    def mock_list_commodity_symbols():
        return ["RB", "AU"]

    monkeypatch.setattr("scripts.predict.list_commodity_symbols", mock_list_commodity_symbols)

    # Mock load_raw_panel to return fake data with enough lookback
    def mock_load_raw_panel(symbols, start, end):
        # Create data spanning the lookback buffer + predict period
        dates = []
        for i in range(100):
            dates.append(f"202311{i:02d}" if i < 30 else f"202312{i-29:02d}" if i < 60 else f"202401{i-59:02d}")

        rows = []
        for date in dates:
            for symbol in ["RB", "AU"]:
                rows.append({
                    "date": date,
                    "symbol": symbol,
                    "open": 3500.0,
                    "high": 3550.0,
                    "low": 3480.0,
                    "close": 3520.0,
                    "volume": 100000.0,
                    "amount": 352000000.0,
                    "open_interest": 80000.0,
                })
        return pd.DataFrame(rows)

    monkeypatch.setattr("scripts.predict.load_raw_panel", mock_load_raw_panel)

    # Mock add_engineered_features
    def mock_add_engineered_features(raw):
        df = raw.copy()
        # Add mock engineered features
        for col in ["mom_5", "mom_10", "mom_20", "mom_60", "rev_5", "rev_20", "vol_5"]:
            df[col] = 0.1
        return df

    monkeypatch.setattr("scripts.predict.add_engineered_features", mock_add_engineered_features)

    # Mock xsec_zscore
    def mock_xsec_zscore(df, exclude=None):
        return df

    monkeypatch.setattr("scripts.predict.xsec_zscore", mock_xsec_zscore)

    # Mock add_label
    def mock_add_label(df, horizon=5):
        df = df.copy()
        df["ret_5d"] = 0.01
        df["label"] = 0.5
        return df

    monkeypatch.setattr("scripts.predict.add_label", mock_add_label)

    # Mock parquet write
    def mock_to_parquet(path, index=False):
        pass

    # Call the function
    result = build_predict_features("20240101", "20240110")

    # Verify it returns a DataFrame
    assert isinstance(result, pd.DataFrame)

    # Verify it has date and symbol columns
    assert "date" in result.columns
    assert "symbol" in result.columns

    # Verify it has at least some rows
    assert len(result) > 0


def test_build_predict_features_writes_parquet(monkeypatch, tmp_path):
    """Verify build_predict_features writes predict_features.parquet."""
    from scripts.predict import build_predict_features

    # Set environment variables
    monkeypatch.setenv("PANDA_DATA_PREDICT_START", "20240101")
    monkeypatch.setenv("PANDA_DATA_PREDICT_END", "20240110")
    monkeypatch.setenv("PANDA_DATA_USERNAME", "test_user")
    monkeypatch.setenv("PANDA_DATA_PASSWORD", "test_pass")

    # Mock panda_data.init_token
    def mock_init_token(**kwargs):
        pass

    monkeypatch.setattr("panda_data.init_token", mock_init_token)

    # Mock list_commodity_symbols
    def mock_list_commodity_symbols():
        return ["RB", "AU"]

    monkeypatch.setattr("scripts.predict.list_commodity_symbols", mock_list_commodity_symbols)

    # Mock load_raw_panel
    def mock_load_raw_panel(symbols, start, end):
        rows = []
        for i in range(100):
            for symbol in ["RB", "AU"]:
                rows.append({
                    "date": f"{20240000 + i + 1:08d}",
                    "symbol": symbol,
                    "open": 3500.0,
                    "high": 3550.0,
                    "low": 3480.0,
                    "close": 3520.0,
                    "volume": 100000.0,
                    "amount": 352000000.0,
                    "open_interest": 80000.0,
                })
        return pd.DataFrame(rows)

    monkeypatch.setattr("scripts.predict.load_raw_panel", mock_load_raw_panel)

    # Mock add_engineered_features
    def mock_add_engineered_features(raw):
        df = raw.copy()
        for col in ["mom_5", "mom_10", "mom_20", "mom_60", "rev_5", "rev_20", "vol_5"]:
            df[col] = 0.1
        return df

    monkeypatch.setattr("scripts.predict.add_engineered_features", mock_add_engineered_features)

    # Mock xsec_zscore
    def mock_xsec_zscore(df, exclude=None):
        return df

    monkeypatch.setattr("scripts.predict.xsec_zscore", mock_xsec_zscore)

    # Mock add_label
    def mock_add_label(df, horizon=5):
        df = df.copy()
        df["ret_5d"] = 0.01
        df["label"] = 0.5
        return df

    monkeypatch.setattr("scripts.predict.add_label", mock_add_label)

    # Mock the production directory
    production_dir = tmp_path / "dl-transformer-multiasset-production" / "data"
    production_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(
        "scripts.predict.PRODUCTION_DATA_DIR",
        production_dir,
    )

    # Call the function
    result = build_predict_features("20240101", "20240110")

    # Verify parquet file was created
    output_file = production_dir / "predict_features.parquet"
    assert output_file.exists(), f"File {output_file} was not created"

    # Verify we can read it back
    loaded = pd.read_parquet(output_file)
    assert isinstance(loaded, pd.DataFrame)
    assert len(loaded) > 0


def test_missing_env_vars_raises(monkeypatch):
    """Verify raises RuntimeError if env vars missing."""
    from scripts.predict import build_predict_features

    # Don't set environment variables, or set only one
    monkeypatch.delenv("PANDA_DATA_PREDICT_START", raising=False)
    monkeypatch.delenv("PANDA_DATA_PREDICT_END", raising=False)

    # Should raise RuntimeError about missing environment variable
    with pytest.raises(RuntimeError):
        build_predict_features()


class TestIntegration:
    """Integration tests for the full prediction pipeline."""

    def test_full_pipeline_smoke(self, monkeypatch, tmp_path):
        """Test full pipeline: build_predict_features → run_prediction → evaluate_predictions.

        This smoke test validates:
        1. Features parquet written and not empty
        2. Predictions parquet written and not empty
        3. 12-column output schema correct
        4. data_version = "predict-v1"
        5. Evaluation returns None when forward returns are empty
        """
        from datetime import datetime, timedelta
        from scripts.predict import (
            build_predict_features,
            run_prediction,
            evaluate_predictions,
            PRODUCTION_DATA_DIR,
        )

        # Set environment variables for test period
        monkeypatch.setenv("PANDA_DATA_PREDICT_START", "20241201")
        monkeypatch.setenv("PANDA_DATA_PREDICT_END", "20250131")
        monkeypatch.setenv("PANDA_DATA_USERNAME", "test_user")
        monkeypatch.setenv("PANDA_DATA_PASSWORD", "test_pass")

        # Mock production directory to use temp directory
        mock_prod_dir = tmp_path / "production" / "data"
        mock_prod_dir.mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr("scripts.predict.PRODUCTION_DATA_DIR", mock_prod_dir)

        # Mock panda_data.init_token
        def mock_init_token(**kwargs):
            pass

        monkeypatch.setattr("panda_data.init_token", mock_init_token)

        # Mock list_commodity_symbols
        def mock_list_commodity_symbols():
            return ["RB", "AU", "CU"]

        monkeypatch.setattr("scripts.predict.list_commodity_symbols", mock_list_commodity_symbols)

        # Mock load_raw_panel to return realistic data with OHLCVA columns
        def mock_load_raw_panel(symbols, start, end):
            rows = []
            # Create 100 dates spanning the lookback period (starting Sep 1, 2024)
            base_date = datetime(2024, 9, 1)
            for i in range(100):
                date = base_date + timedelta(days=i)
                date_str = date.strftime("%Y%m%d")
                for symbol in symbols:
                    rows.append({
                        "date": date_str,
                        "symbol": symbol,
                        "open": 3500.0 + i * 10,
                        "high": 3550.0 + i * 10,
                        "low": 3480.0 + i * 10,
                        "close": 3520.0 + i * 10,
                        "volume": 100000.0 + i * 100,
                        "amount": 352000000.0 + i * 1000000,
                        "open_interest": 80000.0 + i * 100,
                    })
            return pd.DataFrame(rows)

        monkeypatch.setattr("scripts.predict.load_raw_panel", mock_load_raw_panel)

        # Mock add_engineered_features
        def mock_add_engineered_features(raw):
            df = raw.copy()
            for col in ["mom_5", "mom_10", "mom_20", "mom_60", "rev_5", "rev_20", "vol_5"]:
                df[col] = 0.1
            return df

        monkeypatch.setattr("scripts.predict.add_engineered_features", mock_add_engineered_features)

        # Mock xsec_zscore
        def mock_xsec_zscore(df, exclude=None):
            return df

        monkeypatch.setattr("scripts.predict.xsec_zscore", mock_xsec_zscore)

        # Mock add_label (set ret_5d to NaN to test empty forward returns case)
        def mock_add_label(df, horizon=5):
            df = df.copy()
            df["ret_5d"] = float('nan')  # Empty forward returns
            df["label"] = float('nan')
            return df

        monkeypatch.setattr("scripts.predict.add_label", mock_add_label)

        # Mock pick_device
        import torch
        def mock_pick_device():
            return torch.device("cpu")

        monkeypatch.setattr("scripts.predict.pick_device", mock_pick_device)
        monkeypatch.setattr("scripts.train.pick_device", mock_pick_device)

        # Mock factor.predict_fold to return predictions for multiple dates
        def mock_predict_fold(feature_df, ckpt_dir, device):
            predictions = []
            base_date = datetime(2024, 12, 1)
            for symbol in feature_df["symbol"].unique():
                for i in range(50):  # 50+ predictions per symbol
                    date = base_date + timedelta(days=i)
                    predictions.append({
                        "trade_date": date.strftime("%Y%m%d"),
                        "asset_type": "Future",
                        "symbol": symbol,
                        "factor_value": 0.5 + i * 0.01,
                    })
            return pd.DataFrame(predictions)

        monkeypatch.setattr("scripts.factor.predict_fold", mock_predict_fold)

        # Mock factor.score_and_signal
        def mock_score_and_signal(predictions, buy_q=0.1, sell_q=0.1):
            df = predictions.copy()
            df["score"] = 0.6
            df["rank"] = 1
            df["signal"] = "buy"
            df["confidence"] = 0.8
            return df

        monkeypatch.setattr("scripts.factor.score_and_signal", mock_score_and_signal)

        # Mock factor.emit_factor_table
        def mock_emit_factor_table(scored, update_time):
            df = scored.copy()
            df["factor_id"] = "transformer_v1"
            df["factor_name"] = "DL Transformer Score"
            df["update_time"] = update_time
            return df

        monkeypatch.setattr("scripts.factor.emit_factor_table", mock_emit_factor_table)

        # Mock backtest functions to return None (for evaluate_predictions to gracefully fail)
        def mock_load_real_dominant_and_daily(*args, **kwargs):
            raise Exception("Mock: forward returns not available")

        monkeypatch.setattr("scripts.predict.load_real_dominant_and_daily", mock_load_real_dominant_and_daily)

        # ─────────────────────────────────────────────────────────────────────
        # Step 1: Build predict features
        # ─────────────────────────────────────────────────────────────────────
        features = build_predict_features("20241201", "20250131")

        # Validate features
        assert isinstance(features, pd.DataFrame), "Features should be a DataFrame"
        assert len(features) > 0, "Features should not be empty"
        assert "date" in features.columns, "Features should have 'date' column"
        assert "symbol" in features.columns, "Features should have 'symbol' column"
        assert "data_version" in features.columns, "Features should have 'data_version' column"
        assert all(features["data_version"] == "predict-v1"), "data_version should be 'predict-v1'"

        # Verify features parquet written
        features_path = mock_prod_dir / "predict_features.parquet"
        assert features_path.exists(), "Features parquet should be written"
        loaded_features = pd.read_parquet(features_path)
        assert len(loaded_features) > 0, "Loaded features should not be empty"

        # ─────────────────────────────────────────────────────────────────────
        # Step 2: Run prediction
        # ─────────────────────────────────────────────────────────────────────
        predictions = run_prediction("20241201", "20250131")

        # Validate predictions
        assert isinstance(predictions, pd.DataFrame), "Predictions should be a DataFrame"
        assert len(predictions) > 0, "Predictions should not be empty"

        # Validate 12-column schema
        expected_cols = [
            "trade_date",
            "asset_type",
            "symbol",
            "factor_id",
            "factor_name",
            "factor_value",
            "score",
            "rank",
            "signal",
            "confidence",
            "data_version",
            "update_time",
        ]
        assert len(predictions.columns) == 12, f"Should have 12 columns, got {len(predictions.columns)}"
        for col in expected_cols:
            assert col in predictions.columns, f"Missing column: {col}"

        # Validate data_version
        assert all(predictions["data_version"] == "predict-v1"), "Predictions data_version should be 'predict-v1'"

        # Verify predictions parquet written
        predictions_path = mock_prod_dir / "predict.parquet"
        assert predictions_path.exists(), "Predictions parquet should be written"
        loaded_predictions = pd.read_parquet(predictions_path)
        assert len(loaded_predictions) > 0, "Loaded predictions should not be empty"

        # ─────────────────────────────────────────────────────────────────────
        # Step 3: Evaluate predictions
        # ─────────────────────────────────────────────────────────────────────
        evaluation = evaluate_predictions("20241201", "20250131")

        # Validate evaluation returns None when forward returns are unavailable
        assert evaluation is None, "Evaluation should return None when forward returns are unavailable"


class TestEvaluatePredictions:
    """Tests for evaluate_predictions() function."""

    def test_evaluate_skips_when_no_predict_parquet(self, monkeypatch, tmp_path):
        """Verify returns None if predict.parquet doesn't exist."""
        from scripts.predict import evaluate_predictions

        # Mock the production directory to use tmp_path
        mock_production_dir = tmp_path / "production" / "data"
        mock_production_dir.mkdir(parents=True, exist_ok=True)

        monkeypatch.setattr(
            "scripts.predict.PRODUCTION_DATA_DIR",
            mock_production_dir,
        )

        # Call evaluate_predictions with date range
        result = evaluate_predictions("20240101", "20240110")

        # Should return None since predict.parquet doesn't exist
        assert result is None

    def test_evaluate_returns_metrics_dict(self, monkeypatch, tmp_path):
        """Verify returns dict with 'research' and 'tradeable' keys when predict.parquet exists."""
        from scripts.predict import evaluate_predictions
        import json

        # Mock the production directory
        mock_production_dir = tmp_path / "production" / "data"
        mock_production_dir.mkdir(parents=True, exist_ok=True)

        # Create a minimal predict.parquet
        predict_df = pd.DataFrame({
            "trade_date": ["2024-01-02", "2024-01-03", "2024-01-04"],
            "symbol": ["RB", "RB", "RB"],
            "factor_value": [0.1, -0.2, 0.15],
            "signal": ["buy", "sell", "buy"],
        })
        predict_df.to_parquet(mock_production_dir / "predict.parquet", index=False)

        monkeypatch.setattr(
            "scripts.predict.PRODUCTION_DATA_DIR",
            mock_production_dir,
        )

        # Mock backtest functions to return minimal test data
        def mock_load_real_dominant_and_daily(symbols, start, end):
            # Return minimal dominant and daily data
            dominant = pd.DataFrame({
                "date": ["2024-01-02", "2024-01-03", "2024-01-04"],
                "underlying_symbol": ["RB", "RB", "RB"],
                "symbol": ["RB2401", "RB2401", "RB2402"],
            })
            daily = pd.DataFrame({
                "date": ["2024-01-02", "2024-01-03", "2024-01-04"],
                "symbol": ["RB2401", "RB2401", "RB2402"],
                "close": [3500.0, 3520.0, 3510.0],
            })
            return dominant, daily

        def mock_build_forward_returns(dominant, daily):
            return pd.DataFrame({
                "trade_date": ["2024-01-02", "2024-01-03"],
                "symbol": ["RB", "RB"],
                "contract_symbol": ["RB2401", "RB2401"],
                "forward_return": [0.005, -0.003],
            })

        def mock_build_tradeable_forward_returns(dominant, daily, data_lag, roll_cost_bps):
            return pd.DataFrame({
                "trade_date": ["2024-01-02"],
                "symbol": ["RB"],
                "contract_symbol": ["RB2401"],
                "forward_return": [0.004],
            })

        def mock_calculate_metrics(factor, forward, rollover_count):
            return {
                "Rank IC": 0.05,
                "ICIR": 0.1,
                "ARR(%)": 5.0,
                "MDD(%)": -10.0,
                "IC_by_horizon": {"1D": {"IC": 0.02, "RankIC": 0.03, "IC_CI95": "[0.01, 0.03]"}},
            }

        def mock_count_rollovers(dominant):
            return 1

        def mock_calculate_ic_by_horizon(factor, dominant, daily):
            return {"1D": {"IC": 0.02, "RankIC": 0.03, "IC_CI95": "[0.01, 0.03]"}}

        monkeypatch.setattr(
            "scripts.predict.load_real_dominant_and_daily",
            mock_load_real_dominant_and_daily,
        )
        monkeypatch.setattr(
            "scripts.predict.build_forward_returns",
            mock_build_forward_returns,
        )
        monkeypatch.setattr(
            "scripts.predict.build_tradeable_forward_returns",
            mock_build_tradeable_forward_returns,
        )
        monkeypatch.setattr(
            "scripts.predict.calculate_metrics",
            mock_calculate_metrics,
        )
        monkeypatch.setattr(
            "scripts.predict.count_rollovers",
            mock_count_rollovers,
        )
        monkeypatch.setattr(
            "scripts.predict.calculate_ic_by_horizon",
            mock_calculate_ic_by_horizon,
        )

        # Call evaluate_predictions
        result = evaluate_predictions("20240101", "20240110")

        # Should return a dict
        assert isinstance(result, dict), f"Expected dict but got {type(result)}"

        # Should have research and tradeable keys
        assert "research" in result, f"Missing 'research' key in result: {result.keys()}"
        assert "tradeable" in result, f"Missing 'tradeable' key in result: {result.keys()}"

        # Both should be dicts
        assert isinstance(result["research"], dict)
        assert isinstance(result["tradeable"], dict)

        # Verify predict_report.json was written
        report_path = mock_production_dir / "predict_report.json"
        assert report_path.exists(), f"predict_report.json not created at {report_path}"

        # Load and verify report content
        with open(report_path) as f:
            report = json.load(f)

        # Report should have main keys
        assert "predict_period" in report
        assert "research" in report
        assert "tradeable" in report
        assert "generated_at" in report


# ────────────────────────────────────────────────────────────────────────────
# CLI Tests
# ────────────────────────────────────────────────────────────────────────────


class TestCLI:
    """Test CLI argument parsing and dispatch."""

    def test_main_requires_step_argument(self, monkeypatch, capsys):
        """Verify main() raises SystemExit if no --step argument."""
        from scripts.predict import main
        import sys

        # Mock sys.argv to simulate CLI with no --step
        monkeypatch.setattr(sys, "argv", ["scripts/predict.py"])

        # Should exit with SystemExit
        with pytest.raises(SystemExit):
            main()

    def test_main_invalid_step_raises(self, monkeypatch, capsys):
        """Verify main() raises SystemExit if invalid --step value."""
        from scripts.predict import main
        import sys

        # Mock sys.argv to simulate CLI with invalid --step
        monkeypatch.setattr(sys, "argv", ["scripts/predict.py", "--step", "invalid"])

        # Should exit with SystemExit
        with pytest.raises(SystemExit):
            main()

    def test_main_features_step(self, monkeypatch):
        """Verify main() dispatches to build_predict_features for 'features' step."""
        from scripts.predict import main
        import sys

        # Set environment variables
        monkeypatch.setenv("PANDA_DATA_PREDICT_START", "20240101")
        monkeypatch.setenv("PANDA_DATA_PREDICT_END", "20240110")

        # Mock sys.argv
        monkeypatch.setattr(sys, "argv", ["scripts/predict.py", "--step", "features"])

        # Track if build_predict_features was called
        called = {"count": 0}

        def mock_build_predict_features(*args, **kwargs):
            called["count"] += 1
            return pd.DataFrame()

        monkeypatch.setattr("scripts.predict.build_predict_features", mock_build_predict_features)

        # Call main
        main()

        # Verify it was called
        assert called["count"] == 1

    def test_main_all_step(self, monkeypatch):
        """Verify main() dispatches all steps for 'all' option."""
        from scripts.predict import main
        import sys

        # Set environment variables
        monkeypatch.setenv("PANDA_DATA_PREDICT_START", "20240101")
        monkeypatch.setenv("PANDA_DATA_PREDICT_END", "20240110")

        # Mock sys.argv
        monkeypatch.setattr(sys, "argv", ["scripts/predict.py", "--step", "all"])

        # Track calls
        calls = {"features": 0, "predict": 0, "evaluate": 0}

        def mock_build_predict_features(*args, **kwargs):
            calls["features"] += 1
            return pd.DataFrame()

        def mock_run_prediction(*args, **kwargs):
            calls["predict"] += 1
            return pd.DataFrame()

        def mock_evaluate_predictions(*args, **kwargs):
            calls["evaluate"] += 1
            return {}

        monkeypatch.setattr("scripts.predict.build_predict_features", mock_build_predict_features)
        monkeypatch.setattr("scripts.predict.run_prediction", mock_run_prediction)
        monkeypatch.setattr("scripts.predict.evaluate_predictions", mock_evaluate_predictions)

        # Call main
        main()

        # Verify all were called
        assert calls["features"] == 1
        assert calls["predict"] == 1
        assert calls["evaluate"] == 1


class TestRunPrediction:
    """Test suite for run_prediction() function."""

    def test_run_prediction_missing_checkpoint_raises(self, tmp_path):
        """Verify FileNotFoundError if checkpoint missing."""
        from scripts.predict import run_prediction

        # Mock predict_features.parquet in production directory
        production_dir = tmp_path / "dl-transformer-multiasset-production" / "data"
        production_dir.mkdir(parents=True, exist_ok=True)

        # Create fake predict_features.parquet with minimal schema
        predict_features_df = pd.DataFrame({
            "date": ["20240101", "20240102"],
            "symbol": ["RB", "AU"],
            "mom_5": [0.1, 0.2],
            "mom_10": [0.15, 0.25],
            "mom_20": [0.12, 0.22],
            "mom_60": [0.11, 0.21],
            "rev_5": [0.05, 0.1],
            "rev_20": [0.08, 0.15],
            "vol_5": [0.02, 0.03],
        })
        features_path = production_dir / "predict_features.parquet"
        predict_features_df.to_parquet(features_path, index=False)

        # Mock the PRODUCTION_DATA_DIR and CHECKPOINT_DIR to use tmp_path
        import scripts.predict as predict_module
        original_prod_dir = predict_module.PRODUCTION_DATA_DIR
        original_ckpt_dir = predict_module.CHECKPOINT_DIR

        predict_module.PRODUCTION_DATA_DIR = production_dir
        # Set checkpoint dir to a non-existent location
        predict_module.CHECKPOINT_DIR = tmp_path / "nonexistent_checkpoints"

        try:
            # Should raise FileNotFoundError because checkpoint doesn't exist
            with pytest.raises(FileNotFoundError):
                run_prediction("20240101", "20240102")
        finally:
            # Restore original values
            predict_module.PRODUCTION_DATA_DIR = original_prod_dir
            predict_module.CHECKPOINT_DIR = original_ckpt_dir

    def test_run_prediction_outputs_12_columns(self, tmp_path, monkeypatch):
        """Verify 12-column output with data_version='predict-v1'."""
        from scripts.predict import run_prediction

        # Create mock checkpoint structure
        ckpt_dir = tmp_path / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        # Create a minimal mock checkpoint file with the expected name
        import torch
        checkpoint = {
            "model_state_dict": {},
            "cfg": {
                "LOOKBACK": 10,
                "PATCH_LEN": 4,
                "STRIDE": 2,
                "D_MODEL": 16,
                "N_HEADS": 2,
                "N_LAYERS": 1,
                "DROPOUT": 0.1,
            }
        }
        torch.save(checkpoint, ckpt_dir / "fold_2_best.pth")

        # Mock production directory
        production_dir = tmp_path / "dl-transformer-multiasset-production" / "data"
        production_dir.mkdir(parents=True, exist_ok=True)

        # Create fake predict_features.parquet
        dates = ["20240101", "20240102", "20240103"] * 2
        symbols = ["RB", "RB", "RB", "AU", "AU", "AU"]

        predict_features_df = pd.DataFrame({
            "date": dates,
            "symbol": symbols,
            "mom_5": [0.1] * 6,
            "mom_10": [0.15] * 6,
            "mom_20": [0.12] * 6,
            "mom_60": [0.11] * 6,
            "rev_5": [0.05] * 6,
            "rev_20": [0.08] * 6,
            "vol_5": [0.02] * 6,
        })
        features_path = production_dir / "predict_features.parquet"
        predict_features_df.to_parquet(features_path, index=False)

        # Mock modules and directories
        import scripts.predict as predict_module
        original_prod_dir = predict_module.PRODUCTION_DATA_DIR
        original_ckpt_dir = predict_module.CHECKPOINT_DIR

        predict_module.PRODUCTION_DATA_DIR = production_dir
        predict_module.CHECKPOINT_DIR = ckpt_dir

        # Mock factor functions
        def mock_predict_fold(feature_df, ckpt_dir, device=None):
            # Return mock predictions
            return pd.DataFrame({
                "trade_date": ["20240101", "20240102", "20240103"],
                "asset_type": ["future", "future", "future"],
                "symbol": ["RB", "RB", "AU"],
                "factor_value": [0.5, -0.3, 0.8],
            })

        def mock_score_and_signal(predictions):
            df = predictions.copy()
            df["score"] = 0.5
            df["rank"] = 1
            df["signal"] = "hold"
            return df

        def mock_emit_factor_table(scored, update_time):
            df = scored.copy()
            df["factor_id"] = "DLTX"
            df["factor_name"] = "Transformer多资产联合建模"
            df["confidence"] = 1.0
            df["data_version"] = "predict-v1"
            df["update_time"] = update_time
            return df[[
                "trade_date",
                "asset_type",
                "symbol",
                "factor_id",
                "factor_name",
                "factor_value",
                "score",
                "rank",
                "signal",
                "confidence",
                "data_version",
                "update_time",
            ]]

        monkeypatch.setattr("scripts.factor.predict_fold", mock_predict_fold)
        monkeypatch.setattr("scripts.factor.score_and_signal", mock_score_and_signal)
        monkeypatch.setattr("scripts.factor.emit_factor_table", mock_emit_factor_table)

        try:
            # Call run_prediction
            result = run_prediction("20240101", "20240103")

            # Verify it returns a DataFrame
            assert isinstance(result, pd.DataFrame)

            # Verify exact 12 columns
            assert len(result.columns) == 12, f"Expected 12 columns, got {len(result.columns)}: {list(result.columns)}"

            expected_columns = [
                "trade_date",
                "asset_type",
                "symbol",
                "factor_id",
                "factor_name",
                "factor_value",
                "score",
                "rank",
                "signal",
                "confidence",
                "data_version",
                "update_time",
            ]
            assert list(result.columns) == expected_columns

            # Verify data_version is "predict-v1"
            assert (result["data_version"] == "predict-v1").all()

            # Verify predict.parquet was written
            output_file = production_dir / "predict.parquet"
            assert output_file.exists()

            # Verify we can read it back
            loaded = pd.read_parquet(output_file)
            assert isinstance(loaded, pd.DataFrame)
            assert len(loaded) > 0
            assert len(loaded.columns) == 12

        finally:
            # Restore original values
            predict_module.PRODUCTION_DATA_DIR = original_prod_dir
            predict_module.CHECKPOINT_DIR = original_ckpt_dir
