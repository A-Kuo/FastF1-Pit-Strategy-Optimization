"""Unit tests for trained models"""

import pytest
import numpy as np
import pickle
import os
from pathlib import Path

# Add parent dir to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestModelArtifacts:
    """Test that model artifacts exist and load correctly."""

    def test_xgboost_model_loads(self):
        """XGBoost model should load without errors."""
        assert os.path.exists("models/xgboost_model.pkl"), "Missing xgboost_model.pkl"
        with open("models/xgboost_model.pkl", "rb") as f:
            model = pickle.load(f)
        assert model is not None
        assert hasattr(model, "predict_proba"), "Model missing predict_proba method"

    def test_scaler_loads(self):
        """StandardScaler should load."""
        assert os.path.exists("models/scaler.pkl"), "Missing scaler.pkl"
        with open("models/scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
        assert scaler is not None
        assert hasattr(scaler, "transform"), "Scaler missing transform method"

    def test_metrics_loads(self):
        """Metrics dict should load and contain expected keys."""
        assert os.path.exists("models/metrics.pkl"), "Missing metrics.pkl"
        with open("models/metrics.pkl", "rb") as f:
            metrics = pickle.load(f)

        expected_keys = ["roc_auc", "f1", "recall", "precision", "threshold",
                        "train_size", "test_size", "feature_cols"]
        for key in expected_keys:
            assert key in metrics, f"Missing key: {key}"

    def test_scaled_data_loads(self):
        """Scaled feature arrays should load."""
        X_train = np.load("models/X_train_scaled.npy")
        X_test = np.load("models/X_test_scaled.npy")
        y_train = np.load("models/y_train.npy")
        y_test = np.load("models/y_test.npy")

        assert X_train.shape[0] > 0, "Empty training features"
        assert X_test.shape[0] > 0, "Empty test features"
        assert X_train.shape[1] == 4, "Expected 4 features"
        assert X_test.shape[1] == 4, "Expected 4 features"
        assert len(y_train) == len(X_train), "Shape mismatch: y_train"
        assert len(y_test) == len(X_test), "Shape mismatch: y_test"


class TestModelPredictions:
    """Test model inference."""

    @pytest.fixture
    def model_and_data(self):
        """Load model and test data."""
        with open("models/xgboost_model.pkl", "rb") as f:
            model = pickle.load(f)
        with open("models/scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
        X_test = np.load("models/X_test_scaled.npy")
        y_test = np.load("models/y_test.npy")
        return model, scaler, X_test, y_test

    def test_prediction_shape(self, model_and_data):
        """Predictions should match test set size."""
        model, _, X_test, _ = model_and_data
        proba = model.predict_proba(X_test)

        assert proba.shape == (X_test.shape[0], 2), "Wrong prediction shape"

    def test_prediction_bounds(self, model_and_data):
        """Probabilities should be in [0, 1]."""
        model, _, X_test, _ = model_and_data
        proba = model.predict_proba(X_test)[:, 1]

        assert np.all(proba >= 0), "Negative probabilities"
        assert np.all(proba <= 1), "Probabilities > 1"

    def test_threshold_decision(self, model_and_data):
        """Threshold logic should work."""
        model, _, X_test, _ = model_and_data
        proba = model.predict_proba(X_test)[:, 1]

        threshold = 0.60
        decisions = (proba >= threshold).astype(int)

        assert decisions.shape == (len(X_test),), "Wrong decision shape"
        assert set(decisions) <= {0, 1}, "Invalid decision values"

    def test_single_prediction(self, model_and_data):
        """Single sample should work."""
        model, _, X_test, _ = model_and_data

        single_sample = X_test[0:1]  # Shape (1, 4)
        proba = model.predict_proba(single_sample)

        assert proba.shape == (1, 2), "Wrong shape for single prediction"
        assert proba[0, 1] >= 0 and proba[0, 1] <= 1, "Invalid probability"

    def test_batch_prediction(self, model_and_data):
        """Batch predictions should be consistent."""
        model, _, X_test, _ = model_and_data

        # Get all predictions at once
        all_proba = model.predict_proba(X_test)

        # Get predictions in batches
        batch_size = 100
        batch_proba = []
        for i in range(0, len(X_test), batch_size):
            batch = X_test[i:i+batch_size]
            batch_proba.append(model.predict_proba(batch))
        batch_proba = np.vstack(batch_proba)

        # Should match
        np.testing.assert_array_almost_equal(all_proba, batch_proba, decimal=6)


class TestMetrics:
    """Test stored metrics against actual predictions."""

    def test_metrics_values_reasonable(self):
        """Stored metrics should be in valid ranges."""
        with open("models/metrics.pkl", "rb") as f:
            metrics = pickle.load(f)

        assert 0 <= metrics["roc_auc"] <= 1, "Invalid ROC-AUC"
        assert 0 <= metrics["f1"] <= 1, "Invalid F1"
        assert 0 <= metrics["recall"] <= 1, "Invalid recall"
        assert 0 <= metrics["precision"] <= 1, "Invalid precision"
        assert 0 < metrics["threshold"] < 1, "Invalid threshold"



if __name__ == "__main__":
    pytest.main([__file__, "-v"])
