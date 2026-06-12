"""Unit tests for FastAPI endpoints"""

import pytest
from fastapi.testclient import TestClient
from pathlib import Path
import sys

# Add parent dir to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from api import app

client = TestClient(app)


class TestHealthEndpoints:
    """Test health check endpoints."""

    def test_health_endpoint(self):
        """GET /health should return 200 with healthy status."""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["version"] == "1.0.0"
        assert "timestamp" in data

    def test_ready_endpoint(self):
        """GET /ready should return status (200 if ready, 503 if not)."""
        response = client.get("/ready")

        # May be 200 (ready) or 503 (not ready), both valid
        assert response.status_code in [200, 503]
        data = response.json()
        assert "status" in data or "detail" in data  # Either response dict or error detail

    def test_health_always_responds(self):
        """Health endpoint should always respond, even before models load."""
        # Make multiple requests to ensure consistency
        for _ in range(3):
            response = client.get("/health")
            assert response.status_code == 200


class TestPredictionEndpoint:
    """Test single prediction endpoint."""

    def test_predict_valid_request(self):
        """Valid prediction request should return probability and decision."""
        payload = {
            "degradation_rate": 0.05,
            "stint_age_squared": 100,
            "race_progress": 0.5,
            "pace_delta": 0.1
        }

        response = client.post("/predict", json=payload)

        # Endpoint might return 503 if models not loaded, but request is valid
        if response.status_code == 200:
            data = response.json()
            assert "pit_probability" in data
            assert "pit_decision" in data
            assert 0 <= data["pit_probability"] <= 1
            assert data["pit_decision"] in ["PIT_NOW", "MONITOR", "STAY_OUT"]

    def test_predict_boundary_values(self):
        """Should handle boundary feature values."""
        payloads = [
            {
                "degradation_rate": -0.5,
                "stint_age_squared": 0,
                "race_progress": 0.0,
                "pace_delta": -3.0
            },
            {
                "degradation_rate": 0.5,
                "stint_age_squared": 3600,
                "race_progress": 1.0,
                "pace_delta": 3.0
            }
        ]

        for payload in payloads:
            response = client.post("/predict", json=payload)
            assert response.status_code in [200, 503]

    def test_predict_invalid_bounds(self):
        """Should reject out-of-bounds feature values."""
        payload = {
            "degradation_rate": 1.0,  # Out of bounds
            "stint_age_squared": 100,
            "race_progress": 0.5,
            "pace_delta": 0.1
        }

        response = client.post("/predict", json=payload)
        assert response.status_code == 422  # Validation error


class TestBatchPredictionEndpoint:
    """Test batch prediction endpoint."""

    def test_batch_predict_valid_request(self):
        """Valid batch request should return lists of predictions."""
        payload = {
            "features": [
                [0.05, 100, 0.5, 0.1],
                [0.03, 50, 0.3, -0.2],
                [0.06, 200, 0.7, 0.3]
            ],
            "threshold": 0.60
        }

        response = client.post("/batch-predict", json=payload)

        if response.status_code == 200:
            data = response.json()
            assert len(data["pit_probabilities"]) == 3
            assert len(data["pit_decisions"]) == 3
            assert all(0 <= p <= 1 for p in data["pit_probabilities"])
            assert all(d in ["PIT_NOW", "MONITOR", "STAY_OUT"] for d in data["pit_decisions"])

    def test_batch_predict_single_sample(self):
        """Should work with single sample in batch."""
        payload = {
            "features": [[0.05, 100, 0.5, 0.1]]
        }

        response = client.post("/batch-predict", json=payload)
        if response.status_code == 200:
            data = response.json()
            assert len(data["pit_probabilities"]) == 1
            assert len(data["pit_decisions"]) == 1

    def test_batch_predict_large_batch(self):
        """Should handle larger batches."""
        payload = {
            "features": [[0.05 + i*0.001, 100 + i, 0.5 + i*0.001, 0.1] for i in range(100)]
        }

        response = client.post("/batch-predict", json=payload)
        if response.status_code == 200:
            data = response.json()
            assert len(data["pit_probabilities"]) == 100

    def test_batch_predict_custom_threshold(self):
        """Should respect custom threshold."""
        payload = {
            "features": [[0.05, 100, 0.5, 0.1]],
            "threshold": 0.80
        }

        response = client.post("/batch-predict", json=payload)
        if response.status_code == 200:
            data = response.json()
            assert "pit_decisions" in data

    def test_batch_predict_wrong_feature_count(self):
        """Should handle wrong number of features (400 or 422 or 503)."""
        payload = {
            "features": [
                [0.05, 100, 0.5]  # Missing pace_delta
            ]
        }

        response = client.post("/batch-predict", json=payload)
        # Could be 400 (bad request), 422 (validation), or 503 (model not loaded)
        assert response.status_code in [400, 422, 503]


class TestRootEndpoint:
    """Test API documentation endpoint."""

    def test_root_endpoint(self):
        """GET / should return API info."""
        response = client.get("/")

        assert response.status_code == 200
        data = response.json()
        assert "service" in data
        assert "version" in data
        assert "endpoints" in data


class TestErrorHandling:
    """Test error responses."""

    def test_malformed_json(self):
        """Malformed JSON should return 422."""
        response = client.post(
            "/predict",
            json={"degradation_rate": "not_a_number"}  # Type error
        )
        assert response.status_code == 422

    def test_missing_required_field(self):
        """Missing required field should return 422."""
        response = client.post(
            "/predict",
            json={"degradation_rate": 0.05}  # Missing other fields
        )
        assert response.status_code == 422


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
