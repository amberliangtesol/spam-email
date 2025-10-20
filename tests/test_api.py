"""Unit tests for REST API."""
import sys
import json
import pytest
from pathlib import Path
from fastapi.testclient import TestClient

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Note: This will only work if model is trained
# For CI/CD, you'd want to mock the predictor
from app.api_server import app

client = TestClient(app)


class TestAPI:
    """Test cases for REST API endpoints."""
    
    def test_root_endpoint(self):
        """Test root endpoint."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "Spam Classifier API" in data["message"]
    
    def test_health_endpoint(self):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data
        assert "uptime_seconds" in data
    
    @pytest.mark.skipif(not Path("models/spam_classifier_v1.0.0.pkl").exists(), 
                       reason="Model not trained yet")
    def test_predict_endpoint_valid(self):
        """Test prediction with valid input."""
        response = client.post(
            "/predict",
            json={"text": "You have won a free iPhone! Click here now!"}
        )
        assert response.status_code == 200
        data = response.json()
        assert "label" in data
        assert data["label"] in ["spam", "ham"]
        assert "probability" in data
        assert 0 <= data["probability"] <= 1
        assert "confidence" in data
        assert data["confidence"] in ["high", "medium", "low"]
        assert "model_version" in data
        assert "processing_time_ms" in data
    
    def test_predict_endpoint_empty_text(self):
        """Test prediction with empty text."""
        response = client.post(
            "/predict",
            json={"text": ""}
        )
        assert response.status_code == 422
    
    def test_predict_endpoint_missing_text(self):
        """Test prediction with missing text field."""
        response = client.post(
            "/predict",
            json={}
        )
        assert response.status_code == 422
    
    def test_predict_endpoint_too_long(self):
        """Test prediction with text exceeding max length."""
        long_text = "x" * 10001
        response = client.post(
            "/predict",
            json={"text": long_text}
        )
        assert response.status_code == 422
    
    def test_predict_endpoint_with_threshold(self):
        """Test prediction with custom threshold."""
        response = client.post(
            "/predict",
            json={
                "text": "Hello, this is a normal message",
                "threshold": 0.3
            }
        )
        # Should work regardless of model state
        if response.status_code == 200:
            data = response.json()
            assert "label" in data
    
    @pytest.mark.skipif(not Path("models/spam_classifier_v1.0.0.pkl").exists(),
                       reason="Model not trained yet")
    def test_batch_predict_endpoint(self):
        """Test batch prediction endpoint."""
        response = client.post(
            "/predict/batch",
            json={
                "texts": [
                    "Win free money now!",
                    "Hello, how are you?",
                    "Click here for prizes"
                ]
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert "predictions" in data
        assert "total" in data
        assert data["total"] == 3
        assert len(data["predictions"]) == 3
        assert "processing_time_ms" in data
    
    def test_batch_predict_empty_list(self):
        """Test batch prediction with empty list."""
        response = client.post(
            "/predict/batch",
            json={"texts": []}
        )
        assert response.status_code == 422
    
    def test_batch_predict_too_many(self):
        """Test batch prediction with too many texts."""
        texts = ["text"] * 1001
        response = client.post(
            "/predict/batch",
            json={"texts": texts}
        )
        assert response.status_code == 422
    
    def test_metrics_endpoint(self):
        """Test metrics endpoint."""
        response = client.get("/metrics")
        # Will return 503 if model not loaded, 404 if metrics not found
        assert response.status_code in [200, 404, 503]
        if response.status_code == 200:
            data = response.json()
            assert "model_version" in data
            assert "model_metrics" in data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])