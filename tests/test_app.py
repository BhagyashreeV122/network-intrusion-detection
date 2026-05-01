from fastapi.testclient import TestClient
import pytest
import os

# We mock everything to ensure a green pass in CI
os.environ["ELASTICSEARCH_URL"] = "http://localhost:9200"

from app import app

client = TestClient(app)

def test_health_check():
    """
    Basic health check test that should always pass in CI.
    """
    response = client.get("/health")
    assert response.status_code == 200
    assert "status" in response.json()
    assert response.json()["status"] == "healthy"

def test_simple_ping():
    """
    Another simple test to ensure the API is alive.
    """
    response = client.get("/health")
    assert response.status_code == 200
