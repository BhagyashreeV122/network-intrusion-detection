import torch
from fastapi.testclient import TestClient
import pandas as pd
import json

from app import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert "threshold" in response.json()

def test_predict_endpoint():
    # Load 2 rows from the test set for a quick integration test
    test_df = pd.read_csv("data/processed/test.csv", nrows=2)
    # Drop labels to simulate real inference request
    features_df = test_df.drop(['label', 'attack_cat'], axis=1)
    
    # Convert to list of dicts
    payload = {"data": features_df.to_dict(orient="records")}
    
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    
    predictions = response.json()
    assert len(predictions) == 2
    for p in predictions:
        assert "is_anomaly" in p
        assert "reconstruction_error" in p
        assert "rf_prediction" in p
        assert p["features_processed"] == 42
