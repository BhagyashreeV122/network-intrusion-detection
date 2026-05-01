import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import torch
import numpy as np

# Mock the models BEFORE importing the app
with patch('joblib.load') as mock_joblib, \
     patch('torch.load') as mock_torch_load, \
     patch('builtins.open', create=True) as mock_open:
    
    # Setup mock returns
    mock_open.return_value.__enter__.return_value.read.return_value = "0.1"
    mock_joblib.return_value = MagicMock()
    
    from app import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_predict_endpoint():
    # Mock the internal logic of the predict function to avoid model dependencies
    sample_data = [
        {
            "dur": 0.000011, "proto": "udp", "service": "-", "state": "INT",
            "spkts": 2, "dpkts": 0, "sbytes": 496, "dbytes": 0, "rate": 90909.09,
            "sttl": 254, "dttl": 0, "sload": 180363632.0, "dload": 0.0,
            "sloss": 0, "dloss": 0, "sinpkt": 0.011, "dinpkt": 0.0,
            "sjit": 0.0, "djit": 0.0, "swin": 0, "stcpb": 0, "dtcpb": 0,
            "dwin": 0, "tcprtt": 0, "synack": 0, "ackdat": 0, "smean": 248,
            "dmean": 0, "trans_depth": 0, "response_body_len": 0,
            "ct_srv_src": 2, "ct_state_ttl": 2, "ct_dst_ltm": 1,
            "ct_src_dport_ltm": 1, "ct_dst_sport_ltm": 1, "ct_dst_src_ltm": 2,
            "is_ftp_login": 0, "ct_ftp_cmd": 0, "ct_flw_http_mthd": 0,
            "ct_src_ltm": 1, "ct_srv_dst": 2, "is_sm_ips_ports": 0
        }
    ]
    
    payload = {"data": sample_data}
    
    # We mock the entire predict processing to ensure it returns a valid response format
    # without needing real models or data files
    with patch('app.autoencoder') as mock_ae, \
         patch('app.rf_baseline') as mock_rf, \
         patch('app.scaler') as mock_scaler, \
         patch('app.label_encoders') as mock_le, \
         patch('app.es') as mock_es:
        
        # Setup mock behavior
        mock_ae.return_value = torch.zeros((1, 42))
        mock_rf.predict.return_value = np.array([1])
        mock_scaler.transform.return_value = np.zeros((1, 39)) # 42 - 3 categorical
        mock_scaler.feature_names_in_ = np.array(['dummy'] * 39)
        
        response = client.post("/predict", json=payload)
        assert response.status_code == 200
        
        predictions = response.json()
        assert len(predictions) == 1
        assert "is_anomaly" in predictions[0]
