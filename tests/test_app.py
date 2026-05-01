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
    # Use hardcoded sample data instead of reading from a file
    # This ensures tests pass even without the large CSV data
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
    
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    
    predictions = response.json()
    assert len(predictions) == 1
    for p in predictions:
        assert "is_anomaly" in p
        assert "reconstruction_error" in p
        assert "rf_prediction" in p
        assert p["features_processed"] == 42
