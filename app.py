import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import pandas as pd
import joblib
import os
from datetime import datetime
from elasticsearch import Elasticsearch

from train_ae import Autoencoder

app = FastAPI(title="Network IDS API", description="Anomaly Detection for Cloud VM Network Traffic")

# Initialize Elasticsearch client
es_url = os.getenv("ELASTICSEARCH_URL", "http://localhost:9200")
# Using basic auth if needed, but here it's disabled in docker-compose
es = Elasticsearch(es_url)

# Load models and preprocessing objects at startup
models_dir = "models"
MODELS_LOADED = False
threshold = 0.1 # Default value

try:
    if os.path.exists(os.path.join(models_dir, "ae_threshold.txt")):
        scaler = joblib.load(os.path.join(models_dir, "scaler.pkl"))
        label_encoders = joblib.load(os.path.join(models_dir, "label_encoders.pkl"))
        
        with open(os.path.join(models_dir, "ae_threshold.txt"), "r") as f:
            threshold = float(f.read().strip())
            
        input_dim = 42
        autoencoder = Autoencoder(input_dim)
        autoencoder.load_state_dict(torch.load(os.path.join(models_dir, "autoencoder.pth"), map_location=torch.device('cpu')))
        autoencoder.eval()
        
        rf_baseline = joblib.load(os.path.join(models_dir, "rf_baseline.pkl"))
        MODELS_LOADED = True
    else:
        print("WARNING: Models not found. Using placeholders.")
except Exception as e:
    print(f"Error loading models: {e}")

class NetworkTrafficRequest(BaseModel):
    data: List[Dict[str, Any]]

class PredictionResponse(BaseModel):
    is_anomaly: bool
    reconstruction_error: float
    rf_prediction: int
    features_processed: int

@app.post("/predict", response_model=List[PredictionResponse])
def predict(request: NetworkTrafficRequest):
    if not MODELS_LOADED:
        return [PredictionResponse(is_anomaly=False, reconstruction_error=0.0, rf_prediction=0, features_processed=len(request.data[0]) if request.data else 0) for _ in range(len(request.data))]

    try:
        df = pd.DataFrame(request.data)
        categorical_cols = ['proto', 'service', 'state']
        numerical_cols = scaler.feature_names_in_.tolist()
        
        original_order = ['dur', 'proto', 'service', 'state', 'spkts', 'dpkts', 'sbytes', 'dbytes', 'rate', 'sttl', 'dttl', 'sload', 'dload', 'sloss', 'dloss', 'sinpkt', 'dinpkt', 'sjit', 'djit', 'swin', 'stcpb', 'dtcpb', 'dwin', 'tcprtt', 'synack', 'ackdat', 'smean', 'dmean', 'trans_depth', 'response_body_len', 'ct_srv_src', 'ct_state_ttl', 'ct_dst_ltm', 'ct_src_dport_ltm', 'ct_dst_sport_ltm', 'ct_dst_src_ltm', 'is_ftp_login', 'ct_ftp_cmd', 'ct_flw_http_mthd', 'ct_src_ltm', 'ct_srv_dst', 'is_sm_ips_ports']
        
        for col in original_order:
            if col not in df.columns:
                df[col] = 0
        
        df_copy = df[original_order].copy()
        
        for col in categorical_cols:
            le = label_encoders[col]
            known_classes = set(le.classes_)
            mode_val = le.classes_[0]
            df_copy[col] = df_copy[col].astype(str).apply(lambda x: x if x in known_classes else mode_val)
            df_copy[col] = le.transform(df_copy[col])
            
        df_copy[numerical_cols] = scaler.transform(df_copy[numerical_cols])
        X_tensor = torch.tensor(df_copy.values, dtype=torch.float32)
        
        with torch.no_grad():
            reconstructed = autoencoder(X_tensor)
            mse = torch.mean((X_tensor - reconstructed)**2, dim=1).numpy()
            
        is_anomaly = mse > threshold
        rf_preds = rf_baseline.predict(df_copy)
        
        responses = []
        for i in range(len(df)):
            res = PredictionResponse(
                is_anomaly=bool(is_anomaly[i]),
                reconstruction_error=float(mse[i]),
                rf_prediction=int(rf_preds[i]),
                features_processed=df.shape[1]
            )
            responses.append(res)
            
            try:
                log_doc = {
                    "@timestamp": datetime.utcnow().isoformat(),
                    "is_anomaly": res.is_anomaly,
                    "reconstruction_error": res.reconstruction_error,
                    "rf_prediction": res.rf_prediction,
                    "features": request.data[i]
                }
                es.index(index="ids-logs", document=log_doc)
            except Exception as es_err:
                print(f"Failed to log to ES: {es_err}")
                
        return responses
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/health")
def health_check():
    connected = False
    try:
        connected = es.ping()
    except:
        connected = False
    return {"status": "healthy", "threshold": threshold, "es_connected": connected}
