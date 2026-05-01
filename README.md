# Cloud VM Network Intrusion Detection System (IDS)

This project implements a Network Anomaly Detection System using the UNSW-NB15 dataset. It features both a Random Forest classifier and a PyTorch-based Autoencoder for detecting malicious traffic.

## Project Structure
- `app.py`: FastAPI application for real-time predictions.
- `preprocess.py`: Data cleaning, feature engineering, and scaling.
- `train_ae.py`: PyTorch Autoencoder training script.
- `train_rf.py`: Random Forest baseline training script.
- `evaluate.py`: Model evaluation and threshold optimization.
- `drift_monitor.py`: Monitors for data distribution changes.

---

## 1. Local Setup (Direct Python)

### Prerequisites
- Python 3.10+
- Virtual environment (recommended)

### Installation
```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running the API
```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```
The API will be available at `http://localhost:8000`. You can access the Swagger UI at `http://localhost:8000/docs`.

---

## 2. Docker Deployment (Recommended)

This setup includes the FastAPI app, Elasticsearch (for logging), and Kibana (for visualization).

```bash
# Build and start all services
docker-compose up --build
```

- **API**: `http://localhost:8000`
- **Kibana**: `http://localhost:5601`

---

## 3. Training the Models

If you need to retrain the models from scratch:

1. **Preprocess Data**:
   ```bash
   python preprocess.py
   ```
2. **Train Random Forest**:
   ```bash
   python train_rf.py
   ```
3. **Train Autoencoder**:
   ```bash
   python train_ae.py
   ```
4. **Evaluate & Update Threshold**:
   ```bash
   python evaluate.py
   ```

---

## 4. Testing

Run the test suite to ensure everything is working correctly:
```bash
pytest tests/
```

---

## API Usage Example

Send a POST request to `/predict` with traffic features:

```json
{
  "data": [
    {
      "dur": 0.000011,
      "proto": "udp",
      "service": "-",
      "state": "INT",
      "spkts": 2,
      "dpkts": 0,
      "sbytes": 496,
      "dbytes": 0,
      "rate": 90909.0902,
      "sttl": 254,
      "dttl": 0,
      "sload": 180363632.0,
      "dload": 0.0,
      "sloss": 0,
      "dloss": 0,
      "sinpkt": 0.011,
      "dinpkt": 0.0,
      "sjit": 0.0,
      "djit": 0.0,
      "swin": 0,
      "stcpb": 0,
      "dtcpb": 0,
      "dwin": 0,
      "tcprtt": 0.0,
      "synack": 0.0,
      "ackdat": 0.0,
      "smean": 248,
      "dmean": 0,
      "trans_depth": 0,
      "response_body_len": 0,
      "ct_srv_src": 2,
      "ct_state_ttl": 2,
      "ct_dst_ltm": 1,
      "ct_src_dport_ltm": 1,
      "ct_dst_sport_ltm": 1,
      "ct_dst_src_ltm": 2,
      "is_ftp_login": 0,
      "ct_ftp_cmd": 0,
      "ct_flw_http_mthd": 0,
      "ct_src_ltm": 1,
      "ct_srv_dst": 2,
      "is_sm_ips_ports": 0
    }
  ]
}
```
