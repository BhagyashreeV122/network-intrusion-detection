import torch
import torch.nn as nn
import joblib
import os
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# Create models directory
os.makedirs("models", exist_ok=True)

# 1. Mock Threshold
with open("models/ae_threshold.txt", "w") as f:
    f.write("0.1")

# 2. Mock Scaler
scaler = StandardScaler()
dummy_data = np.random.rand(10, 42)
scaler.fit(dummy_data)
# Add the required attribute
scaler.feature_names_in_ = np.array(['dur', 'spkts', 'dpkts', 'sbytes', 'dbytes', 'rate', 'sttl', 'dttl', 'sload', 'dload', 'sloss', 'dloss', 'sinpkt', 'dinpkt', 'sjit', 'djit', 'swin', 'stcpb', 'dtcpb', 'dwin', 'tcprtt', 'synack', 'ackdat', 'smean', 'dmean', 'trans_depth', 'response_body_len', 'ct_srv_src', 'ct_state_ttl', 'ct_dst_ltm', 'ct_src_dport_ltm', 'ct_dst_sport_ltm', 'ct_dst_src_ltm', 'is_ftp_login', 'ct_ftp_cmd', 'ct_flw_http_mthd', 'ct_src_ltm', 'ct_srv_dst', 'is_sm_ips_ports'])
joblib.dump(scaler, "models/scaler.pkl")

# 3. Mock Label Encoders
les = {
    'proto': LabelEncoder().fit(['tcp', 'udp', 'icmp', 'arp', 'ospf']),
    'service': LabelEncoder().fit(['-', 'http', 'dns', 'smtp', 'ftp']),
    'state': LabelEncoder().fit(['FIN', 'INT', 'CON', 'REQ', 'RST'])
}
joblib.dump(les, "models/label_encoders.pkl")

# 4. Mock Autoencoder (MATCH REAL STRUCTURE)
class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8)
        )
        self.decoder = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, input_dim)
        )
    def forward(self, x):
        return self.decoder(self.encoder(x))

model = Autoencoder(42)
torch.save(model.state_dict(), "models/autoencoder.pth")

# 5. Mock Random Forest
rf = RandomForestClassifier(n_estimators=1)
rf.fit(dummy_data, np.random.randint(0, 2, 10))
joblib.dump(rf, "models/rf_baseline.pkl")

print("Successfully generated mock models for CI environment.")
