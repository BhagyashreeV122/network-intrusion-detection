import pandas as pd
import numpy as np
import joblib

def check_data_drift():
    print("Loading scaler to get reference means and variances...")
    scaler = joblib.load("models/scaler.pkl")
    ref_means = scaler.mean_
    ref_vars = scaler.var_
    numerical_cols = scaler.feature_names_in_
    
    print("Simulating incoming inference data (loading from test set)...")
    # In a real scenario, this would be data collected from the API over the last X hours
    recent_data = pd.read_csv("data/processed/test.csv", nrows=5000)
    
    print("Checking for drift in numerical features...")
    drift_detected = False
    
    for i, col in enumerate(numerical_cols):
        col_mean = recent_data[col].mean()
        # Simple heuristic: if mean shifts by more than 1 standard deviation of the training data
        ref_std = np.sqrt(ref_vars[i])
        
        if abs(col_mean - ref_means[i]) > ref_std:
            print(f"WARNING: Drift detected in feature '{col}'")
            print(f"  Reference Mean: {ref_means[i]:.4f}, Recent Mean: {col_mean:.4f}")
            drift_detected = True
            
    if not drift_detected:
        print("No significant data drift detected.")

if __name__ == "__main__":
    check_data_drift()
