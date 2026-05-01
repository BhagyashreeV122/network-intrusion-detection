import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.metrics import classification_report, precision_recall_curve, f1_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# Import the Autoencoder class from train_ae
from train_ae import Autoencoder

def evaluate_models():
    print("Loading test data...")
    test_df = pd.read_csv("data/processed/test.csv")
    
    X_test = test_df.drop(['label', 'attack_cat'], axis=1)
    y_test = test_df['label']
    y_test_multi = test_df['attack_cat']
    
    # 1. Evaluate Random Forest Baseline
    print("\n--- Evaluating Random Forest Baseline ---")
    rf = joblib.load("models/rf_baseline.pkl")
    rf_preds = rf.predict(X_test)
    rf_probs = rf.predict_proba(X_test)[:, 1]
    
    print("Random Forest Classification Report:")
    print(classification_report(y_test, rf_preds))
    print(f"Random Forest ROC AUC: {roc_auc_score(y_test, rf_probs):.4f}")
    
    # 2. Evaluate Autoencoder
    print("\n--- Evaluating Autoencoder ---")
    input_dim = X_test.shape[1]
    ae = Autoencoder(input_dim)
    ae.load_state_dict(torch.load("models/autoencoder.pth"))
    ae.eval()
    
    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
    
    with torch.no_grad():
        reconstructed = ae(X_test_tensor)
        # Calculate MSE per sample
        mse = torch.mean((X_test_tensor - reconstructed)**2, dim=1).numpy()
    
    # Calculate Precision-Recall curve to find optimal threshold
    precision, recall, thresholds = precision_recall_curve(y_test, mse)
    
    # Calculate F1 scores for each threshold
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    
    # Find the threshold that maximizes F1 score
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]
    best_f1 = f1_scores[best_idx]
    
    print(f"Optimal Autoencoder Threshold (max F1): {best_threshold:.6f}")
    print(f"Autoencoder Best F1 Score: {best_f1:.4f}")
    
    # Predict with optimal threshold
    ae_preds = (mse > best_threshold).astype(int)
    print("\nAutoencoder Classification Report:")
    print(classification_report(y_test, ae_preds))
    print(f"Autoencoder ROC AUC: {roc_auc_score(y_test, mse):.4f}")
    
    # Save the optimal threshold for the inference service
    with open("models/ae_threshold.txt", "w") as f:
        f.write(str(best_threshold))
        
    # Plot ROC curves
    plt.figure(figsize=(10, 8))
    
    rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_probs)
    plt.plot(rf_fpr, rf_tpr, label=f'Random Forest (AUC = {roc_auc_score(y_test, rf_probs):.4f})')
    
    ae_fpr, ae_tpr, _ = roc_curve(y_test, mse)
    plt.plot(ae_fpr, ae_tpr, label=f'Autoencoder (AUC = {roc_auc_score(y_test, mse):.4f})')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve: Anomaly Detection')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig('roc_curve_comparison.png')
    print("Saved ROC curve comparison to roc_curve_comparison.png")

if __name__ == "__main__":
    evaluate_models()
