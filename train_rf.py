import pandas as pd
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score, roc_auc_score

def train_rf():
    print("Loading preprocessed data for RF...")
    train_df = pd.read_csv("data/processed/train.csv")
    val_df = pd.read_csv("data/processed/val.csv")
    
    # Separate features and labels
    X_train = train_df.drop(['label', 'attack_cat'], axis=1)
    y_train = train_df['label'] # We will train RF to detect anomalies (binary)
    
    X_val = val_df.drop(['label', 'attack_cat'], axis=1)
    y_val = val_df['label']
    
    print("Training Random Forest Baseline...")
    rf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1, max_depth=15)
    rf.fit(X_train, y_train)
    
    print("Evaluating on Validation Set...")
    preds = rf.predict(X_val)
    probs = rf.predict_proba(X_val)[:, 1]
    
    print("\nClassification Report:")
    print(classification_report(y_val, preds))
    
    f1 = f1_score(y_val, preds)
    roc_auc = roc_auc_score(y_val, probs)
    print(f"Validation F1 Score: {f1:.4f}")
    print(f"Validation ROC AUC: {roc_auc:.4f}")
    
    os.makedirs("models", exist_ok=True)
    joblib.dump(rf, "models/rf_baseline.pkl")
    print("Saved RF model to models/rf_baseline.pkl")

if __name__ == "__main__":
    train_rf()
