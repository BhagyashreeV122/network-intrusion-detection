import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

def preprocess_data():
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    print("Loading datasets...")
    train_df = pd.read_csv("data/UNSW_NB15_training-set.csv")
    test_df = pd.read_csv("data/UNSW_NB15_testing-set.csv")
    
    # Drop 'id' column
    if 'id' in train_df.columns:
        train_df.drop('id', axis=1, inplace=True)
    if 'id' in test_df.columns:
        test_df.drop('id', axis=1, inplace=True)
        
    # Split train_df into train and val (stratified by attack_cat to ensure all classes represented)
    print("Creating stratified train/val splits...")
    train_df, val_df = train_test_split(train_df, test_size=0.2, stratify=train_df['attack_cat'], random_state=42)
    
    print(f"Train size: {train_df.shape}, Val size: {val_df.shape}, Test size: {test_df.shape}")
    
    # Separate features and targets
    def separate_features_targets(df):
        X = df.drop(['label', 'attack_cat'], axis=1)
        y_binary = df['label']
        y_multi = df['attack_cat']
        return X, y_binary, y_multi
        
    X_train, y_train_bin, y_train_multi = separate_features_targets(train_df)
    X_val, y_val_bin, y_val_multi = separate_features_targets(val_df)
    X_test, y_test_bin, y_test_multi = separate_features_targets(test_df)
    
    # Preprocessing
    categorical_cols = ['proto', 'service', 'state']
    numerical_cols = [col for col in X_train.columns if col not in categorical_cols]
    
    print("Encoding categorical features...")
    # Using LabelEncoder for categoricals
    label_encoders = {}
    for col in categorical_cols:
        X_train[col] = X_train[col].astype(str)
        le = LabelEncoder()
        le.fit(X_train[col])
        
        # Get mode before transforming X_train
        mode_val_str = X_train[col].mode()[0]
        known_classes = set(le.classes_)
        
        # Function to safely transform
        def safe_transform(s):
            s = s.astype(str)
            s = s.apply(lambda x: x if x in known_classes else mode_val_str)
            return le.transform(s)
            
        X_train[col] = le.transform(X_train[col])
        X_val[col] = safe_transform(X_val[col])
        X_test[col] = safe_transform(X_test[col])
        label_encoders[col] = le
        
    print("Scaling numerical features...")
    scaler = StandardScaler()
    X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
    X_val[numerical_cols] = scaler.transform(X_val[numerical_cols])
    X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])
    
    # Save encoders and scalers
    joblib.dump(scaler, "models/scaler.pkl")
    joblib.dump(label_encoders, "models/label_encoders.pkl")
    
    # Encode target 'attack_cat'
    target_le = LabelEncoder()
    y_train_multi_encoded = target_le.fit_transform(y_train_multi)
    y_val_multi_encoded = target_le.transform(y_val_multi)
    # Handle unseen test labels if any (usually 'Normal' is shared, and 9 attack cats)
    y_test_multi_encoded = target_le.transform(y_test_multi)
    joblib.dump(target_le, "models/target_encoder.pkl")
    
    # Save processed data
    print("Saving processed datasets...")
    X_train['label'] = y_train_bin.values
    X_train['attack_cat'] = y_train_multi_encoded
    X_train.to_csv("data/processed/train.csv", index=False)
    del X_train
    
    X_val['label'] = y_val_bin.values
    X_val['attack_cat'] = y_val_multi_encoded
    X_val.to_csv("data/processed/val.csv", index=False)
    del X_val
    
    X_test['label'] = y_test_bin.values
    X_test['attack_cat'] = y_test_multi_encoded
    X_test.to_csv("data/processed/test.csv", index=False)
    del X_test
    print("Preprocessing completed successfully.")

if __name__ == "__main__":
    preprocess_data()
