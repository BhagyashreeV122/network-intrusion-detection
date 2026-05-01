import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure the data directory exists
os.makedirs("data", exist_ok=True)

url_train = "https://raw.githubusercontent.com/Nir-J/ML-Projects/master/UNSW-Network_Packet_Classification/UNSW_NB15_training-set.csv"
url_test = "https://raw.githubusercontent.com/Nir-J/ML-Projects/master/UNSW-Network_Packet_Classification/UNSW_NB15_testing-set.csv"

train_file = "data/UNSW_NB15_training-set.csv"
test_file = "data/UNSW_NB15_testing-set.csv"

def download_data():
    if not os.path.exists(train_file):
        print(f"Downloading {train_file}...")
        df_train = pd.read_csv(url_train)
        df_train.to_csv(train_file, index=False)
    if not os.path.exists(test_file):
        print(f"Downloading {test_file}...")
        df_test = pd.read_csv(url_test)
        df_test.to_csv(test_file, index=False)

def perform_eda():
    print("\n--- Starting EDA ---")
    df = pd.read_csv(train_file)
    print(f"Dataset Shape: {df.shape}")
    
    # Missing Values
    missing_vals = df.isnull().sum()
    missing_vals = missing_vals[missing_vals > 0]
    if len(missing_vals) > 0:
        print("\nMissing Values:")
        print(missing_vals)
    else:
        print("\nNo Missing Values Found.")

    # Data Types
    print("\nData Types:")
    print(df.dtypes.value_counts())
    
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    print(f"\nCategorical Columns: {categorical_cols}")

    # Target Class Distribution
    # Target columns usually are 'label' (0 or 1) and 'attack_cat' (category)
    if 'label' in df.columns:
        print("\nBinary Label Distribution:")
        print(df['label'].value_counts(normalize=True) * 100)
        
    if 'attack_cat' in df.columns:
        print("\nAttack Category Distribution:")
        print(df['attack_cat'].value_counts())
        
        # Plotting class distribution
        plt.figure(figsize=(10, 6))
        sns.countplot(y='attack_cat', data=df, order=df['attack_cat'].value_counts().index)
        plt.title('Attack Category Distribution (Class Imbalance)')
        plt.xlabel('Count')
        plt.ylabel('Attack Category')
        plt.tight_layout()
        plt.savefig('attack_cat_distribution.png')
        print("Saved class distribution plot to 'attack_cat_distribution.png'.")
        
    # Check for constant columns
    constant_cols = [col for col in df.columns if df[col].nunique() <= 1]
    print(f"\nConstant Columns: {constant_cols}")
    
if __name__ == "__main__":
    download_data()
    perform_eda()
