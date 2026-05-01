import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import os

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
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

def train_ae():
    print("Loading data for Autoencoder...")
    train_df = pd.read_csv("data/processed/train.csv")
    val_df = pd.read_csv("data/processed/val.csv")
    
    # Train only on normal traffic for reconstruction
    normal_train = train_df[train_df['label'] == 0].drop(['label', 'attack_cat'], axis=1)
    
    # For validation, we can use the whole val set to see reconstruction error differences
    X_val = val_df.drop(['label', 'attack_cat'], axis=1)
    y_val = val_df['label']
    
    X_train_tensor = torch.tensor(normal_train.values, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val.values, dtype=torch.float32)
    
    train_dataset = TensorDataset(X_train_tensor, X_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    
    input_dim = X_train_tensor.shape[1]
    model = Autoencoder(input_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    epochs = 20
    print(f"Training Autoencoder for {epochs} epochs...")
    model.train()
    
    for epoch in range(epochs):
        epoch_loss = 0
        for data, _ in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, data)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(train_loader):.4f}")
            
    print("Evaluating reconstruction errors on Validation Set...")
    model.eval()
    with torch.no_grad():
        reconstructed = model(X_val_tensor)
        # Calculate MSE per sample
        mse = torch.mean((X_val_tensor - reconstructed)**2, dim=1).numpy()
    
    # Simple check on normal vs attack reconstruction error
    normal_mse = mse[y_val.values == 0].mean()
    attack_mse = mse[y_val.values == 1].mean()
    print(f"Mean MSE for Normal traffic: {normal_mse:.4f}")
    print(f"Mean MSE for Attack traffic: {attack_mse:.4f}")
    
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/autoencoder.pth")
    print("Saved Autoencoder model to models/autoencoder.pth")

if __name__ == "__main__":
    train_ae()
