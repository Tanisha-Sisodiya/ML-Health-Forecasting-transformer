
"""
Training Script for Transformer RUL Prediction

This script handles the training of the TransformerBaseline model using the NASA C-MAPSS dataset.
It performs the following:
1. Loads preprocessed training and validation data (artifacts).
2. Creates PyTorch DataLoaders.
3. Initializes the Transformer model (defined in src/model.py).
4. Trains the model using MSE Loss and Adam optimizer.
5. Saves the best model checkpoint to 'training_logs/transformer_baseline/best_model.pth'.

Usage:
    python train.py --epochs 30 --batch_size 64
"""

import pandas as pd
import numpy as np
import torch
import os
import joblib
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import math
import argparse
import time

# Import shared model definition
from src.model import TransformerBaseline 

# --- Constants ---
DATA_DIR = 'data'
ARTIFACTS_DIR = os.path.join(DATA_DIR, 'artifacts')
LOG_DIR = os.path.join('training_logs', 'transformer_baseline')
os.makedirs(LOG_DIR, exist_ok=True)

# --- Helper Functions ---

def load_data():
    print("Loading training data artifacts...")
    if not os.path.exists(ARTIFACTS_DIR):
        raise FileNotFoundError(f"Artifacts directory not found at {ARTIFACTS_DIR}. Run feature engineering first.")
        
    try:
        X_train = np.load(os.path.join(ARTIFACTS_DIR, 'X_train.npy'))
        y_train = np.load(os.path.join(ARTIFACTS_DIR, 'y_train.npy'))
        X_val = np.load(os.path.join(ARTIFACTS_DIR, 'X_val.npy'))
        y_val = np.load(os.path.join(ARTIFACTS_DIR, 'y_val.npy'))
        print(f"Data Loaded: Train {X_train.shape}, Val {X_val.shape}")
        return X_train, y_train, X_val, y_val
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Missing .npy files in artifacts: {e}")

def create_dataloaders(X_train, y_train, X_val, y_val, batch_size=64):
    train_data = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    val_data = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32))
    
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size)
    return train_loader, val_loader

def train_model(model, train_loader, val_loader, epochs=30, lr=0.0005):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Training on {device}...")

    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch.view(-1, 1))
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * X_batch.size(0)
        
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                y_pred = model(X_batch)
                loss = criterion(y_pred, y_batch.view(-1, 1))
                val_loss += loss.item() * X_batch.size(0)
        
        val_loss /= len(val_loader.dataset)
        
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")

        # Checkout saving logic
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = os.path.join(LOG_DIR, 'best_model.pth')
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  Saved best model to {checkpoint_path}")

    return model

def main():
    parser = argparse.ArgumentParser(description='Train Transformer Model')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    args = parser.parse_args()

    # 1. Load Data
    X_train, y_train, X_val, y_val = load_data()
    
    # 2. Dataloaders
    train_loader, val_loader = create_dataloaders(X_train, y_train, X_val, y_val, args.batch_size)
    
    # 3. Model
    input_dim = X_train.shape[2]
    # Ideally checking input_features.json length here to match would be good but redundant if data matches
    model = TransformerBaseline(input_dim=input_dim)
    
    # 4. Train
    train_model(model, train_loader, val_loader, epochs=args.epochs)
    
    print("\nTraining Complete.")
    print(f"Run evaluation with: python evaluate.py training_logs/transformer_baseline/best_model.pth")

if __name__ == "__main__":
    main()
