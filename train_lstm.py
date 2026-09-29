
"""
Training Script for LSTM Baseline RUL Prediction

This script trains the LSTMBaseline model on the NASA C-MAPSS dataset so the
baseline is reproducible from the CLI (previously it only existed in
baseline_model.ipynb). It performs the following:
1. Loads preprocessed training and validation data (artifacts).
2. Creates PyTorch DataLoaders.
3. Initializes the LSTM baseline (architecture matches evaluate_lstm.py).
4. Trains the model using MSE Loss and Adam optimizer.
5. Saves the best model checkpoint to 'training_logs/lstm_baseline/best_model.pth'.

Usage:
    python train_lstm.py --epochs 30 --batch_size 64
"""

import numpy as np
import torch
import os
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import argparse

# --- Constants ---
DATA_DIR = 'data'
ARTIFACTS_DIR = os.path.join(DATA_DIR, 'artifacts')
LOG_DIR = os.path.join('training_logs', 'lstm_baseline')
os.makedirs(LOG_DIR, exist_ok=True)


class LSTMBaseline(nn.Module):
    """Plain nn.Module twin of the LightningModule in evaluate_lstm.py.

    Attribute names (lstm, fc) match exactly so the saved state_dict loads
    into either class.
    """
    def __init__(self, input_dim, hidden_dim=50, num_layers=2, output_dim=1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        last_out = out[:, -1, :]  # Get the last time step output
        prediction = self.fc(last_out)
        return prediction


# --- Helper Functions ---

def load_data():
    print("Loading training data artifacts...")
    if not os.path.exists(ARTIFACTS_DIR):
        raise FileNotFoundError(f"Artifacts directory not found at {ARTIFACTS_DIR}. Run feature engineering first.")

    X_train = np.load(os.path.join(ARTIFACTS_DIR, 'X_train.npy'))
    y_train = np.load(os.path.join(ARTIFACTS_DIR, 'y_train.npy'))
    X_val = np.load(os.path.join(ARTIFACTS_DIR, 'X_val.npy'))
    y_val = np.load(os.path.join(ARTIFACTS_DIR, 'y_val.npy'))
    print(f"Data Loaded: Train {X_train.shape}, Val {X_val.shape}")
    return X_train, y_train, X_val, y_val

def create_dataloaders(X_train, y_train, X_val, y_val, batch_size=64):
    train_data = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    val_data = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32))

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size)
    return train_loader, val_loader

def train_model(model, train_loader, val_loader, epochs=30, lr=0.001):
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

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = os.path.join(LOG_DIR, 'best_model.pth')
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  Saved best model to {checkpoint_path}")

    return model

def main():
    parser = argparse.ArgumentParser(description='Train LSTM Baseline Model')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    args = parser.parse_args()

    # 1. Load Data
    X_train, y_train, X_val, y_val = load_data()

    # 2. Dataloaders
    train_loader, val_loader = create_dataloaders(X_train, y_train, X_val, y_val, args.batch_size)

    # 3. Model
    input_dim = X_train.shape[2]
    model = LSTMBaseline(input_dim=input_dim)

    # 4. Train
    train_model(model, train_loader, val_loader, epochs=args.epochs)

    print("\nTraining Complete.")
    print(f"Run evaluation with: python evaluate_lstm.py training_logs/lstm_baseline/best_model.pth")

if __name__ == "__main__":
    main()
