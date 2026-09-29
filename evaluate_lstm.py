"""
Evaluation Script for LSTM RUL Prediction

This script evaluates a trained LSTMBaseline model on the test dataset.

Usage:
    python evaluate_lstm.py <checkpoint_path>
    
Example:
    python evaluate_lstm.py training_logs/lstm_baseline/best_model.pth
"""

import argparse
import torch
import numpy as np
import os
import joblib
import json
import pandas as pd
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import torch.nn as nn
import pytorch_lightning as pl

# LSTM Model Definition (extracted from baseline_model.ipynb)
class LSTMBaseline(pl.LightningModule):
    def __init__(self, input_dim, hidden_dim=50, num_layers=2, output_dim=1, learning_rate=0.001):
        super().__init__()
        self.save_hyperparameters()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.criterion = nn.MSELoss()
        
    def forward(self, x):
        out, _ = self.lstm(x)
        last_out = out[:, -1, :]  # Get the last time step output
        prediction = self.fc(last_out)
        return prediction
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.criterion(y_pred, y.view(-1, 1))
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.criterion(y_pred, y.view(-1, 1))
        self.log('val_loss', loss)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

# --- Constants ---
DATA_DIR = 'data'
ARTIFACTS_DIR = os.path.join(DATA_DIR, 'artifacts')
SEQ_LEN = 30

def load_resources():
    print("Loading resources...")
    # Load feature list
    feature_path = os.path.join(ARTIFACTS_DIR, 'input_features.json')
    if not os.path.exists(feature_path):
        raise FileNotFoundError(f"Feature list not found at {feature_path}. Run src/generate_features_list.py first.")
    
    with open(feature_path, 'r') as f:
        training_features = json.load(f)
        
    # Load Scalers
    if not os.path.exists(os.path.join(ARTIFACTS_DIR, 'feature_scaler.pkl')):
        raise FileNotFoundError("Scalers not found. Run feature engineering first.")

    feature_scaler = joblib.load(os.path.join(ARTIFACTS_DIR, 'feature_scaler.pkl'))
    rul_scaler = joblib.load(os.path.join(ARTIFACTS_DIR, 'rul_scaler.pkl'))
    
    return training_features, feature_scaler, rul_scaler

def load_data(training_features):
    # Load Raw Test Data
    cols = ['engine_id', 'time_cycles'] + [f'op_setting_{i}' for i in range(1,4)] + [f'sensor_{i}' for i in range(1,22)]
    test_path = os.path.join(DATA_DIR, 'test_FD001.txt')
    test_df = pd.read_csv(test_path, sep=r'\s+', header=None, names=cols)
    
    # Load Ground Truth
    truth_path = os.path.join(DATA_DIR, 'RUL_FD001.txt')
    truth_df = pd.read_csv(truth_path, sep=r'\s+', header=None, names=['RUL'])

    # Check for missing columns
    missing_cols = set(training_features) - set(test_df.columns)
    if missing_cols:
        raise ValueError(f"Test data is missing columns required by model: {missing_cols}")

    return test_df, truth_df

def preprocess_data(test_df, feature_scaler, training_features):
    test_df_norm = test_df.copy()
    test_df_norm[training_features] = feature_scaler.transform(test_df[training_features])
    return test_df_norm

def create_test_sequences(df, seq_len, features):
    X_test = []
    kept_engine_ids = [] # Track IDs to align Ground Truth
    
    engine_ids = df['engine_id'].unique()
    
    for engine in engine_ids:
        engine_data = df[df['engine_id'] == engine].sort_values('time_cycles')
        
        if len(engine_data) >= seq_len:
            # Take the LAST 30 cycles
            seq = engine_data[features].values[-seq_len:]
            X_test.append(seq)
            kept_engine_ids.append(engine)
        else:
            print(f"Skipping Engine {engine}: Sequence length {len(engine_data)} < {seq_len}")
            
    return np.array(X_test), kept_engine_ids

def evaluate(model, X_test, rul_scaler, truth_df, kept_ids):
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    
    print("Running inference...")
    with torch.no_grad():
        predictions_normalized = model(X_test_tensor).numpy()

    # Invert Scaling
    predictions_cycles = rul_scaler.inverse_transform(predictions_normalized)
    
    # Align Ground Truth
    # truth_df is index 0..99 which corresponds to Engine 1..100
    # kept_ids are the Engine IDs (1-based) that we actually predicted for
    
    # Create an explicit engine_id column in truth_df to match against
    truth_df['engine_id'] = np.arange(1, len(truth_df) + 1)
    
    # Filter truth_df to only include the engines we kept
    aligned_truth = truth_df[truth_df['engine_id'].isin(kept_ids)]
    y_true = aligned_truth['RUL'].values.reshape(-1, 1)
    
    # Validation check
    if len(predictions_cycles) != len(y_true):
        raise ValueError(f"Critical Error: Predictions ({len(predictions_cycles)}) and Truth ({len(y_true)}) size mismatch!")

    rmse = np.sqrt(mean_squared_error(y_true, predictions_cycles))
    print(f"\n=== FINAL TEST RMSE: {rmse:.4f} ===")
    
    return y_true, predictions_cycles, rmse

def plot_results(y_true, y_pred, rmse):
    plt.figure(figsize=(12, 6))
    plt.plot(y_true, label='Actual RUL', color='black')
    plt.plot(y_pred, label='Predicted RUL', color='red')
    plt.title(f'LSTM RUL Prediction (RMSE: {rmse:.2f})')
    plt.xlabel('Engine Index (Aligned)')
    plt.ylabel('Remaining Useful Life')
    plt.legend()
    plt.savefig('evaluation_results_lstm.png')
    print("Plot saved to evaluation_results_lstm.png")

def main():
    parser = argparse.ArgumentParser(description='Evaluate LSTM Model')
    parser.add_argument('checkpoint_path', type=str, help='Path to the model checkpoint')
    args = parser.parse_args()

    try:
        # 1. Resources
        training_features, feature_scaler, rul_scaler = load_resources()
        
        # 2. Data
        test_df, truth_df = load_data(training_features)
        
        # 3. Preprocess
        test_df_norm = preprocess_data(test_df, feature_scaler, training_features)
        
        # 4. Sequences
        X_test, kept_ids = create_test_sequences(test_df_norm, SEQ_LEN, training_features)
        print(f"Test Sequences Shape: {X_test.shape} (Kept {len(kept_ids)} engines)")
        
        # 5. Model
        print(f"Loading model from {args.checkpoint_path}...")
        model = LSTMBaseline(input_dim=len(training_features))
        
        checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
        
        # Handle PyTorch Lightning checkpoint format
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            # Remove 'model.' prefix if present (for Lightning checkpoints)
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('model.'):
                    new_state_dict[k[6:]] = v
                elif k.startswith('lstm.') or k.startswith('fc.'):
                    new_state_dict[k] = v
                else:
                    new_state_dict[k] = v
            model.load_state_dict(new_state_dict, strict=False)
        else:
            model.load_state_dict(checkpoint)
        
        model.eval()
        print("Model loaded.")
        
        # 6. Evaluate
        y_true, y_pred, rmse = evaluate(model, X_test, rul_scaler, truth_df, kept_ids)
        
        # 7. Plot
        plot_results(y_true, y_pred, rmse)

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

