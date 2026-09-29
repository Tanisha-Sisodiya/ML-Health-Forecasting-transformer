"""
Feature Engineering Script

This script processes the raw NASA C-MAPSS data (train_FD001.txt) to generate 
training artifacts.
Steps:
1. Loads the raw training data.
2. Computes RUL (Remaining Useful Life) for the training set.
3. Drops sensors with low variance (constant sensors).
4. Normalizes the features using Min-Max Scaling.
5. Generates sliding time-window sequences (Seq Length = 30).
6. Splits data into Training and Validation sets.
7. Saves the processed tensors (.npy) and scalers (.pkl) to 'data/artifacts/'.

Usage:
    python feature_engineering.py
"""

import pandas as pd
import numpy as np
import os
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# --- Constants ---
DATA_DIR = 'data'
ARTIFACTS_DIR = os.path.join(DATA_DIR, 'artifacts')
os.makedirs(ARTIFACTS_DIR, exist_ok=True)
SEQ_LEN = 30
VAL_SIZE = 0.2
RANDOM_SEED = 42

def load_data():
    print("Loading raw data...")
    # Define columns
    cols = ['engine_id', 'time_cycles'] + [f'op_setting_{i}' for i in range(1,4)] + [f'sensor_{i}' for i in range(1,22)]
    train_path = os.path.join(DATA_DIR, 'train_FD001.txt')
    
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Raw data not found at {train_path}")

    train_df = pd.read_csv(train_path, sep=r'\s+', header=None, names=cols)
    return train_df, cols

def compute_rul(train_df):
    # Calculate RUL: Max Cycle - Current Cycle
    max_cycles = train_df.groupby('engine_id')['time_cycles'].max()
    train_df = train_df.merge(max_cycles.to_frame(name='max_cycle'), on='engine_id')
    train_df['RUL'] = train_df['max_cycle'] - train_df['time_cycles']
    train_df.drop('max_cycle', axis=1, inplace=True)
    
    # Cap RUL at 125 (common practice for C-MAPSS)
    train_df['RUL'] = train_df['RUL'].clip(upper=125)
    return train_df

def process_features(train_df, cols):
    # Drop known constant sensors
    drop_sensors = ['sensor_1', 'sensor_5', 'sensor_6', 'sensor_10', 'sensor_16', 'sensor_18', 'sensor_19']
    features_to_keep = [c for c in cols if c not in drop_sensors and c not in ['engine_id', 'time_cycles', 'RUL']]
    
    feature_scaler = MinMaxScaler()
    train_df[features_to_keep] = feature_scaler.fit_transform(train_df[features_to_keep])
    
    rul_scaler = MinMaxScaler()
    train_df[['RUL']] = rul_scaler.fit_transform(train_df[['RUL']])
    
    # Save scalers
    joblib.dump(feature_scaler, os.path.join(ARTIFACTS_DIR, 'feature_scaler.pkl'))
    joblib.dump(rul_scaler, os.path.join(ARTIFACTS_DIR, 'rul_scaler.pkl'))
    print("Scalers saved.")
    
    return train_df, features_to_keep

def create_sequences(df, features, seq_len):
    X = []
    y = []
    
    for engine_id in df['engine_id'].unique():
        engine_data = df[df['engine_id'] == engine_id]
        
        data_array = engine_data[features].values
        rul_array = engine_data['RUL'].values
        
        for i in range(len(data_array) - seq_len + 1):
            X.append(data_array[i:i+seq_len])
            y.append(rul_array[i+seq_len-1])
            
    return np.array(X), np.array(y)

def main():
    try:
        # Load
        train_df, cols = load_data()
        
        # RUL
        train_df = compute_rul(train_df)
        
        # Features
        train_df, features = process_features(train_df, cols)
        
        # Sequences
        print("Generating sequences...")
        X, y = create_sequences(train_df, features, SEQ_LEN)
        print(f"Total Sequences: {X.shape}")
        
        # Split
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=VAL_SIZE, random_state=RANDOM_SEED)
        
        # Save
        np.save(os.path.join(ARTIFACTS_DIR, 'X_train.npy'), X_train)
        np.save(os.path.join(ARTIFACTS_DIR, 'y_train.npy'), y_train)
        np.save(os.path.join(ARTIFACTS_DIR, 'X_val.npy'), X_val)
        np.save(os.path.join(ARTIFACTS_DIR, 'y_val.npy'), y_val)
        
        print("Feature Engineering Complete. Artifacts saved.")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
