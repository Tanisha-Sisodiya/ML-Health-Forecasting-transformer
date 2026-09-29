
"""
Feature Configuration Generator

This script defines the feature selection logic (sensors to drop/keep) and saves 
the configuration to 'data/artifacts/input_features.json'.
This ensures that both training and evaluation scripts use the exact same set of input features.
"""

import os
import json

# --- Constants ---
DATA_DIR = 'data'
ARTIFACTS_DIR = os.path.join(DATA_DIR, 'artifacts')

# Known sensor configuration
cols = ['engine_id', 'time_cycles'] + [f'op_setting_{i}' for i in range(1,4)] + [f'sensor_{i}' for i in range(1,22)]
drop_sensors = ['sensor_1', 'sensor_5', 'sensor_6', 'sensor_10', 'sensor_16', 'sensor_18', 'sensor_19']
features_to_keep = [c for c in cols if c not in drop_sensors and c not in ['engine_id', 'time_cycles', 'RUL']]

def save_features_list():
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    out_path = os.path.join(ARTIFACTS_DIR, 'input_features.json')
    with open(out_path, 'w') as f:
        json.dump(features_to_keep, f)
    print(f"Features list saved to {out_path}")
    print(f"Features: {features_to_keep}")

if __name__ == "__main__":
    save_features_list()
