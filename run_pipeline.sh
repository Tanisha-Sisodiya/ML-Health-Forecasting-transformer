#!/bin/bash

# run_pipeline.sh
# Runs the full End-to-End RUL Prediction Pipeline

# Exit immediately if a command exits with a non-zero status
set -e

echo "=================================================="
echo "   STARTING END-TO-END PIPELINE"
echo "=================================================="

# 1. Feature Engineering
echo ""
echo "[Step 1/4] Running Feature Engineering..."
python feature_engineering.py

# 2. Generate Feature Configuration
echo ""
echo "[Step 2/4] Generating Feature List Configuration..."
python src/generate_features_list.py

# 3. Training
echo ""
echo "[Step 3/4] Starting Model Training..."
# You can adjust epochs and batch_size here
python train.py --epochs 30 --batch_size 64

# 4. Evaluation
echo ""
echo "[Step 4/4] Evaluating Best Model..."
# Note: Ensure the path matches where train.py saves the model
python evaluate.py training_logs/transformer_baseline/best_model.pth

echo ""
echo "=================================================="
echo "   PIPELINE COMPLETED SUCCESSFULLY"
echo "=================================================="
