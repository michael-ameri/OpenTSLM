# Fraud Detection with OpenTSLM

This directory contains demo scripts to train and test OpenTSLM on mobile carrier fraud detection data.

## 1. Generate Dummy Data
First, generate the dummy dataset based on your requirements (multi-variate time series, 7 days, 15-min intervals).

```bash
python demo/fraud_detection/generate_data.py
```
This creates `fraud_data.csv` containing 100 episodes.

## 2. Train the Model
Train the OpenTSLM model (Flamingo architecture) on the generated data. This script fine-tunes the model to reason about fraud based on the time series and TPA trigger info.

```bash
python demo/fraud_detection/train.py --epochs 5 --batch_size 4
```

**Note:** You must have access to `meta-llama/Llama-3.2-1B` on HuggingFace and be logged in via `hf auth login`. You can specify a different model with `--model_id`.

## 3. Run Inference
Test the trained model on the test split.

```bash
python demo/fraud_detection/inference.py --checkpoint demo/fraud_detection/checkpoints/best_model.pt
```

## Structure
- `generate_data.py`: Creates dummy CSV data.
- `FraudDetectionDataset.py`: Custom Dataset class inheriting from `QADataset`. Handles data loading and prompt formatting.
- `train.py`: Training script with a custom training loop.
- `inference.py`: Inference script to generate predictions.
