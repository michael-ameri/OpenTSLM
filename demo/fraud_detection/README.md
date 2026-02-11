# Fraud Detection Demo

This directory contains scripts to train an OpenTSLM model for fraud detection using multivariate time series data from mobile carriers.

## Files

- **`generate_data.py`**: Generates synthetic training and testing data in JSONL format.
- **`FraudDataset.py`**: A custom PyTorch Dataset class that loads the JSONL data and formats it for OpenTSLM.
- **`train.py`**: A training script to fine-tune `OpenTSLM-Flamingo` on the generated data.

## Usage

1. **Generate Data**
   Run the data generator to create `train.jsonl` and `test.jsonl` in the `data/` subdirectory.
   ```bash
   python demo/fraud_detection/generate_data.py
   ```

2. **Train the Model**
   Run the training script. You can specify the model ID, batch size, and number of epochs.
   ```bash
   python demo/fraud_detection/train.py --llm_id google/gemma-3-270m --epochs 3
   ```

   **Note:** You must have access to the specified Hugging Face model. If using a gated model (like Llama or Gemma), ensure you are logged in with `huggingface-cli login`.

## Data Format

The data generator creates samples with 3 time-series channels:
1. **Call Duration** (Minutes)
2. **Startcall Count** (Number of calls)
3. **Revenue** (Log-transformed currency)

Each sample represents a 7-day period aggregated into 1-hour bins (168 time steps).
The text prompt includes statistical summaries (mean, std) for each channel to help the model reason about the data scale.
