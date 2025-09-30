# SAM Admissions Predictor – HISP Uganda

This is a CHAP-compatible external model for predicting Severe Acute Malnutrition (SAM) admissions.

## Features
- Predicts SAM admissions using DHIS2 data.
- Uses climate and disease indicators as features.
- Computes risk levels (Low, Medium, High, Very High) for mapping.

## Files
- `model.py`: Python code for training and prediction.
- `metadata.yaml`: CHAP model metadata.
- `requirements.txt`: Python dependencies.
- `models/`: Contains model artifacts:
  - `model.pkl`
  - `scaler.pkl`
  - `features.json`
  - `percentiles.json`

## Usage
### Train
```python
from model import train_model
import pandas as pd

df = pd.read_csv("SAM_training_data.csv")
train_model(df, target="108-NA01b1_2019. No. of new SAM admissions in ITC")
