# model.py
#  SAM Admissions Predictor with Risk Levels
# Produces both numeric predictions and risk categories for mapping

import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import os
import json

# Metadata for CHAP registration
MODEL_METADATA = {
    "name": "SAM Admissions Predictor",
    "version": "1.0.0",
    "description": "Predicts Severe Acute Malnutrition (SAM) admissions using climate and disease indicators, and classifies risk levels for mapping.",
    "author": "HISP Uganda",
    "type": "supervised",
    "algorithm": "linear_regression",
    "requirements": {
        "python_version": ">=3.8",
        "min_training_samples": 30
    }
}


class SAMModel:
    def __init__(self, model_dir="models"):
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)

        self.model_path = os.path.join(self.model_dir, "model.pkl")
        self.scaler_path = os.path.join(self.model_dir, "scaler.pkl")
        self.features_path = os.path.join(self.model_dir, "features.json")
        self.percentiles_path = os.path.join(self.model_dir, "percentiles.json")

        self.model = None
        self.scaler = None
        self.features = None
        self.percentiles = None

    def train(self, df: pd.DataFrame, target: str, feature_columns=None):
        """
        Train SAM model using provided DHIS2 dataset
        """
        print("Training SAM predictor...")

        # Default feature columns from your DHIS2 dataset
        if feature_columns is None:
            feature_columns = [
                'CCH - Air temperature (ERA5-Land)',
                'CCH - Precipitation (CHIRPS)',
                'CCH - Relative humidity (ERA5-Land)',
                '108-CD06a. Pneumonia - Cases',
                '108-EP01a1. Malaria Total - Cases',
                '108-NA01b1_2019. No. of new SAM admissions in ITC_lag1',
                '108-NA01b1_2019. No. of new SAM admissions in ITC_lag2',
                '108-NA01b1_2019. No. of new SAM admissions in ITC_lag3',
                '108-NA01b1_2019. No. of new SAM admissions in ITC_lag4',
                '108-NA01b1_2019. No. of new SAM admissions in ITC_lag5',
                '108-NA01b1_2019. No. of new SAM admissions in ITC_lag6'
            ]

        # Filter dataset
        df = df.dropna(subset=[target] + feature_columns)
        X = df[feature_columns]
        y = df[target]

        #  Save feature names
        self.features = feature_columns

        # Scale predictors
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        #  Train linear regression model
        self.model = LinearRegression()
        self.model.fit(X_scaled, y)

        #  Compute percentiles for risk categories
        self.percentiles = {
            "p25": float(np.percentile(y, 25)),
            "p50": float(np.percentile(y, 50)),
            "p75": float(np.percentile(y, 75))
        }

        #  Save artifacts
        joblib.dump(self.model, self.model_path)
        joblib.dump(self.scaler, self.scaler_path)
        with open(self.features_path, "w") as f:
            json.dump(self.features, f)
        with open(self.percentiles_path, "w") as f:
            json.dump(self.percentiles, f)

        #  Evaluate
        y_pred = self.model.predict(X_scaled)
        r2 = r2_score(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        print(f"Training completed. R² = {r2:.2f}, RMSE = {rmse:.2f}")

        return {"r2": r2, "rmse": rmse, "percentiles": self.percentiles}

    def _classify_risk(self, value):
        """
        Assign risk category based on percentiles
        """
        if value <= self.percentiles["p25"]:
            return "Low"
        elif value <= self.percentiles["p50"]:
            return "Medium"
        elif value <= self.percentiles["p75"]:
            return "High"
        else:
            return "Very High"

    def predict(self, df: pd.DataFrame):
        """
        Predict SAM admissions on new DHIS2 data and classify risk levels
        """
        # Load model if not already loaded
        if self.model is None or self.scaler is None:
            self.model = joblib.load(self.model_path)
            self.scaler = joblib.load(self.scaler_path)
            with open(self.features_path, "r") as f:
                self.features = json.load(f)
            with open(self.percentiles_path, "r") as f:
                self.percentiles = json.load(f)

        #  Ensure same features
        df = df.dropna(subset=self.features)
        X = df[self.features]
        X_scaled = self.scaler.transform(X)

        #  Predict
        preds = self.model.predict(X_scaled)

        #  Classify risk levels
        risk_levels = [self._classify_risk(v) for v in preds]

        #  Return both numeric predictions and risk categories
        return {
            "predictions": preds.tolist(),
            "risk_levels": risk_levels
        }


#  Required CHAP hooks
def train_model(df: pd.DataFrame, target: str):
    model = SAMModel()
    metrics = model.train(df, target)
    return metrics


def predict_model(df: pd.DataFrame):
    model = SAMModel()
    result = model.predict(df)
    return result


def get_metadata():
    return MODEL_METADATA
