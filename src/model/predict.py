"""Prediction module for trained model"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from src.utils.constants import EXPERIMENTS_PATH


class Predictor:
    def __init__(self, model_path: Path = None):
        """Initialize predictor with trained model"""
        if model_path is None:
            model_path = EXPERIMENTS_PATH / 'model_xgboost.pkl'

        if model_path.exists():
            self.model = joblib.load(model_path)
            print(f"✓ Model loaded from {model_path}")
        else:
            print(f"⚠️ Model not found at {model_path}")
            self.model = None

    def predict_single(self, features: dict) -> float:
        """Predict price for a single item"""
        if self.model is None:
            raise ValueError("Model not loaded")

        # Convert dict to array in correct order
        feature_order = ['sales', 'cost', 'inventory', 'has_promo',
                         'seasonality_index', 'elasticity', 'competitor_price',
                         'item_encoded', 'store_encoded']

        feature_array = np.array([[features[f] for f in feature_order]])

        prediction = self.model.predict(feature_array)[0]
        return float(prediction)

    def predict_batch(self, features_df: pd.DataFrame) -> np.ndarray:
        """Predict prices for multiple items"""
        if self.model is None:
            raise ValueError("Model not loaded")

        return self.model.predict(features_df)

    def predict_with_uncertainty(self, features: dict, n_iterations: int = 10) -> dict:
        """Predict with uncertainty estimation using dropout approximation"""
        if self.model is None:
            raise ValueError("Model not loaded")

        # Simple uncertainty estimation using prediction variance
        predictions = []
        for _ in range(n_iterations):
            # Add small noise to features
            noisy_features = {}
            for k, v in features.items():
                noise = np.random.normal(0, v * 0.01)
                noisy_features[k] = v + noise

            pred = self.predict_single(noisy_features)
            predictions.append(pred)

        return {
            'mean_price': float(np.mean(predictions)),
            'std_price': float(np.std(predictions)),
            'lower_bound': float(np.percentile(predictions, 5)),
            'upper_bound': float(np.percentile(predictions, 95)),
            'uncertainty_pct': float(np.std(predictions) / np.mean(predictions) * 100)
        }