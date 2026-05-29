"""Model training with XGBoost (Alternative version)"""

import xgboost as xgb
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
from pathlib import Path
import json
import time
from src.utils.constants import MODEL_CONFIG, PROCESSED_DATA_PATH, EXPERIMENTS_PATH


class PricePredictor:
    def __init__(self, config: dict = None):
        self.config = config or MODEL_CONFIG
        self.model = None
        self.training_history = {}

    def train(self, X_train, y_train, X_val, y_val):
        """Train XGBoost model"""
        print("\n" + "=" * 50)
        print("Training XGBoost Model")
        print("=" * 50)

        params = self.config['params'].copy()
        params['objective'] = 'reg:squarederror'
        params['eval_metric'] = 'rmse'

        # Create model
        self.model = xgb.XGBRegressor(**params)

        # Train without early stopping (simpler approach)
        start_time = time.time()

        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=True
        )

        training_time = time.time() - start_time

        # Evaluate
        train_pred = self.model.predict(X_train)
        val_pred = self.model.predict(X_val)

        metrics = {
            'train_rmse': float(np.sqrt(mean_squared_error(y_train, train_pred))),
            'train_mae': float(mean_absolute_error(y_train, train_pred)),
            'train_r2': float(r2_score(y_train, train_pred)),
            'val_rmse': float(np.sqrt(mean_squared_error(y_val, val_pred))),
            'val_mae': float(mean_absolute_error(y_val, val_pred)),
            'val_r2': float(r2_score(y_val, val_pred)),
            'training_time_seconds': training_time
        }

        print("\nTraining Metrics:")
        print(f"  Train RMSE: {metrics['train_rmse']:.4f}")
        print(f"  Train MAE:  {metrics['train_mae']:.4f}")
        print(f"  Train R2:   {metrics['train_r2']:.4f}")
        print(f"  Val RMSE:   {metrics['val_rmse']:.4f}")
        print(f"  Val MAE:    {metrics['val_mae']:.4f}")
        print(f"  Val R2:     {metrics['val_r2']:.4f}")
        print(f"  Time:       {training_time:.2f} seconds")

        self.training_history = metrics

        return metrics

    def predict(self, X):
        """Make predictions"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        return self.model.predict(X)

    def save_model(self, path: Path = None):
        """Save model and feature info"""
        if path is None:
            path = EXPERIMENTS_PATH / 'model_xgboost.pkl'

        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path)
        print(f"\nModel saved to {path}")

        # Save training history
        history_path = EXPERIMENTS_PATH / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)

    def load_model(self, path: Path):
        """Load trained model"""
        self.model = joblib.load(path)
        print(f"Model loaded from {path}")

    def get_feature_importance(self, feature_names):
        """Get feature importance from trained model"""
        if self.model is None:
            raise ValueError("Model not trained")

        importance = self.model.feature_importances_
        feature_importance = dict(zip(feature_names, importance))
        feature_importance = dict(sorted(feature_importance.items(),
                                         key=lambda x: x[1], reverse=True))

        return feature_importance