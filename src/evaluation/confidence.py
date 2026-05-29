"""Confidence Module - Estimate uncertainty and reliability of predictions"""

import numpy as np
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
import pandas as pd
from typing import Dict, Tuple


class ConfidenceModule:
    """
    Confidence Module for price predictions.
    Estimates uncertainty and reliability scores for model predictions.
    """

    def __init__(self, model, feature_names):
        self.model = model
        self.feature_names = feature_names
        self.uncertainty_model = None
        self.historical_errors = []
        self.training_data = None

    def set_training_data(self, X_train, y_train):
        """Set training data for distance-based uncertainty estimation"""
        self.training_data = X_train
        self.training_targets = y_train

    def calculate_prediction_interval(self, X, confidence_level=0.95):
        """
        Calculate prediction intervals using jackknife resampling.

        Args:
            X: Input features
            confidence_level: Confidence level (0.95 = 95% interval)

        Returns:
            Dict with lower_bound, upper_bound, and interval_width
        """
        predictions = []

        # Handle single prediction
        if len(X.shape) == 1:
            X = X.reshape(1, -1)

        n_samples = X.shape[0]

        # Generate multiple predictions with bootstrapping
        n_bootstrap = 100
        for _ in range(n_bootstrap):
            # Sample with replacement
            indices = np.random.choice(n_samples, n_samples, replace=True)
            X_sample = X[indices]
            pred = self.model.predict(X_sample)
            predictions.extend(pred)

        predictions = np.array(predictions)

        # Calculate percentiles
        lower = np.percentile(predictions, (1 - confidence_level) / 2 * 100)
        upper = np.percentile(predictions, (1 + confidence_level) / 2 * 100)

        return {
            'lower_bound': float(lower),
            'upper_bound': float(upper),
            'interval_width': float(upper - lower),
            'confidence_level': confidence_level
        }

    def calculate_uncertainty_score(self, X, prediction):
        """
        Calculate uncertainty score based on model agreement and feature stability.

        Returns:
            uncertainty_score: 0-1 (0 = very certain, 1 = very uncertain)
            reliability_score: 0-1 (1 = very reliable)
        """
        # Handle single prediction
        if len(X.shape) == 1:
            X = X.reshape(1, -1)

        # 1. Prediction interval width
        interval = self.calculate_prediction_interval(X, confidence_level=0.9)
        interval_uncertainty = min(1.0, interval['interval_width'] / (prediction * 0.5))

        # 2. Model disagreement (using different model types)
        model_disagreement = self._calculate_model_disagreement(X, prediction)

        # 3. Feature stability (how sensitive is prediction to small changes)
        feature_stability = self._calculate_feature_stability(X)

        # 4. Distance from training data (out-of-distribution detection)
        distance_uncertainty = self._calculate_distance_uncertainty(X)

        # Combine uncertainties with weights
        weights = {
            'interval': 0.35,
            'disagreement': 0.25,
            'stability': 0.25,
            'distance': 0.15
        }

        uncertainty_score = (
                weights['interval'] * interval_uncertainty +
                weights['disagreement'] * model_disagreement +
                weights['stability'] * feature_stability +
                weights['distance'] * distance_uncertainty
        )

        # Clip to valid range
        uncertainty_score = np.clip(uncertainty_score, 0, 1)

        # Reliability score (inverse of uncertainty)
        reliability_score = 1 - uncertainty_score

        # Determine confidence level text
        if reliability_score > 0.7:
            confidence_level = "High"
        elif reliability_score > 0.4:
            confidence_level = "Medium"
        else:
            confidence_level = "Low"

        return {
            'uncertainty_score': float(uncertainty_score),
            'reliability_score': float(reliability_score),
            'confidence_level': confidence_level,
            'components': {
                'prediction_interval_uncertainty': float(interval_uncertainty),
                'model_disagreement': float(model_disagreement),
                'feature_stability': float(feature_stability),
                'distance_uncertainty': float(distance_uncertainty)
            }
        }

    def _calculate_model_disagreement(self, X, prediction):
        """
        Calculate disagreement between different model types.
        """
        try:
            # Simple random forest for comparison
            if self.uncertainty_model is None:
                # Create a simple random forest for uncertainty estimation
                self.uncertainty_model = RandomForestRegressor(
                    n_estimators=30,
                    max_depth=5,
                    random_state=42,
                    n_jobs=-1
                )

                # Train on a subset if training data is available
                if self.training_data is not None and len(self.training_data) > 0:
                    sample_size = min(5000, len(self.training_data))
                    indices = np.random.choice(len(self.training_data), sample_size, replace=False)
                    self.uncertainty_model.fit(
                        self.training_data[indices],
                        self.training_targets[indices]
                    )

            # Get prediction from random forest
            if self.uncertainty_model is not None:
                rf_pred = self.uncertainty_model.predict(X)[0]
                disagreement = abs(prediction - rf_pred) / (prediction + 0.01)
                return min(0.5, disagreement)
            else:
                return 0.1
        except:
            return 0.1

    def _calculate_feature_stability(self, X):
        """
        Calculate how stable the prediction is under small feature perturbations.
        """
        original_pred = self.model.predict(X)[0]
        predictions = []

        # Apply small perturbations to each feature
        for i in range(X.shape[1]):
            X_perturbed = X.copy()
            # Add 5% noise with multiple samples
            for j in range(5):
                noise = np.random.normal(0, 0.05 * np.abs(X[0, i]))
                X_perturbed[0, i] = X[0, i] + noise
                pred = self.model.predict(X_perturbed)[0]
                predictions.append(abs(pred - original_pred))

        # Coefficient of variation of changes
        if predictions:
            mean_change = np.mean(predictions)
            relative_change = mean_change / (original_pred + 0.01)
            return min(0.5, relative_change)

        return 0.1

    def _calculate_distance_uncertainty(self, X):
        """
        Detect if input is far from training distribution.
        """
        if self.training_data is None or len(self.training_data) == 0:
            return 0.1  # Low uncertainty by default

        # Calculate distance to nearest training point
        from sklearn.metrics.pairwise import euclidean_distances
        distances = euclidean_distances(X, self.training_data[:1000])  # Use subset for speed
        min_distance = np.min(distances)

        # Normalize distance to uncertainty (0-1)
        max_expected_distance = 10.0
        uncertainty = min(1.0, min_distance / max_expected_distance)

        return float(uncertainty)

    def get_reliability_factors(self, shap_explanation, prediction):
        """
        Get factors that affect reliability of the explanation.

        Returns:
            Dict with reliability factors and recommendations
        """
        factors = []
        warnings = []

        # Check SHAP value balance
        shap_values = [abs(d['shap_value']) for d in shap_explanation['features'].values()]
        if shap_values:
            max_shap = max(shap_values)
            total_shap = sum(shap_values)
            dominance = max_shap / total_shap if total_shap > 0 else 0

            if dominance > 0.6:
                factors.append({
                    'factor': 'Single feature dominance',
                    'impact': 'reducing',
                    'message': f"Prediction heavily driven by one factor"
                })
                warnings.append("Single factor dominates the prediction - consider market validation")

        # Check prediction vs base value
        base = shap_explanation['base_value']
        if abs(prediction - base) / (base + 0.01) > 1.0:
            factors.append({
                'factor': 'Large deviation from base',
                'impact': 'neutral',
                'message': f"Significant deviation from historical baseline"
            })
            warnings.append("Price deviates significantly from historical patterns")

        # Check for extreme feature values
        extreme_features = []
        for feature, details in shap_explanation['features'].items():
            if abs(details['value']) > 1000:  # Extreme value threshold
                extreme_features.append(feature)

        if extreme_features:
            warnings.append(f"Extreme values detected in: {', '.join(extreme_features)}")

        # Generate recommendations
        recommendations = []
        if warnings:
            recommendations.append("Verify input data accuracy")
            recommendations.append("Cross-reference with market benchmarks")
            recommendations.append("Consider manual review before implementation")
        else:
            recommendations.append("Prediction appears reliable for implementation")
            recommendations.append("Monitor actual sales for validation")

        return {
            'reliability_factors': factors,
            'warnings': warnings,
            'recommendations': recommendations
        }


class UncertaintyEstimator:
    """
    Ensemble-based uncertainty estimation using multiple models.
    """

    def __init__(self, base_model):
        self.base_model = base_model
        self.ensemble_models = []

    def create_ensemble(self, X_train, y_train, n_models=3):
        """
        Create an ensemble of models for uncertainty estimation.
        """
        from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
        from sklearn.linear_model import Ridge

        # Use subset for faster training
        sample_size = min(50000, len(X_train))
        indices = np.random.choice(len(X_train), sample_size, replace=False)
        X_sample = X_train[indices] if hasattr(X_train, '__getitem__') else X_train[:sample_size]
        y_sample = y_train[indices] if hasattr(y_train, '__getitem__') else y_train[:sample_size]

        self.ensemble_models = [
            ('random_forest', RandomForestRegressor(n_estimators=50, max_depth=8, random_state=42, n_jobs=-1)),
            ('gradient_boosting', GradientBoostingRegressor(n_estimators=50, max_depth=5, random_state=42)),
            ('ridge', Ridge(alpha=1.0))
        ]

        # Train all models
        for name, model in self.ensemble_models:
            print(f"  Training {name} for uncertainty estimation...")
            model.fit(X_sample, y_sample)

        return self.ensemble_models

    def predict_with_uncertainty(self, X):
        """
        Get predictions from all ensemble models and calculate uncertainty.
        """
        predictions = []
        model_names = []

        for name, model in self.ensemble_models:
            try:
                pred = model.predict(X)
                predictions.append(pred)
                model_names.append(name)
            except Exception as e:
                print(f"Error with {name}: {e}")
                continue

        if not predictions:
            return {
                'mean_prediction': float(self.base_model.predict(X)[0]),
                'std_prediction': 0.0,
                'uncertainty': 0.0,
                'model_agreement': 1.0
            }

        predictions = np.array(predictions)

        # If predictions are 2D, take first column
        if len(predictions.shape) > 1 and predictions.shape[1] > 1:
            predictions = predictions[:, 0]

        mean_pred = np.mean(predictions)
        std_pred = np.std(predictions)

        return {
            'mean_prediction': float(mean_pred),
            'std_prediction': float(std_pred),
            'all_predictions': predictions.tolist(),
            'model_names': model_names,
            'uncertainty': float(std_pred / mean_pred if mean_pred > 0 else 0),
            'model_agreement': float(1.0 - (std_pred / mean_pred if mean_pred > 0 else 0))
        }