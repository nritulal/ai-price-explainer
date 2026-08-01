"""Confidence Module - Estimate uncertainty and reliability of predictions"""

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics.pairwise import euclidean_distances


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
        self.training_targets = None

    def set_training_data(self, X_train, y_train):
        """Set training data for distance-based uncertainty estimation"""
        self.training_data = X_train
        self.training_targets = y_train

    def calculate_prediction_interval(self, X, confidence_level=0.95):
        """
        Calculate prediction intervals using bootstrapping with noise.
        This method works with scaled features.
        """
        if len(X.shape) == 1:
            X = X.reshape(1, -1)

        # First, get the base prediction (this is in actual dollar amount)
        base_prediction = float(self.model.predict(X)[0])

        # Generate bootstrap predictions by adding noise to features
        bootstrap_predictions = []
        n_bootstrap = 200

        for _ in range(n_bootstrap):
            X_noisy = X.copy()

            # Add noise to scaled features (0.01-0.03 is appropriate for scaled data)
            noise_scale = np.random.uniform(0.01, 0.03)
            noise = np.random.normal(0, noise_scale, X_noisy.shape)
            X_noisy = X_noisy + noise

            try:
                pred = float(self.model.predict(X_noisy)[0])
                bootstrap_predictions.append(pred)
            except:
                bootstrap_predictions.append(base_prediction * (1 + np.random.normal(0, 0.01)))

        bootstrap_predictions = np.array(bootstrap_predictions)

        # Remove outliers using IQR method
        q1 = np.percentile(bootstrap_predictions, 25)
        q3 = np.percentile(bootstrap_predictions, 75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        filtered_predictions = bootstrap_predictions[
            (bootstrap_predictions >= lower_bound) & (bootstrap_predictions <= upper_bound)
            ]

        if len(filtered_predictions) < 10:
            filtered_predictions = bootstrap_predictions

        # Calculate percentiles based on filtered predictions
        lower = np.percentile(filtered_predictions, (1 - confidence_level) / 2 * 100)
        upper = np.percentile(filtered_predictions, (1 + confidence_level) / 2 * 100)

        # ============================================================
        # FIX: Ensure the prediction interval contains the base prediction
        # and has a meaningful width
        # ============================================================

        # Calculate the spread of predictions
        pred_std = np.std(filtered_predictions)

        # If interval doesn't contain the base prediction, expand it
        if not (lower <= base_prediction <= upper):
            # Use standard deviation to create a reasonable range
            margin = max(pred_std * 1.96, abs(base_prediction) * 0.05)  # 95% CI or 5% margin
            lower = max(0, base_prediction - margin)
            upper = base_prediction + margin

        # Ensure minimum width (at least 5% of base prediction)
        min_width = abs(base_prediction) * 0.05
        if (upper - lower) < min_width:
            center = (lower + upper) / 2
            half_width = min_width / 2
            lower = max(0, center - half_width)
            upper = center + half_width

        return {
            'lower_bound': float(lower),
            'upper_bound': float(upper),
            'interval_width': float(upper - lower),
            'confidence_level': confidence_level,
            'n_predictions': len(filtered_predictions),
            'base_prediction': float(base_prediction)
        }

    def calculate_uncertainty_score(self, X, prediction):
        """Calculate uncertainty score based on model agreement and feature stability."""
        if len(X.shape) == 1:
            X = X.reshape(1, -1)

        # 1. Prediction interval width
        interval = self.calculate_prediction_interval(X, confidence_level=0.9)
        interval_width = interval['interval_width']
        interval_uncertainty = min(1.0, interval_width / (abs(prediction) * 0.5 + 0.01))

        # 2. Model disagreement
        model_disagreement = self._calculate_model_disagreement(X, prediction)

        # 3. Feature stability
        feature_stability = self._calculate_feature_stability(X)

        # 4. Distance from training data
        distance_uncertainty = self._calculate_distance_uncertainty(X)

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

        uncertainty_score = np.clip(uncertainty_score, 0, 1)
        reliability_score = 1 - uncertainty_score

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
        """Calculate disagreement between different model types."""
        try:
            if self.uncertainty_model is None:
                self.uncertainty_model = RandomForestRegressor(
                    n_estimators=30,
                    max_depth=5,
                    random_state=42,
                    n_jobs=-1
                )

                if self.training_data is not None and len(self.training_data) > 0:
                    sample_size = min(5000, len(self.training_data))
                    indices = np.random.choice(len(self.training_data), sample_size, replace=False)
                    self.uncertainty_model.fit(
                        self.training_data[indices],
                        self.training_targets[indices]
                    )

            if self.uncertainty_model is not None:
                rf_pred = float(self.uncertainty_model.predict(X)[0])
                disagreement = abs(prediction - rf_pred) / (abs(prediction) + 0.01)
                return min(0.5, disagreement)
            else:
                return 0.1
        except:
            return 0.1

    def _calculate_feature_stability(self, X):
        """Calculate how stable the prediction is under small feature perturbations."""
        try:
            original_pred = float(self.model.predict(X)[0])
            predictions = []

            for i in range(min(X.shape[1], 10)):
                X_perturbed = X.copy()
                for j in range(5):
                    noise = np.random.normal(0, 0.05 * abs(X[0, i] + 0.01))
                    X_perturbed[0, i] = X[0, i] + noise
                    pred = float(self.model.predict(X_perturbed)[0])
                    predictions.append(abs(pred - original_pred))

            if predictions:
                mean_change = np.mean(predictions)
                relative_change = mean_change / (abs(original_pred) + 0.01)
                return min(0.5, relative_change)
            return 0.1
        except:
            return 0.1

    def _calculate_distance_uncertainty(self, X):
        """Detect if input is far from training distribution."""
        if self.training_data is None or len(self.training_data) == 0:
            return 0.1

        try:
            distances = euclidean_distances(X, self.training_data[:1000])
            min_distance = np.min(distances)
            max_expected_distance = 10.0
            uncertainty = min(1.0, min_distance / max_expected_distance)
            return float(uncertainty)
        except:
            return 0.1

    def get_reliability_factors(self, shap_explanation, prediction):
        """Get factors that affect reliability of the explanation."""
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
                    'message': "Prediction heavily driven by one factor"
                })
                warnings.append("Single factor dominates the prediction - consider market validation")

        # Check prediction vs base value
        base = shap_explanation['base_value']
        if abs(prediction - base) / (abs(base) + 0.01) > 1.0:
            factors.append({
                'factor': 'Large deviation from base',
                'impact': 'neutral',
                'message': "Significant deviation from historical baseline"
            })
            warnings.append("Price deviates significantly from historical patterns")

        # Check for extreme feature values
        extreme_features = []
        for feature, details in shap_explanation['features'].items():
            if abs(details['value']) > 1000:
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
        """Create an ensemble of models for uncertainty estimation."""
        from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
        from sklearn.linear_model import Ridge

        sample_size = min(10000, len(X_train))
        indices = np.random.choice(len(X_train), sample_size, replace=False)
        X_sample = X_train[indices]
        y_sample = y_train[indices]

        self.ensemble_models = [
            ('random_forest', RandomForestRegressor(n_estimators=50, max_depth=8, random_state=42, n_jobs=-1)),
            ('gradient_boosting', GradientBoostingRegressor(n_estimators=50, max_depth=5, random_state=42)),
            ('ridge', Ridge(alpha=1.0))
        ]

        for name, model in self.ensemble_models:
            print(f"  Training {name} for uncertainty estimation...")
            model.fit(X_sample, y_sample)

        return self.ensemble_models

    def predict_with_uncertainty(self, X):
        """Get predictions from all ensemble models and calculate uncertainty."""
        predictions = []

        for name, model in self.ensemble_models:
            try:
                pred = model.predict(X)
                if isinstance(pred, np.ndarray):
                    pred = pred[0] if len(pred) > 0 else 0
                predictions.append(float(pred))
            except Exception as e:
                print(f"Error with {name}: {e}")
                continue

        if not predictions:
            base_pred = float(self.base_model.predict(X)[0])
            return {
                'mean_prediction': float(base_pred),
                'std_prediction': 0.0,
                'uncertainty': 0.0,
                'model_agreement': 1.0
            }

        predictions = np.array(predictions)
        mean_pred = np.mean(predictions)
        std_pred = np.std(predictions)

        if mean_pred != 0:
            cv = std_pred / abs(mean_pred)
            model_agreement = max(0, min(1, 1 - cv))
        else:
            model_agreement = 0.5

        return {
            'mean_prediction': float(mean_pred),
            'std_prediction': float(std_pred),
            'all_predictions': [float(p) for p in predictions],
            'model_names': [name for name, _ in self.ensemble_models],
            'uncertainty': float(std_pred / abs(mean_pred) if mean_pred != 0 else 0),
            'model_agreement': float(model_agreement)
        }