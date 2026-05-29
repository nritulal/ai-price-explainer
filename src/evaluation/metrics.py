"""Evaluation metrics for model and explanations"""

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy import stats


class ModelMetrics:
    @staticmethod
    def calculate_all(y_true, y_pred):
        """Calculate comprehensive metrics"""

        # Basic regression metrics
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)

        # Percentage errors
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

        # Accuracy within thresholds
        within_5pct = np.mean(np.abs((y_true - y_pred) / y_true) <= 0.05) * 100
        within_10pct = np.mean(np.abs((y_true - y_pred) / y_true) <= 0.10) * 100

        # Correlation
        correlation = np.corrcoef(y_true, y_pred)[0, 1]

        return {
            'MAE': mae,
            'RMSE': rmse,
            'R2': r2,
            'MAPE': mape,
            'Accuracy_Within_5%': within_5pct,
            'Accuracy_Within_10%': within_10pct,
            'Correlation': correlation
        }

    @staticmethod
    def evaluate_explanations(shap_values, lime_values, feature_names):
        """Evaluate explanation quality"""

        # Agreement between SHAP and LIME
        agreement_scores = []

        for i in range(len(shap_values)):
            shap_top = np.argsort(np.abs(shap_values[i]))[-3:]
            lime_top = np.argsort(np.abs(lime_values[i]))[-3:]

            agreement = len(set(shap_top) & set(lime_top)) / 3
            agreement_scores.append(agreement)

        return {
            'mean_explanation_agreement': np.mean(agreement_scores),
            'std_agreement': np.std(agreement_scores),
            'consistency_score': np.mean(agreement_scores) * 100
        }