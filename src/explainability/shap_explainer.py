"""SHAP explainer for model predictions"""

import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.utils.constants import PROCESSED_DATA_PATH
import json
import re


class SHAPExplainer:
    def __init__(self, model, background_data):
        """Initialize SHAP explainer"""
        self.model = model
        self.background_data = background_data

        # Convert to numpy if needed
        if isinstance(background_data, pd.DataFrame):
            background_data = background_data.values

        # Create explainer
        self.explainer = shap.TreeExplainer(model, background_data)

    def explain_prediction(self, instance, feature_names):
        """Explain a single prediction"""
        # Convert to numpy if needed
        if isinstance(instance, pd.DataFrame):
            instance = instance.values

        if len(instance.shape) == 1:
            instance = instance.reshape(1, -1)

        # Get SHAP values
        shap_values = self.explainer.shap_values(instance)

        # Handle case where shap_values is list (multi-output)
        if isinstance(shap_values, list):
            shap_values = shap_values[0]

        # Create explanation dict
        explanation = {
            'base_value': float(self.explainer.expected_value),
            'prediction': float(self.model.predict(instance)[0]),
            'features': {}
        }

        for i, feature in enumerate(feature_names):
            explanation['features'][feature] = {
                'value': float(instance[0, i]),
                'shap_value': float(shap_values[0, i]),
                'impact': 'positive' if shap_values[0, i] > 0 else 'negative',
                'abs_impact': abs(float(shap_values[0, i]))
            }

        # Sort by absolute SHAP value
        explanation['features'] = dict(
            sorted(explanation['features'].items(),
                   key=lambda x: x[1]['abs_impact'],
                   reverse=True)
        )

        return explanation

    def get_top_features(self, instance, feature_names, top_k=5):
        """Get top k features impacting the prediction"""
        shap_values = self.explainer.shap_values(instance)

        if isinstance(shap_values, list):
            shap_values = shap_values[0]

        feature_impacts = []
        for i, (feature, sv) in enumerate(zip(feature_names, shap_values[0])):
            feature_impacts.append({
                'feature': feature,
                'shap_value': float(sv),
                'abs_impact': abs(float(sv))
            })

        feature_impacts.sort(key=lambda x: x['abs_impact'], reverse=True)

        return feature_impacts[:top_k]

    def plot_summary(self, X_sample, feature_names):
        """Create SHAP summary plot"""
        if isinstance(X_sample, pd.DataFrame):
            X_sample = X_sample.values

        shap_values = self.explainer.shap_values(X_sample)

        if isinstance(shap_values, list):
            shap_values = shap_values[0]

        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False)
        plt.tight_layout()
        plt.savefig(PROCESSED_DATA_PATH / 'shap_summary.png', dpi=150, bbox_inches='tight')
        plt.close()

        return str(PROCESSED_DATA_PATH / 'shap_summary.png')

    def get_feature_importance_global(self, X_sample, feature_names):
        """Get global feature importance"""
        if isinstance(X_sample, pd.DataFrame):
            X_sample = X_sample.values

        shap_values = self.explainer.shap_values(X_sample)

        if isinstance(shap_values, list):
            shap_values = shap_values[0]

        mean_abs_shap = np.mean(np.abs(shap_values), axis=0)

        importance = dict(zip(feature_names, mean_abs_shap))
        importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

        return importance


# For backward compatibility
class ShapExplainer(SHAPExplainer):
    pass