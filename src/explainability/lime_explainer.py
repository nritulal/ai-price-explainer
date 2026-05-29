"""LIME explainer for local interpretations"""

from lime.lime_tabular import LimeTabularExplainer
import numpy as np
import pandas as pd


class LIMExplainer:
    def __init__(self, model, train_data, feature_names, mode='regression'):
        """Initialize LIME explainer"""
        self.model = model
        self.train_data = train_data
        self.feature_names = feature_names
        self.mode = mode

        # Ensure train_data is numpy array
        if isinstance(train_data, pd.DataFrame):
            train_data = train_data.values

        self.explainer = LimeTabularExplainer(
            train_data,
            feature_names=feature_names,
            mode=mode,
            discretize_continuous=True,
            random_state=42
        )

    def explain_prediction(self, instance, num_features=5):
        """Explain a single prediction using LIME"""
        # LIME expects 2D array
        if isinstance(instance, pd.DataFrame):
            instance = instance.values

        if len(instance.shape) == 1:
            instance = instance.reshape(1, -1)

        # Get prediction function
        def predict_fn(x):
            return self.model.predict(x)

        explanation = self.explainer.explain_instance(
            instance[0],
            predict_fn,
            num_features=num_features
        )

        # Parse LIME explanation
        features = {}
        for feature, weight in explanation.as_list():
            features[feature] = {
                'value': weight,
                'impact': 'positive' if weight > 0 else 'negative',
                'abs_impact': abs(weight)
            }

        # Sort by absolute impact
        features = dict(sorted(features.items(), key=lambda x: x[1]['abs_impact'], reverse=True))

        return {
            'prediction': float(self.model.predict(instance)[0]),
            'features': features,
            'lime_explanation': explanation,
            'intercept': explanation.intercept[1] if hasattr(explanation, 'intercept') else 0
        }

    def get_feature_contributions(self, instance):
        """Get numerical feature contributions"""
        explanation = self.explain_prediction(instance)
        return explanation['features']

    def explain_batch(self, instances, num_features=5):
        """Explain multiple instances"""
        explanations = []
        for i, instance in enumerate(instances):
            explanations.append(self.explain_prediction(instance, num_features))
        return explanations