"""LIME explainer for local interpretations - Improved Version"""

from lime.lime_tabular import LimeTabularExplainer
import numpy as np
import pandas as pd
import re


class LIMExplainer:
    def __init__(self, model, train_data, feature_names, mode='regression'):
        """Initialize LIME explainer"""
        self.model = model
        self.feature_names = feature_names
        self.mode = mode

        # Ensure train_data is numpy array
        if isinstance(train_data, pd.DataFrame):
            train_data = train_data.values
        elif isinstance(train_data, list):
            train_data = np.array(train_data)

        if len(train_data.shape) == 1:
            train_data = train_data.reshape(1, -1)

        # Ensure feature_names is a list of strings
        if isinstance(feature_names, np.ndarray):
            feature_names = feature_names.tolist()
        if not isinstance(feature_names, list):
            feature_names = list(feature_names)

        # Create LIME explainer
        self.explainer = LimeTabularExplainer(
            train_data,
            feature_names=feature_names,
            mode=mode,
            discretize_continuous=True,
            random_state=42
        )

    def _normalize_name(self, name):
        """Normalize feature name for matching"""
        if not name:
            return ""
        name = name.lower()
        name = re.sub(r'[^a-z0-9]', '_', name)
        name = re.sub(r'_+', '_', name)
        name = name.strip('_')
        return name

    def _match_feature_name(self, lime_name):
        """Match LIME feature name to original feature name"""
        # Remove common prefixes
        clean_name = lime_name
        for prefix in ["> ", "< ", "= ", "~ ", "¬ "]:
            if clean_name.startswith(prefix):
                clean_name = clean_name[len(prefix):]

        # Try exact match
        for orig_name in self.feature_names:
            if orig_name == clean_name:
                return orig_name

        # Try normalized match
        clean_norm = self._normalize_name(clean_name)
        for orig_name in self.feature_names:
            if self._normalize_name(orig_name) == clean_norm:
                return orig_name

        # Try partial match
        for orig_name in self.feature_names:
            if clean_norm in self._normalize_name(orig_name) or self._normalize_name(orig_name) in clean_norm:
                return orig_name

        # Return cleaned name as fallback
        return clean_name

    def explain_prediction(self, instance, num_features=8):
        """Explain a single prediction using LIME"""
        # Ensure instance is 2D numpy array
        if isinstance(instance, pd.DataFrame):
            instance = instance.values
        elif isinstance(instance, list):
            instance = np.array(instance)

        if len(instance.shape) == 1:
            instance = instance.reshape(1, -1)

        # Define prediction function
        def predict_fn(x):
            return self.model.predict(x)

        # Get explanation
        explanation = self.explainer.explain_instance(
            instance[0],
            predict_fn,
            num_features=num_features
        )

        # Parse LIME explanation
        features = {}
        for feature_name, weight in explanation.as_list():
            # Match to original feature name
            matched_name = self._match_feature_name(feature_name)

            features[matched_name] = {
                'value': float(weight),
                'impact': 'positive' if weight > 0 else 'negative',
                'abs_impact': abs(float(weight))
            }

        # Sort by absolute impact
        features = dict(sorted(features.items(), key=lambda x: x[1]['abs_impact'], reverse=True))

        return {
            'prediction': float(self.model.predict(instance)[0]),
            'features': features,
            'lime_explanation': explanation
        }

    def get_feature_contributions(self, instance):
        """Get numerical feature contributions"""
        explanation = self.explain_prediction(instance)
        return explanation['features']

    def explain_batch(self, instances, num_features=5):
        """Explain multiple instances"""
        explanations = []
        for instance in instances:
            explanations.append(self.explain_prediction(instance, num_features))
        return explanations


class LimeExplainer(LIMExplainer):
    pass