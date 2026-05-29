"""Unit tests for explainability modules"""

import unittest
import numpy as np
from src.explainability.shap_explainer import SHAPExplainer
from src.explainability.lime_explainer import LIMExplainer


class TestExplainability(unittest.TestCase):

    def setUp(self):
        """Setup test data"""

        # Create a simple mock model
        class MockModel:
            def predict(self, X):
                return np.ones(len(X)) * 10

        self.model = MockModel()
        self.train_data = np.random.randn(100, 5)
        self.feature_names = ['f1', 'f2', 'f3', 'f4', 'f5']

    def test_shap_explainer_initialization(self):
        """Test SHAP explainer initialization"""
        explainer = SHAPExplainer(self.model, self.train_data)
        self.assertIsNotNone(explainer)
        self.assertIsNotNone(explainer.explainer)

    def test_lime_explainer_initialization(self):
        """Test LIME explainer initialization"""
        explainer = LIMExplainer(self.model, self.train_data, self.feature_names)
        self.assertIsNotNone(explainer)
        self.assertIsNotNone(explainer.explainer)

    def test_shap_prediction_explanation(self):
        """Test SHAP prediction explanation"""
        explainer = SHAPExplainer(self.model, self.train_data)
        instance = np.random.randn(1, 5)
        explanation = explainer.explain_prediction(instance, self.feature_names)

        self.assertIn('prediction', explanation)
        self.assertIn('features', explanation)
        self.assertEqual(len(explanation['features']), 5)


if __name__ == '__main__':
    unittest.main()