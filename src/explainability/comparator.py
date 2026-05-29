"""Compare SHAP and LIME explanations"""

import numpy as np
from typing import Dict, List


class ExplanationComparator:
    def __init__(self, shap_explainer, lime_explainer):
        self.shap_explainer = shap_explainer
        self.lime_explainer = lime_explainer

    def compare_explanations(self, instance, feature_names):
        """Compare SHAP and LIME for the same instance"""

        # Get both explanations
        shap_exp = self.shap_explainer.explain_prediction(instance, feature_names)
        lime_exp = self.lime_explainer.explain_prediction(instance)

        # Get top features
        shap_top3 = set(list(shap_exp['features'].keys())[:3])
        lime_top3 = set(list(lime_exp['features'].keys())[:3])

        agreement = len(shap_top3.intersection(lime_top3)) / 3

        # Calculate correlation of SHAP and LIME values for common features
        common_features = shap_top3.intersection(lime_top3)
        if common_features:
            shap_vals = [shap_exp['features'][f]['shap_value'] for f in common_features]
            lime_vals = [lime_exp['features'][f]['value'] for f in common_features]

            # Normalize
            shap_vals = np.array(shap_vals) / (np.abs(shap_vals).sum() + 1e-8)
            lime_vals = np.array(lime_vals) / (np.abs(lime_vals).sum() + 1e-8)

            correlation = np.corrcoef(shap_vals, lime_vals)[0, 1]
        else:
            correlation = 0

        return {
            'shap_explanation': shap_exp,
            'lime_explanation': lime_exp,
            'agreement_score': agreement,
            'correlation': float(correlation),
            'consistent': agreement >= 0.66
        }

    def get_consensus_explanation(self, instance, feature_names, top_k=3):
        """Get consensus explanation from both methods"""
        comparison = self.compare_explanations(instance, feature_names)

        # Combine feature importance
        combined_scores = {}

        for feature in feature_names:
            shap_score = comparison['shap_explanation']['features'].get(feature, {}).get('abs_impact', 0)
            lime_score = comparison['lime_explanation']['features'].get(feature, {}).get('abs_impact', 0)

            # Normalize and combine
            combined = (shap_score + lime_score) / 2
            combined_scores[feature] = combined

        # Get top features
        top_features = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        return {
            'top_features': [f[0] for f in top_features],
            'scores': dict(top_features),
            'agreement': comparison['agreement_score'],
            'correlation': comparison['correlation']
        }

    def batch_compare(self, instances, feature_names):
        """Compare explanations for multiple instances"""
        results = []
        for instance in instances:
            results.append(self.compare_explanations(instance, feature_names))

        avg_agreement = np.mean([r['agreement_score'] for r in results])
        avg_correlation = np.mean([r['correlation'] for r in results])

        return {
            'average_agreement': avg_agreement,
            'average_correlation': avg_correlation,
            'individual_results': results
        }