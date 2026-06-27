"""Compare SHAP and LIME explanations - Improved Version"""

import numpy as np
import re


class ExplanationComparator:
    def __init__(self, shap_explainer, lime_explainer):
        self.shap_explainer = shap_explainer
        self.lime_explainer = lime_explainer

    def _normalize_name(self, name: str) -> str:
        """Normalize feature names for comparison"""
        if not name:
            return ""
        name = name.lower()
        name = re.sub(r'[^a-z0-9]', '_', name)
        name = re.sub(r'_+', '_', name)
        name = name.strip('_')
        return name

    def compare_explanations(self, instance, feature_names):
        """Compare SHAP and LIME for the same instance"""

        # Get both explanations
        shap_exp = self.shap_explainer.explain_prediction(instance, feature_names)
        lime_exp = self.lime_explainer.explain_prediction(instance)

        # Get feature keys
        shap_keys = list(shap_exp['features'].keys())
        lime_keys = list(lime_exp['features'].keys())

        # Build normalized mapping for LIME
        lime_norm_map = {}
        for key in lime_keys:
            norm = self._normalize_name(key)
            lime_norm_map[norm] = key

        # Find matches
        matches = []
        for shap_key in shap_keys:
            shap_norm = self._normalize_name(shap_key)

            # Try exact match
            if shap_norm in lime_norm_map:
                matches.append((shap_key, lime_norm_map[shap_norm]))
                continue

            # Try partial match
            for lime_norm, lime_key in lime_norm_map.items():
                if shap_norm in lime_norm or lime_norm in shap_norm:
                    matches.append((shap_key, lime_key))
                    break

        # Calculate agreement
        if matches:
            # Get top 3 SHAP features
            shap_top3 = shap_keys[:3] if len(shap_keys) >= 3 else shap_keys
            # Get top 3 LIME features
            lime_top3 = lime_keys[:3] if len(lime_keys) >= 3 else lime_keys

            # Normalize for comparison
            shap_top3_norm = {self._normalize_name(k) for k in shap_top3}
            lime_top3_norm = {self._normalize_name(k) for k in lime_top3}

            # Jaccard similarity on top 3
            intersection = len(shap_top3_norm & lime_top3_norm)
            union = len(shap_top3_norm | lime_top3_norm)
            agreement = intersection / union if union > 0 else 0

            # Correlation on common features
            common_shap = []
            common_lime = []
            for shap_key, lime_key in matches:
                if shap_key in shap_exp['features'] and lime_key in lime_exp['features']:
                    common_shap.append(shap_key)
                    common_lime.append(lime_key)

            if len(common_shap) > 1:
                shap_vals = [shap_exp['features'][k]['shap_value'] for k in common_shap]
                lime_vals = [lime_exp['features'][k]['value'] for k in common_lime]

                # Normalize
                shap_vals = np.array(shap_vals) / (np.abs(shap_vals).sum() + 1e-8)
                lime_vals = np.array(lime_vals) / (np.abs(lime_vals).sum() + 1e-8)

                if len(shap_vals) > 1 and np.std(shap_vals) > 1e-8 and np.std(lime_vals) > 1e-8:
                    correlation = np.corrcoef(shap_vals, lime_vals)[0, 1]
                    correlation = float(correlation) if not np.isnan(correlation) else 0.0
                else:
                    correlation = 0.0
            else:
                correlation = 0.0
        else:
            agreement = 0.0
            correlation = 0.0
            matches = []

        return {
            'shap_explanation': shap_exp,
            'lime_explanation': lime_exp,
            'agreement_score': float(agreement),
            'correlation': float(correlation) if not np.isnan(correlation) else 0.0,
            'consistent': agreement >= 0.66,
            'matched_features': matches
        }

    def get_consensus_explanation(self, instance, feature_names, top_k=3):
        """Get consensus explanation from both methods"""
        comparison = self.compare_explanations(instance, feature_names)

        # Combine feature importance
        combined_scores = {}

        for feature in feature_names:
            shap_score = comparison['shap_explanation']['features'].get(feature, {}).get('abs_impact', 0)
            lime_score = 0
            for shap_key, lime_key in comparison.get('matched_features', []):
                if shap_key == feature:
                    lime_score = comparison['lime_explanation']['features'].get(lime_key, {}).get('abs_impact', 0)
                    break

            combined = (shap_score + lime_score) / 2 if (shap_score + lime_score) > 0 else 0
            combined_scores[feature] = combined

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
            'average_agreement': float(avg_agreement),
            'average_correlation': float(avg_correlation),
            'individual_results': results
        }