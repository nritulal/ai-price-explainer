"""LLM-based explanation summarization - Supports Local LLM and Mock"""

import json
from typing import Dict, Optional
import numpy as np

# Try to import languagemodels
try:
    import languagemodels as lm

    LM_AVAILABLE = True
except ImportError:
    LM_AVAILABLE = False
    print("[WARNING] languagemodels not installed. Run: pip install languagemodels")


class LLMSummarizer:
    def __init__(self, provider='mock', max_ram="4gb"):
        """
        Initialize LLM summarizer.

        Args:
            provider: 'local' or 'mock'
            max_ram: Max RAM for local model (e.g., "4gb", "8gb")
        """
        self.provider = provider

        if provider == 'local' and LM_AVAILABLE:
            try:
                lm.config["max_ram"] = max_ram
                print(f"[OK] Local LLM initialized with {max_ram} RAM")
                # Test the model
                test_response = lm.do("Hello")
                print(f"[OK] Local LLM test successful")
                self.client = lm
            except Exception as e:
                print(f"[WARNING] Local LLM init failed: {e}")
                self.provider = 'mock'
                self.client = None
        else:
            self.provider = 'mock'
            self.client = None
            print("[INFO] Using mock LLM mode (no API calls)")

    def generate_explanation(
            self,
            shap_explanation: Dict,
            lime_explanation: Dict = None,
            business_context: Dict = None
    ) -> str:
        """Generate human-readable explanation."""

        if self.provider == 'mock':
            return self._mock_explanation(shap_explanation, business_context)
        elif self.provider == 'local' and self.client is not None:
            return self._local_explanation(shap_explanation, business_context)
        else:
            return self._mock_explanation(shap_explanation, business_context)

    def _local_explanation(self, shap_explanation: Dict, business_context: Dict = None) -> str:
        """Generate explanation using local LLM."""
        try:
            # Get prediction and features
            prediction = shap_explanation.get('prediction', 0)
            base_value = shap_explanation.get('base_value', 0)
            features = shap_explanation.get('features', {})
            top_features = list(features.items())[:5]

            # Build prompt
            feature_lines = []
            for feature, details in top_features:
                shap_val = details.get('shap_value', 0)
                impact = "increases" if shap_val > 0 else "decreases"
                val = details.get('value', 'N/A')
                if isinstance(val, float):
                    val = f"{val:.2f}"
                feature_lines.append(f"  - {feature}: {val} ({impact} price by ${abs(shap_val):.2f})")

            features_text = "\n".join(feature_lines) if feature_lines else "No key factors identified."

            prompt = f"""You are a pricing expert. Explain why an AI recommended a price.

RECOMMENDED PRICE: ${prediction:.2f}
BASE PRICE: ${base_value:.2f}

KEY FACTORS:
{features_text}

Provide a short, business-friendly explanation (max 100 words). Use simple language. No jargon.

EXPLANATION:"""

            response = self.client.do(prompt)
            return response.strip()

        except Exception as e:
            print(f"[WARNING] Local LLM failed: {e}")
            return self._mock_explanation(shap_explanation, business_context)

    def _mock_explanation(self, shap_explanation: Dict, business_context: Dict = None) -> str:
        """Mock LLM explanation for testing."""
        prediction = shap_explanation.get('prediction', 0)
        base_value = shap_explanation.get('base_value', 0)

        features = shap_explanation.get('features', {})
        top_features = list(features.items())[:4]

        parts = []
        for feature, details in top_features:
            impact_symbol = "+" if details.get('impact') == 'positive' else "-"
            shap_val = abs(details.get('shap_value', 0))
            val = details.get('value', 'N/A')
            if isinstance(val, float):
                val = f"{val:.2f}"
            parts.append(f"  * {impact_symbol} {feature}: {val} ({impact_symbol} price by ${shap_val:.2f})")

        explanation = f"""
The recommended price is based on the following key factors:

{chr(10).join(parts)}

Business Insight:
The base price would have been ${base_value:.2f}. The adjustments above reflect current market conditions, demand patterns, and competitive landscape.

Recommendation:
This price aligns with market dynamics and maximizes expected revenue.

Actionable Next Steps:
- Monitor competitor prices for elastic items
- Consider promotional bundling if inventory is high
- Review pricing strategy for items with high seasonality
        """
        return explanation.strip()