"""Local LLM client using languagemodels - No API key required"""

import os
import json
from typing import Dict, Optional

# Try to import languagemodels
try:
    import languagemodels as lm

    LM_AVAILABLE = True
except ImportError:
    LM_AVAILABLE = False
    print("[WARNING] languagemodels not installed. Run: pip install languagemodels")


class LocalLLMClient:
    def __init__(self, model: str = None, max_ram: str = "4gb"):
        """
        Initialize Local LLM client.

        Args:
            model: Model name (ignored, languagemodels uses its default)
            max_ram: Maximum RAM to use (e.g., "4gb", "8gb")
        """
        if not LM_AVAILABLE:
            raise ImportError("languagemodels not installed. Run: pip install languagemodels")

        # Configure memory usage
        lm.config["max_ram"] = max_ram
        print(f"[OK] Local LLM initialized with max_ram: {max_ram}")

        # Test if model works
        try:
            test_response = lm.do("Hello, are you working?")
            print(f"[OK] Local LLM test successful: {test_response[:50]}...")
        except Exception as e:
            print(f"[WARNING] Local LLM test failed: {e}")

    def generate_explanation(
            self,
            shap_explanation: Dict,
            lime_explanation: Optional[Dict] = None,
            prediction: float = None,
            base_value: float = None,
            business_context: Optional[Dict] = None,
            max_tokens: int = 200
    ) -> str:
        """
        Generate business-friendly explanation using local LLM.

        Returns:
            Natural language explanation
        """
        # Extract top features from SHAP
        features = shap_explanation.get('features', {})
        top_features = list(features.items())[:6] if features else []

        # Build prompt
        prompt = self._build_prompt(
            top_features=top_features,
            prediction=prediction or shap_explanation.get('prediction', 0),
            base_value=base_value or shap_explanation.get('base_value', 0),
            business_context=business_context
        )

        try:
            # Generate using local model
            response = lm.do(prompt)

            # Clean up response
            explanation = response.strip()

            # If response is too long, truncate
            if len(explanation) > 500:
                explanation = explanation[:500] + "..."

            return explanation

        except Exception as e:
            print(f"[ERROR] Local LLM generation failed: {e}")
            # Fallback to mock
            from src.llm.summarizer import LLMSummarizer
            fallback = LLMSummarizer(provider='mock')
            return fallback._mock_explanation(shap_explanation, business_context)

    def _build_prompt(
            self,
            top_features: list,
            prediction: float,
            base_value: float,
            business_context: Optional[Dict] = None
    ) -> str:
        """Build the prompt for local LLM."""

        # Format features
        feature_lines = []
        for feature, details in top_features:
            shap_val = details.get('shap_value', 0)
            impact = "increases" if shap_val > 0 else "decreases"
            val = details.get('value', 'N/A')
            if isinstance(val, float):
                val = f"{val:.2f}"
            feature_lines.append(
                f"  - {feature}: {val} ({impact} price by ${abs(shap_val):.2f})"
            )

        features_text = "\n".join(feature_lines) if feature_lines else "No key factors identified."

        # Build clean prompt
        prompt = f"""You are a pricing expert at a retail company. Explain why an AI recommended a specific price.

RECOMMENDED PRICE: ${prediction:.2f}
BASE PRICE: ${base_value:.2f}

KEY FACTORS:
{features_text}

TASK:
Provide a short, business-friendly explanation (max 150 words) that tells a manager:
1. Why this price was recommended
2. Which factors had the biggest impact
3. One action item

RULES:
- Use simple language. No jargon like "SHAP" or "elasticity".
- Say "price sensitivity" instead of "elasticity".
- Say "demand patterns" instead of "seasonality".
- Keep it short and practical.

EXPLANATION:"""

        return prompt

    def generate_batch_summary(self, explanations: list) -> str:
        """Generate summary for multiple predictions."""
        if not explanations:
            return "No explanations provided."

        summary_data = {
            "total_items": len(explanations),
            "avg_price": sum(e.get('prediction', 0) for e in explanations) / len(explanations)
        }

        prompt = f"""Summarize these pricing insights:

Total Items: {summary_data['total_items']}
Average Price: ${summary_data['avg_price']:.2f}

Provide a very short summary (max 80 words) for the pricing team."""

        try:
            response = lm.do(prompt)
            return response.strip()
        except:
            return f"Summary: {summary_data['total_items']} items, average price ${summary_data['avg_price']:.2f}"