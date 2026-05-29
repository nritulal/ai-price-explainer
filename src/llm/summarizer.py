"""LLM-based explanation summarization"""

import json
from typing import Dict
import os
import numpy as np


class LLMSummarizer:
    def __init__(self, provider='mock', api_key=None):
        self.provider = provider

        if provider == 'openai' and api_key:
            try:
                import openai
                openai.api_key = api_key
                self.client = openai
            except ImportError:
                print("OpenAI not installed. Using mock mode.")
                self.provider = 'mock'

    def generate_explanation(self, shap_explanation: Dict, lime_explanation: Dict = None,
                             business_context: Dict = None) -> str:
        """Generate human-readable explanation"""

        if self.provider == 'mock':
            return self._mock_explanation(shap_explanation, business_context)
        else:
            return self._llm_explanation(shap_explanation, lime_explanation, business_context)

    def _mock_explanation(self, shap_explanation: Dict, business_context: Dict = None) -> str:
        """Mock LLM explanation for testing"""

        prediction = shap_explanation['prediction']
        base_value = shap_explanation['base_value']

        # Get top 3 features
        top_features = list(shap_explanation['features'].items())[:3]

        explanation_parts = []

        for feature, details in top_features:
            impact_symbol = "+" if details['impact'] == 'positive' else "-"
            explanation_parts.append(
                f"  * {impact_symbol} {feature}: {details['value']:.2f} "
                f"({impact_symbol} price by ${abs(details['shap_value']):.2f})"
            )

        explanation = f"""
PRICE RECOMMENDATION: ${prediction:.2f}

Why this price?

The recommended price is based on the following key factors:

{chr(10).join(explanation_parts)}

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

    def _llm_explanation(self, shap_explanation: Dict, lime_explanation: Dict,
                         business_context: Dict = None) -> str:
        """Generate explanation using actual LLM"""

        # Get top features
        top_features = list(shap_explanation['features'].items())[:5]

        prompt = f"""
You are a pricing expert at a major retail company. Explain why the AI model recommended a specific price.

Recommended Price: ${shap_explanation['prediction']:.2f}
Base Price: ${shap_explanation['base_value']:.2f}

Key factors influencing this price:
{json.dumps(dict(top_features), indent=2)}

Business Context:
{json.dumps(business_context, indent=2) if business_context else 'Not provided'}

Please provide a clear, business-friendly explanation (max 150 words) that helps a retail manager understand:
1. Why this price was recommended
2. Which factors had the biggest impact
3. A simple action item or insight

Keep it professional but accessible. Avoid technical jargon like "SHAP values" or "elasticity coefficients" - instead say things like "price sensitivity" or "demand patterns".
        """

        try:
            if hasattr(self, 'client'):
                response = self.client.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    max_tokens=300
                )
                return response.choices[0].message.content
            else:
                return self._mock_explanation(shap_explanation, business_context)
        except Exception as e:
            print(f"LLM error: {e}")
            return self._mock_explanation(shap_explanation, business_context)

    def generate_business_summary(self, explanations: list) -> str:
        """Generate summary for multiple predictions"""

        if not explanations:
            return "No explanations provided."

        avg_price = np.mean([e['prediction'] for e in explanations])

        # Collect common factors
        all_factors = []
        for exp in explanations[:10]:  # Limit to first 10
            all_factors.extend(list(exp['features'].keys())[:2])

        from collections import Counter
        common_factors = Counter(all_factors).most_common(3)

        summary = f"""
PRICING SUMMARY REPORT

Total Items Analyzed: {len(explanations)}
Average Recommended Price: ${avg_price:.2f}

Key Drivers Across All Items:
{chr(10).join([f"  * {factor} (appeared in {count} cases)" for factor, count in common_factors])}

Actionable Insights:
1. Inventory Management: Consider bundling products with high price sensitivity
2. Competitive Strategy: Monitor competitor prices for elastic items  
3. Seasonal Planning: Adjust inventory based on seasonality signals

Recommendations:
- Review top 20% most price-sensitive items weekly
- Implement A/B testing for items with high elasticity
- Update competitor price monitoring frequency
        """

        return summary.strip()