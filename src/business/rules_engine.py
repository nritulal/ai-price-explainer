"""Business rules enforcement for pricing"""

import numpy as np
from src.utils.constants import BUSINESS_RULES


class BusinessRulesEngine:
    def __init__(self, rules: dict = None):
        self.rules = rules or BUSINESS_RULES

    def apply_rules(self, predicted_price: float, current_price: float = None,
                    inventory_level: float = None, is_holiday: bool = False) -> dict:
        """Apply business constraints to predicted price"""

        original_price = predicted_price

        # Min/max price caps
        predicted_price = max(predicted_price, self.rules['min_price'])
        predicted_price = min(predicted_price, self.rules['max_price'])

        # Price change limits (if current price provided)
        price_change_pct = 0
        rules_applied = []

        if current_price and current_price > 0:
            price_change_pct = (predicted_price - current_price) / current_price

            if price_change_pct > self.rules['max_price_increase_pct']:
                predicted_price = current_price * (1 + self.rules['max_price_increase_pct'])
                price_change_pct = self.rules['max_price_increase_pct']
                rules_applied.append(f"Price increase capped at {self.rules['max_price_increase_pct'] * 100}%")
            elif price_change_pct < -self.rules['max_price_decrease_pct']:
                predicted_price = current_price * (1 - self.rules['max_price_decrease_pct'])
                price_change_pct = -self.rules['max_price_decrease_pct']
                rules_applied.append(f"Price decrease capped at {self.rules['max_price_decrease_pct'] * 100}%")

        # Inventory-based adjustment
        if inventory_level is not None:
            if inventory_level > 10000:  # High inventory - discount
                discount = min(0.15, inventory_level / 100000)
                predicted_price = predicted_price * (1 - discount)
                rules_applied.append(f"High inventory discount: {discount * 100:.1f}%")
            elif inventory_level < 500:  # Low inventory - slight premium
                premium = min(0.10, (500 - inventory_level) / 5000)
                predicted_price = predicted_price * (1 + premium)
                rules_applied.append(f"Low inventory premium: {premium * 100:.1f}%")

        # Holiday adjustment
        if is_holiday:
            predicted_price = predicted_price * 0.95  # 5% holiday discount
            rules_applied.append("Holiday discount: 5%")

        return {
            'original_price': original_price,
            'adjusted_price': predicted_price,
            'price_change_pct': price_change_pct,
            'price_change_abs': predicted_price - original_price,
            'rules_applied': rules_applied,
            'min_max_cap_applied': original_price != predicted_price and len(rules_applied) == 0
        }

    def validate_explanation(self, explanation: dict) -> dict:
        """Validate if explanation aligns with business rules"""

        warnings = []

        # Check for extreme values
        for feature, details in explanation.get('features', {}).items():
            if abs(details.get('shap_value', 0)) > 2.0:
                warnings.append(f"Extreme impact from {feature}: {details['shap_value']:.2f}")

            # Check for unrealistic values
            if feature == 'elasticity' and abs(details.get('value', 0)) > 3:
                warnings.append(f"Unusual elasticity value: {details['value']:.2f}")

        return {
            'valid': len(warnings) == 0,
            'warnings': warnings,
            'severity': 'high' if len(warnings) > 2 else 'medium' if warnings else 'low'
        }