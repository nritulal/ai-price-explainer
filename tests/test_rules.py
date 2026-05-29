"""Unit tests for business rules engine"""

import unittest
from src.business.rules_engine import BusinessRulesEngine


class TestBusinessRules(unittest.TestCase):

    def setUp(self):
        """Setup test cases"""
        self.engine = BusinessRulesEngine()

    def test_price_caps(self):
        """Test min/max price caps"""
        # Test min price
        result = self.engine.apply_rules(0.1)
        self.assertGreaterEqual(result['adjusted_price'], 0.5)

        # Test max price
        result = self.engine.apply_rules(200)
        self.assertLessEqual(result['adjusted_price'], 100)

    def test_price_change_limits(self):
        """Test price change percentage limits"""
        # Test increase limit
        result = self.engine.apply_rules(12, current_price=10)
        self.assertLessEqual(result['price_change_pct'], 0.20)

        # Test decrease limit
        result = self.engine.apply_rules(5, current_price=10)
        self.assertGreaterEqual(result['price_change_pct'], -0.15)

    def test_inventory_adjustment(self):
        """Test inventory-based adjustments"""
        # High inventory should reduce price
        result = self.engine.apply_rules(10, inventory_level=20000)
        self.assertLess(result['adjusted_price'], 10)

        # Low inventory should increase price
        result = self.engine.apply_rules(10, inventory_level=100)
        self.assertGreater(result['adjusted_price'], 10)


if __name__ == '__main__':
    unittest.main()