"""Constants and configuration loading - Enhanced Version"""

import yaml
import os
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Load config
with open(PROJECT_ROOT / "config" / "config.yaml", "r") as f:
    CONFIG = yaml.safe_load(f)

# Paths
RAW_DATA_PATH = PROJECT_ROOT / CONFIG["data"]["raw_path"]
PROCESSED_DATA_PATH = PROJECT_ROOT / CONFIG["data"]["processed_path"]
EXPERIMENTS_PATH = PROJECT_ROOT / "experiments"
LOGS_PATH = EXPERIMENTS_PATH / "logs"

# Feature names
INPUT_FEATURES = CONFIG["features"]["input_features"]
TARGET = CONFIG["features"]["target"]
CATEGORICAL_FEATURES = CONFIG["features"]["categorical_features"]
NUMERICAL_FEATURES = CONFIG["features"]["numerical_features"]

# Model params
MODEL_CONFIG = CONFIG["model"]

# Business rules
BUSINESS_RULES = CONFIG["business_rules"]

# Create directories
for path in [RAW_DATA_PATH, PROCESSED_DATA_PATH, LOGS_PATH]:
    path.mkdir(parents=True, exist_ok=True)

# Category to margin mapping (for cost derivation)
CATEGORY_MARGINS = {
    'HOBBIES_1': 0.35,
    'HOBBIES_2': 0.32,
    'FOODS_1': 0.28,
    'FOODS_2': 0.25,
    'FOODS_3': 0.30,
    'HOUSEHOLD_1': 0.38,
    'HOUSEHOLD_2': 0.40,
    'default': 0.30
}

# Column derivation logic summary (for documentation)
DERIVATION_LOGIC = {
    'sales_log': 'Log transformation of sales to handle wide range',
    'sales_ma7': '7-day rolling average for demand trend',
    'sales_growth': 'Sales growth rate for momentum',
    'sales_volatility': 'Standard deviation of sales for stability',
    'sales_normalized': 'Normalized sales within item-store',
    'sales_trend': 'Increasing/decreasing sales trend',
    'cost': 'sell_price * (1 - margin_ratio)',
    'cost_ratio': 'Cost as percentage of price',
    'inventory': 'Simulated inventory with replenishment',
    'inventory_ratio': 'Days of cover',
    'inventory_status': 'Low/medium/high inventory flag',
    'has_promo': 'Binary promotion indicator',
    'promo_intensity': 'Promotion intensity score (0-1)',
    'promo_factor': 'Price adjustment factor (0.85-1.0)',
    'promo_frequency': 'Rolling 30-day promotion frequency',
    'seasonality_index': 'Combined seasonality (quarterly * weekly * monthly)',
    'seasonality_strength': 'Deviation from normal (|index - 1.0|)',
    'elasticity': 'Price sensitivity (log-log regression)',
    'elasticity_magnitude': 'Absolute elasticity value',
    'elasticity_confidence': 'Confidence in elasticity estimate',
    'elasticity_category': 'Low/medium/high sensitivity category',
    'competitor_price': 'Estimated competitor price',
    'price_gap': 'Competitor price - our price',
    'price_gap_ratio': 'Relative price gap',
    'competitive_position': 'Our price / competitor price',
    'category_encoded': 'Product category (FOODS, HOBBIES, etc.)'
}