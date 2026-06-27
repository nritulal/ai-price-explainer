"""Constants and configuration loading"""

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

# Add these to constants.py
ITEM_MAPPING_PATH = PROCESSED_DATA_PATH / 'item_mapping.json'
STORE_MAPPING_PATH = PROCESSED_DATA_PATH / 'store_mapping.json'
ITEM_ENCODER_PATH = PROCESSED_DATA_PATH / 'item_encoder.pkl'
STORE_ENCODER_PATH = PROCESSED_DATA_PATH / 'store_encoder.pkl'

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
    'cost': 'sell_price * (1 - margin_ratio), margin_ratio by category (25-40%)',
    'inventory': 'rolling_7day_sales * 14 - cumulative_sales + random_noise',
    'has_promo': 'from event_type_1/event_name_1 columns, special holidays',
    'seasonality_index': 'sales / rolling_13week_avg_sales * day_of_week_factor',
    'elasticity': 'log-log regression: β from log(sales) ~ log(price) per item-store',
    'competitor_price': 'sell_price * (1 + random_diff), diff ∈ [-0.15, +0.30]'
}