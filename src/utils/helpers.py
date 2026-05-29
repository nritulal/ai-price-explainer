"""Helper utility functions"""

import json
import pickle
from pathlib import Path
from typing import Any, Dict
import pandas as pd
import numpy as np


def save_json(data: Any, filepath: Path):
    """Save data as JSON"""
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2, default=str)


def load_json(filepath: Path) -> Dict:
    """Load JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)


def ensure_dir(path: Path):
    """Ensure directory exists"""
    path.mkdir(parents=True, exist_ok=True)


def save_pickle(obj: Any, filepath: Path):
    """Save object as pickle"""
    with open(filepath, 'wb') as f:
        pickle.dump(obj, f)


def load_pickle(filepath: Path):
    """Load pickle file"""
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def calculate_confidence_score(prediction: float, historical_range: tuple) -> float:
    """Calculate confidence score based on historical range"""
    min_price, max_price = historical_range
    range_width = max_price - min_price

    if range_width == 0:
        return 0.5

    # Closer to mean = higher confidence
    mean_price = (min_price + max_price) / 2
    distance_from_mean = abs(prediction - mean_price)
    confidence = 1 - (distance_from_mean / range_width)

    return np.clip(confidence, 0.3, 0.95)


def format_currency(value: float) -> str:
    """Format value as currency"""
    return f"${value:,.2f}"


def format_percentage(value: float) -> str:
    """Format value as percentage"""
    return f"{value * 100:.1f}%"