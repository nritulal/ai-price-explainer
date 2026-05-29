"""Data loading and preparation for training"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import json
import warnings

warnings.filterwarnings('ignore')

from src.utils.constants import PROCESSED_DATA_PATH, CONFIG
from src.data.preprocessing import FeatureEngineer


class DataLoader:
    def __init__(self, config: dict = None):
        self.config = config or CONFIG
        self.scaler = StandardScaler()
        self.engineer = FeatureEngineer({})

    def load_raw_m5(self):
        """Load raw M5 data files from local directory"""
        raw_path = Path(self.config['data']['raw_path'])

        print(f"Loading data from: {raw_path}")

        # Check if files exist
        required_files = ['sales_train_evaluation.csv', 'sell_prices.csv', 'calendar.csv']
        for file in required_files:
            if not (raw_path / file).exists():
                raise FileNotFoundError(f"Missing required file: {file}")

        # Load with low memory options
        sales_train = pd.read_csv(raw_path / 'sales_train_evaluation.csv', low_memory=False)
        sell_prices = pd.read_csv(raw_path / 'sell_prices.csv', low_memory=False)
        calendar = pd.read_csv(raw_path / 'calendar.csv', low_memory=False)

        print(f"Loaded sales: {sales_train.shape}")
        print(f"Loaded prices: {sell_prices.shape}")
        print(f"Loaded calendar: {calendar.shape}")

        return sales_train, sell_prices, calendar

    def prepare_features(self, df: pd.DataFrame) -> tuple:
        """Prepare features for model training"""

        # Select features
        feature_cols = ['sales', 'cost', 'inventory', 'has_promo',
                        'seasonality_index', 'elasticity', 'competitor_price',
                        'item_encoded', 'store_encoded']

        # Ensure all columns exist
        missing_cols = [col for col in feature_cols if col not in df.columns]
        if missing_cols:
            print(f"Warning: Missing columns: {missing_cols}")
            # Try to create missing columns with defaults
            for col in missing_cols:
                if col in ['item_encoded', 'store_encoded']:
                    df[col] = 0
                else:
                    df[col] = 0.0

        X = df[feature_cols].copy()
        y = df['price'].copy()

        # Scale numerical features
        numerical_cols = ['sales', 'cost', 'inventory', 'seasonality_index',
                          'elasticity', 'competitor_price']

        # Only scale columns that exist
        existing_num_cols = [col for col in numerical_cols if col in X.columns]
        if existing_num_cols:
            X[existing_num_cols] = self.scaler.fit_transform(X[existing_num_cols])

        return X, y

    def get_train_val_test(self, df: pd.DataFrame):
        """Split data into train/val/test sets (time-based)"""

        # Sort by date
        df = df.sort_values('date').reset_index(drop=True)
        n = len(df)

        test_size = int(n * self.config['data']['test_size'])
        val_size = int(n * self.config['data']['validation_size'])

        # Ensure we have enough data
        if n < (test_size + val_size + 100):
            print(f"Warning: Limited data ({n} rows), adjusting split sizes")
            test_size = max(1, int(n * 0.1))
            val_size = max(1, int(n * 0.1))

        train_df = df[:-test_size - val_size].copy()
        val_df = df[-test_size - val_size:-test_size].copy() if test_size + val_size < n else df[-test_size:].copy()
        test_df = df[-test_size:].copy() if test_size <= n else df.copy()

        print(f"\nData Split:")
        print(f"  Train: {len(train_df):,} rows")
        print(f"  Val:   {len(val_df):,} rows")
        print(f"  Test:  {len(test_df):,} rows")

        X_train, y_train = self.prepare_features(train_df)
        X_val, y_val = self.prepare_features(val_df)
        X_test, y_test = self.prepare_features(test_df)

        # Save feature names
        feature_names = X_train.columns.tolist()
        PROCESSED_DATA_PATH.mkdir(parents=True, exist_ok=True)

        with open(PROCESSED_DATA_PATH / 'feature_names.json', 'w') as f:
            json.dump(feature_names, f, indent=2)

        return X_train, X_val, X_test, y_train, y_val, y_test, train_df, val_df, test_df

    def process_pipeline(self, sample_size: int = None):
        """Complete data processing pipeline"""
        print("=" * 60)
        print("Starting Data Processing Pipeline")
        print("=" * 60)

        # Load raw data
        sales_df, prices_df, calendar_df = self.load_raw_m5()

        # Use sample for faster development if specified
        if sample_size and sample_size < len(sales_df):
            sales_df = sales_df.sample(n=sample_size, random_state=42)
            print(f"Using sample of {sample_size} items for faster processing")

        # Feature engineering
        print("\nEngineering features...")
        processed_df = self.engineer.preprocess_m5_data(sales_df, prices_df, calendar_df)

        # Save processed data
        PROCESSED_DATA_PATH.mkdir(parents=True, exist_ok=True)
        processed_df.to_csv(PROCESSED_DATA_PATH / 'processed_data.csv', index=False)
        print(f"Saved processed data to {PROCESSED_DATA_PATH / 'processed_data.csv'}")

        # Split data
        X_train, X_val, X_test, y_train, y_val, y_test, train_df, val_df, test_df = self.get_train_val_test(
            processed_df)

        print("Data processing complete!")

        return X_train, X_val, X_test, y_train, y_val, y_test, train_df, val_df, test_df