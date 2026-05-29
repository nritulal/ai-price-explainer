"""Feature engineering and column derivation for M5 data"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from typing import Dict, Tuple
import warnings

warnings.filterwarnings('ignore')

# Set pandas options for better performance
pd.options.mode.chained_assignment = None


class FeatureEngineer:
    """Derive missing features from M5 dataset"""

    def __init__(self, category_margins: Dict):
        from src.utils.constants import CATEGORY_MARGINS
        self.category_margins = category_margins or CATEGORY_MARGINS
        self.label_encoders = {}

    def derive_cost(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Derive cost from sell_price using category-based margin assumptions
        Logic: cost = price * (1 - margin_ratio)
        """
        df['cost'] = df['sell_price'] * (1 - df['margin_ratio'])
        # Add small noise to avoid perfect correlation
        np.random.seed(42)
        df['cost'] = df['cost'] * np.random.uniform(0.98, 1.02, len(df))
        return df

    def derive_inventory(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Derive inventory using sales patterns
        Logic: inventory_t = sales_{t-1} * days_of_cover + noise
        """
        df = df.sort_values(['item_id', 'store_id', 'date']).reset_index(drop=True)

        # Calculate rolling average sales (7-day)
        df['sales_ma7'] = df.groupby(['item_id', 'store_id'])['sales'].transform(
            lambda x: x.rolling(7, min_periods=1).mean()
        )

        # Initial inventory assumption (2 weeks of sales)
        df['inventory'] = df['sales_ma7'] * 14

        # Fill first week with reasonable assumption
        df['inventory'] = df['inventory'].fillna(df['sales'] * 21)

        # Add inventory dynamics (replenishment logic)
        df['inventory'] = df['inventory'] - df['sales'].cumsum()
        df['inventory'] = df.groupby(['item_id', 'store_id'])['inventory'].cumsum()

        # Ensure inventory doesn't go negative
        df['inventory'] = df.groupby(['item_id', 'store_id'])['inventory'].transform(
            lambda x: x.clip(lower=x.min() + 100)
        )

        # Add random noise
        np.random.seed(42)
        df['inventory'] = df['inventory'] * np.random.uniform(0.9, 1.1, len(df))
        df['inventory'] = df['inventory'].clip(lower=df['sales'] * 2)

        # Drop intermediate column
        df = df.drop('sales_ma7', axis=1)

        return df

    def derive_has_promo(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract promo flag from event columns"""
        # Check if event columns exist
        promo_cols = ['event_type_1', 'event_type_2', 'event_name_1']
        existing_cols = [col for col in promo_cols if col in df.columns]

        if not existing_cols:
            # If no promo data, create random but realistic promo flags
            np.random.seed(42)
            df['has_promo'] = np.random.choice([0, 1], size=len(df), p=[0.85, 0.15])
        else:
            df['has_promo'] = 0
            for col in existing_cols:
                df['has_promo'] = df['has_promo'] | (df[col].notna())

            # Special holiday detection
            if 'event_name_1' in df.columns:
                df['has_promo'] = df['has_promo'] | (
                    df['event_name_1'].isin(['Christmas', 'Thanksgiving', 'Super Bowl'])
                )

        df['has_promo'] = df['has_promo'].astype(int)
        return df

    def derive_seasonality_index(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate weekly seasonality index
        Logic: sales_ratio = sales / rolling_13week_avg_sales
        """
        df = df.sort_values(['item_id', 'store_id', 'date']).reset_index(drop=True)

        # Rolling 13-week average sales (quarterly seasonality)
        df['sales_ma13'] = df.groupby(['item_id', 'store_id'])['sales'].transform(
            lambda x: x.rolling(13, min_periods=1).mean()
        )

        # Avoid division by zero
        df['sales_ma13'] = df['sales_ma13'].replace(0, 1)
        df['seasonality_index'] = df['sales'] / df['sales_ma13']
        df['seasonality_index'] = df['seasonality_index'].clip(0.3, 2.5)

        # Add day-of-week pattern
        if 'date' in df.columns:
            df['day_of_week'] = pd.to_datetime(df['date']).dt.dayofweek
            dow_multiplier = {0: 1.0, 1: 1.1, 2: 1.1, 3: 1.05, 4: 1.2, 5: 1.5, 6: 0.8}
            df['seasonality_index'] = df['seasonality_index'] * df['day_of_week'].map(dow_multiplier)
            df = df.drop('day_of_week', axis=1)

        # Drop intermediate column
        df = df.drop('sales_ma13', axis=1)

        return df

    def derive_elasticity(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate price elasticity per item using log-log regression
        Logic: ln(sales) = B * ln(price) + a, elasticity = B
        """
        elasticities = {}

        for (item, store), group in df.groupby(['item_id', 'store_id']):
            if len(group) > 10:  # Need sufficient data points
                # Filter out zero/negative values
                group_clean = group[(group['sales'] > 0) & (group['sell_price'] > 0)].copy()

                if len(group_clean) > 5:
                    # Log transform
                    log_sales = np.log(group_clean['sales'].values)
                    log_price = np.log(group_clean['sell_price'].values)

                    # Simple regression
                    numerator = np.sum((log_sales - log_sales.mean()) * (log_price - log_price.mean()))
                    denominator = np.sum((log_price - log_price.mean()) ** 2)

                    if denominator != 0:
                        elasticity = numerator / denominator
                    else:
                        elasticity = -1.0
                else:
                    elasticity = -1.0
            else:
                elasticity = -1.2  # Default for new items

            # Clip to realistic range
            elasticity = np.clip(elasticity, -2.5, -0.3)
            elasticities[(item, store)] = elasticity

        df['elasticity'] = df.apply(
            lambda row: elasticities.get((row['item_id'], row['store_id']), -1.2),
            axis=1
        )

        return df

    def derive_competitor_price(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Estimate competitor prices based on store and category
        Logic: competitor_price = price * (1 + competitor_diff)
        """
        # Competitor price varies by store and category
        competitor_diffs = {}
        np.random.seed(42)

        for (store, cat), group in df.groupby(['store_id', 'category']):
            # Random but realistic competitor difference (-15% to +30%)
            diff = np.random.uniform(-0.15, 0.30)
            competitor_diffs[(store, cat)] = diff

        df['competitor_price'] = df.apply(
            lambda row: row['sell_price'] * (1 + competitor_diffs.get((row['store_id'], row['category']), 0.05)),
            axis=1
        )

        # Add some noise
        df['competitor_price'] = df['competitor_price'] * np.random.uniform(0.95, 1.05, len(df))
        df['competitor_price'] = df['competitor_price'].clip(lower=df['sell_price'] * 0.7)

        return df

    def add_category_info(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract category from item_id and assign margins"""
        # M5 item_id format: e.g., 'FOODS_1_001'
        df['category'] = df['item_id'].str.extract(r'([A-Z]+_\d+)')[0]
        df['category'] = df['category'].fillna('default')

        # Add margin ratios
        df['margin_ratio'] = df['category'].map(self.category_margins).fillna(self.category_margins['default'])

        return df

    def preprocess_m5_data(self, sales_df: pd.DataFrame, prices_df: pd.DataFrame,
                           calendar_df: pd.DataFrame, sample_size: int = None) -> pd.DataFrame:
        """
        Complete preprocessing pipeline for M5 data
        """
        print("Starting M5 data preprocessing...")

        # Optionally sample for faster development
        if sample_size and sample_size < len(sales_df):
            sales_df = sales_df.sample(n=sample_size, random_state=42)
            print(f"Using sample of {sample_size} items")

        # Melt sales data from wide to long format
        print("Melting sales data (this may take a moment)...")
        id_vars = ['item_id', 'store_id']
        sales_cols = [col for col in sales_df.columns if col.startswith('d_')]

        # Take only first 100 days to save memory and time
        if len(sales_cols) > 100:
            sales_cols = sales_cols[:100]
            print(f"Limiting to first 100 days to save memory")

        sales_long = pd.melt(
            sales_df,
            id_vars=id_vars,
            value_vars=sales_cols,
            var_name='day',
            value_name='sales'
        )

        # Extract day number from 'd_1', 'd_2', etc. - convert to int properly
        sales_long['day_num'] = sales_long['day'].apply(lambda x: int(x.split('_')[1]))

        # Process calendar - extract day number safely
        calendar_df = calendar_df.copy()

        # Check if 'd' column contains strings like 'd_1' or just numbers
        sample_val = calendar_df['d'].iloc[0]
        if isinstance(sample_val, str) and '_' in sample_val:
            calendar_df['day_num'] = calendar_df['d'].apply(lambda x: int(x.split('_')[1]))
        else:
            calendar_df['day_num'] = pd.to_numeric(calendar_df['d'], errors='coerce').astype(int)

        # Select relevant calendar columns
        calendar_cols = ['day_num', 'date', 'wm_yr_wk']
        if 'event_name_1' in calendar_df.columns:
            calendar_cols.extend(['event_name_1', 'event_type_1'])
        if 'event_name_2' in calendar_df.columns:
            calendar_cols.extend(['event_name_2', 'event_type_2'])

        # Merge with calendar
        sales_long = sales_long.merge(
            calendar_df[calendar_cols],
            on='day_num',
            how='left'
        )

        # Prepare prices - ensure wm_yr_wk is integer
        prices_df = prices_df.copy()
        prices_df['wm_yr_wk'] = pd.to_numeric(prices_df['wm_yr_wk'], errors='coerce').fillna(0).astype(int)

        # Merge with prices
        sales_long = sales_long.merge(
            prices_df[['item_id', 'store_id', 'wm_yr_wk', 'sell_price']],
            on=['item_id', 'store_id', 'wm_yr_wk'],
            how='left'
        )

        # Convert date
        sales_long['date'] = pd.to_datetime(sales_long['date'])
        sales_long['week'] = sales_long['date'].dt.isocalendar().week
        sales_long['year'] = sales_long['date'].dt.year

        # Add category info
        sales_long = self.add_category_info(sales_long)

        # Drop rows with missing critical values
        initial_len = len(sales_long)
        sales_long = sales_long.dropna(subset=['sales', 'sell_price'])
        sales_long = sales_long[sales_long['sales'] >= 0]
        print(f"Dropped {initial_len - len(sales_long)} rows with missing values")

        # Derive all features
        print("Deriving features...")
        sales_long = self.derive_has_promo(sales_long)
        sales_long = self.derive_cost(sales_long)
        sales_long = self.derive_elasticity(sales_long)
        sales_long = self.derive_seasonality_index(sales_long)
        sales_long = self.derive_inventory(sales_long)
        sales_long = self.derive_competitor_price(sales_long)

        # Create target variable (price)
        sales_long['price'] = sales_long['sell_price']

        # Select required columns
        required_cols = [
            'item_id', 'store_id', 'date', 'week', 'sales', 'cost', 'inventory',
            'has_promo', 'seasonality_index', 'elasticity', 'competitor_price', 'price'
        ]

        result_df = sales_long[required_cols].copy()

        # Handle any remaining NaN - FIXED for new pandas version
        result_df = result_df.ffill().bfill()

        # Encode categorical variables
        le_item = LabelEncoder()
        le_store = LabelEncoder()

        result_df['item_encoded'] = le_item.fit_transform(result_df['item_id'])
        result_df['store_encoded'] = le_store.fit_transform(result_df['store_id'])

        # Save encoders for later use
        self.label_encoders['item'] = le_item
        self.label_encoders['store'] = le_store

        print(f"\nPreprocessing complete!")
        print(f"  Shape: {result_df.shape}")
        print(f"  Date range: {result_df['date'].min()} to {result_df['date'].max()}")
        print(f"  Unique items: {result_df['item_id'].nunique()}")
        print(f"  Unique stores: {result_df['store_id'].nunique()}")

        return result_df