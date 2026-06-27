"""Feature engineering and column derivation for M5 data - Enhanced Version"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from typing import Dict, Tuple
import warnings

warnings.filterwarnings('ignore')

# Set pandas options for better performance
pd.options.mode.chained_assignment = None


class FeatureEngineer:
    """Derive enhanced features from M5 dataset"""

    def __init__(self, category_margins: Dict):
        from src.utils.constants import CATEGORY_MARGINS
        self.category_margins = category_margins or CATEGORY_MARGINS
        self.label_encoders = {}

    def derive_sales_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Enhanced sales features to capture demand signals
        Sales volume should be a key pricing factor (15-25% importance)
        """
        df = df.sort_values(['item_id', 'store_id', 'date']).reset_index(drop=True)

        # Log transformation to handle wide range of sales values
        df['sales_log'] = np.log1p(df['sales'])

        # Rolling average (7-day) - captures demand trend
        df['sales_ma7'] = df.groupby(['item_id', 'store_id'])['sales'].transform(
            lambda x: x.rolling(7, min_periods=1).mean()
        )
        df['sales_ma7'] = df['sales_ma7'].fillna(df['sales'])
        df['sales_ma7_log'] = np.log1p(df['sales_ma7'])

        # Sales growth rate - captures momentum
        df['sales_growth'] = df.groupby(['item_id', 'store_id'])['sales'].pct_change().fillna(0)
        df['sales_growth'] = df['sales_growth'].clip(-0.5, 0.5)  # Cap extreme changes

        # Sales volatility - how much sales fluctuate
        df['sales_volatility'] = df.groupby(['item_id', 'store_id'])['sales'].transform(
            lambda x: x.rolling(7, min_periods=1).std().fillna(0)
        )
        df['sales_volatility'] = df['sales_volatility'] / (df['sales_ma7'] + 1)

        # Normalized sales within item-store (relative demand)
        df['sales_normalized'] = df.groupby(['item_id', 'store_id'])['sales'].transform(
            lambda x: (x - x.mean()) / (x.std() + 1)
        )
        df['sales_normalized'] = df['sales_normalized'].fillna(0)

        # Sales trend (increasing/decreasing)
        df['sales_ma7_prev'] = df.groupby(['item_id', 'store_id'])['sales_ma7'].shift(1)
        df['sales_trend'] = ((df['sales_ma7'] - df['sales_ma7_prev']) / (df['sales_ma7_prev'] + 1)).fillna(0)
        df['sales_trend'] = df['sales_trend'].clip(-0.5, 0.5)

        # Drop intermediate columns
        df = df.drop(['sales_ma7_prev'], axis=1, errors='ignore')

        return df

    def derive_cost(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Derive cost from sell_price using category-based margin assumptions
        Cost should be a key pricing factor (20-30% importance)
        """
        df['cost'] = df['sell_price'] * (1 - df['margin_ratio'])

        # Add small noise to avoid perfect correlation
        np.random.seed(42)
        df['cost'] = df['cost'] * np.random.uniform(0.98, 1.02, len(df))

        # Cost as percentage of price
        df['cost_ratio'] = df['cost'] / (df['sell_price'] + 0.01)
        df['cost_ratio'] = df['cost_ratio'].clip(0, 1)

        return df

    def derive_inventory_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Enhanced inventory features with ratio-based indicators
        Inventory should be moderately important (5-10% importance)
        """
        df = df.sort_values(['item_id', 'store_id', 'date']).reset_index(drop=True)

        # Calculate rolling average sales (7-day)
        df['sales_ma7'] = df.groupby(['item_id', 'store_id'])['sales'].transform(
            lambda x: x.rolling(7, min_periods=1).mean()
        )
        df['sales_ma7'] = df['sales_ma7'].fillna(df['sales'])

        # Initial inventory assumption (2 weeks of sales)
        df['inventory'] = df['sales_ma7'] * 14
        df['inventory'] = df['inventory'].fillna(df['sales'] * 21)

        # Inventory dynamics (replenishment logic)
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

        # Inventory ratio (days of cover)
        df['inventory_ratio'] = df['inventory'] / (df['sales_ma7'] + 1)
        df['inventory_ratio'] = df['inventory_ratio'].clip(0, 100)

        # Inventory status (low/medium/high) - FIXED: Handle NaN values
        df['inventory_status'] = pd.cut(
            df['inventory_ratio'].fillna(1),  # Fill NaN with 1 (medium)
            bins=[-1, 5, 20, 100],  # Use -1 as lower bound to catch 0 values
            labels=[0, 1, 2]  # 0=low, 1=medium, 2=high
        )
        # Convert to int, filling any remaining NaN with 1
        df['inventory_status'] = df['inventory_status'].fillna(1).astype(int)

        # Drop intermediate column
        df = df.drop('sales_ma7', axis=1)

        return df

    def derive_promo_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Enhanced promotion features with intensity scoring
        Promotions should be moderately important (5-10% importance)
        """
        # Initialize promo flags
        df['has_promo'] = 0
        df['promo_intensity'] = 0.0

        # Check event columns
        if 'event_type_1' in df.columns:
            df['has_promo'] = df['has_promo'] | (df['event_type_1'].notna())

            # Promo intensity based on event type
            promo_intensity_map = {
                'National holiday': 0.8,
                'Religious holiday': 0.6,
                'Sporting': 0.5,
                'Cultural': 0.4,
                'Regional': 0.3
            }
            df['promo_intensity'] = df['event_type_1'].map(promo_intensity_map).fillna(0)

        # Special holiday detection
        if 'event_name_1' in df.columns:
            holiday_map = {
                'Christmas': 0.9,
                'Thanksgiving': 0.8,
                'Super Bowl': 0.7,
                'Easter': 0.6,
                'Valentine': 0.5,
                'Labor Day': 0.4,
                'Memorial Day': 0.4
            }
            holiday_intensity = df['event_name_1'].map(holiday_map).fillna(0)
            df['promo_intensity'] = df['promo_intensity'] + holiday_intensity
            df['promo_intensity'] = df['promo_intensity'].clip(0, 1)

            # Flag major holidays
            df['is_major_holiday'] = df['event_name_1'].isin(['Christmas', 'Thanksgiving', 'Super Bowl']).astype(int)

        # If no promo data, create realistic synthetic promos
        if df['has_promo'].sum() == 0:
            np.random.seed(42)
            # 15% of days have promos
            df['has_promo'] = np.random.choice([0, 1], size=len(df), p=[0.85, 0.15])
            df['promo_intensity'] = df['has_promo'] * np.random.uniform(0.2, 0.8, len(df))

        # Create promo price adjustment factor
        df['promo_factor'] = 1 - (df['promo_intensity'] * 0.15)  # Up to 15% discount

        # Promo frequency (rolling 30-day)
        df['promo_frequency'] = df.groupby(['item_id', 'store_id'])['has_promo'].transform(
            lambda x: x.rolling(30, min_periods=1).mean()
        )
        df['promo_frequency'] = df['promo_frequency'].fillna(0)

        df['has_promo'] = df['has_promo'].astype(int)

        return df

    def derive_seasonality_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Enhanced seasonality with multiple time components
        Seasonality should be moderately important (5-10% importance)
        """
        df = df.sort_values(['item_id', 'store_id', 'date']).reset_index(drop=True)

        # 1. Quarterly seasonality (13-week rolling)
        df['sales_ma13'] = df.groupby(['item_id', 'store_id'])['sales'].transform(
            lambda x: x.rolling(13, min_periods=1).mean()
        )
        df['sales_ma13'] = df['sales_ma13'].replace(0, 1)
        df['seasonality_quarterly'] = df['sales'] / df['sales_ma13']
        df['seasonality_quarterly'] = df['seasonality_quarterly'].clip(0.3, 2.5)

        # 2. Weekly seasonality
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df['day_of_week'] = df['date'].dt.dayofweek
            df['week_of_year'] = df['date'].dt.isocalendar().week
            df['month'] = df['date'].dt.month

            # Day-of-week multipliers
            dow_multiplier = {0: 1.0, 1: 1.1, 2: 1.1, 3: 1.05, 4: 1.2, 5: 1.5, 6: 0.8}
            df['seasonality_weekly'] = df['day_of_week'].map(dow_multiplier)

        # 3. Monthly seasonality
        month_multiplier = {
            1: 0.9, 2: 0.9, 3: 1.0, 4: 1.0, 5: 1.1, 6: 1.1,
            7: 1.0, 8: 0.9, 9: 1.0, 10: 1.1, 11: 1.2, 12: 1.3
        }
        df['seasonality_monthly'] = df['month'].map(month_multiplier)

        # Combined seasonality index
        df['seasonality_index'] = (
                df['seasonality_quarterly'] *
                df['seasonality_weekly'] *
                df['seasonality_monthly']
        )
        df['seasonality_index'] = df['seasonality_index'].clip(0.3, 2.5)

        # Seasonality strength (how much it deviates from 1.0)
        df['seasonality_strength'] = np.abs(df['seasonality_index'] - 1.0)

        # Drop intermediate columns
        df = df.drop([
            'sales_ma13', 'seasonality_quarterly', 'seasonality_weekly',
            'seasonality_monthly', 'day_of_week', 'week_of_year', 'month'
        ], axis=1, errors='ignore')

        return df

    def derive_elasticity_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Enhanced elasticity with multiple estimation methods
        Elasticity should be important (10-15% importance)
        """
        elasticities = {}
        elasticity_confidence = {}

        for (item, store), group in df.groupby(['item_id', 'store_id']):
            if len(group) > 20:
                # Filter valid data
                group_clean = group[(group['sales'] > 0) & (group['sell_price'] > 0)].copy()

                if len(group_clean) > 10:
                    # Method 1: Log-log regression
                    log_sales = np.log(group_clean['sales'].values)
                    log_price = np.log(group_clean['sell_price'].values)
                    numerator = np.sum((log_sales - log_sales.mean()) * (log_price - log_price.mean()))
                    denominator = np.sum((log_price - log_price.mean()) ** 2)
                    elasticity1 = numerator / denominator if denominator != 0 else -1.0

                    # Method 2: Percentage change method
                    sales_pct = group_clean['sales'].pct_change().fillna(0)
                    price_pct = group_clean['sell_price'].pct_change().fillna(0)
                    valid = (price_pct != 0) & (sales_pct != 0)
                    if valid.any():
                        elasticity2 = (sales_pct[valid] / price_pct[valid]).mean()
                    else:
                        elasticity2 = -1.0

                    # Average the methods
                    elasticity = (elasticity1 + elasticity2) / 2

                    # Confidence based on data quality
                    confidence = min(1.0, len(group_clean) / 50)
                else:
                    elasticity = -1.0
                    confidence = 0.3
            else:
                elasticity = -1.2  # Default for new items
                confidence = 0.2

            # Clip to realistic range
            elasticity = np.clip(elasticity, -2.5, -0.3)
            elasticities[(item, store)] = elasticity
            elasticity_confidence[(item, store)] = confidence

        df['elasticity'] = df.apply(
            lambda row: elasticities.get((row['item_id'], row['store_id']), -1.2),
            axis=1
        )

        # Elasticity magnitude (absolute value) - higher = more sensitive
        df['elasticity_magnitude'] = df['elasticity'].abs()

        # Elasticity confidence (how reliable is this estimate)
        df['elasticity_confidence'] = df.apply(
            lambda row: elasticity_confidence.get((row['item_id'], row['store_id']), 0.2),
            axis=1
        )

        # Elasticity category (low/medium/high sensitivity) - FIXED: Handle NaN
        df['elasticity_category'] = pd.cut(
            df['elasticity_magnitude'].fillna(0.5),
            bins=[-1, 0.5, 1.0, 2.5],
            labels=[0, 1, 2]  # 0=low, 1=medium, 2=high sensitivity
        )
        df['elasticity_category'] = df['elasticity_category'].fillna(1).astype(int)

        return df

    def derive_competitor_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Enhanced competitor price features with gap analysis
        Competitor price should be important (15-20% importance)
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

        # Price gap (competitor - our price)
        df['price_gap'] = df['competitor_price'] - df['sell_price']
        df['price_gap'] = df['price_gap'].clip(-5, 5)

        # Price gap ratio
        df['price_gap_ratio'] = df['price_gap'] / (df['sell_price'] + 0.01)
        df['price_gap_ratio'] = df['price_gap_ratio'].clip(-0.5, 0.5)

        # Competitive position (relative to competitor)
        df['competitive_position'] = (df['sell_price'] / (df['competitor_price'] + 0.01)).clip(0.5, 1.5)

        return df

    def add_category_info(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract category from item_id and assign margins
        Use category instead of individual item IDs
        """
        # Extract category from item_id
        df['category'] = df['item_id'].str.extract(r'([A-Z]+_\d+)')[0]
        df['category'] = df['category'].fillna('default')

        # Add margin ratios
        df['margin_ratio'] = df['category'].map(self.category_margins).fillna(self.category_margins['default'])

        # Encode category (not individual item)
        le_category = LabelEncoder()
        df['category_encoded'] = le_category.fit_transform(df['category'].astype(str))

        # Store encoder for later use
        self.label_encoders['category'] = le_category

        return df

    def preprocess_m5_data(self, sales_df: pd.DataFrame, prices_df: pd.DataFrame,
                           calendar_df: pd.DataFrame, sample_size: int = None) -> pd.DataFrame:
        """
        Complete preprocessing pipeline with enhanced features
        """
        print("Starting M5 data preprocessing with enhanced features...")

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

        # Extract day number from 'd_1', 'd_2', etc.
        sales_long['day_num'] = sales_long['day'].apply(lambda x: int(x.split('_')[1]))

        # Process calendar
        calendar_df = calendar_df.copy()
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

        # Prepare prices
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

        # Derive all enhanced features
        print("Deriving enhanced features...")
        sales_long = self.derive_promo_features(sales_long)
        sales_long = self.derive_cost(sales_long)
        sales_long = self.derive_elasticity_features(sales_long)
        sales_long = self.derive_seasonality_features(sales_long)
        sales_long = self.derive_inventory_features(sales_long)
        sales_long = self.derive_competitor_features(sales_long)
        sales_long = self.derive_sales_features(sales_long)

        # Create target variable (price)
        sales_long['price'] = sales_long['sell_price']

        # Select enhanced features (NO item_encoded or store_encoded!)
        required_cols = [
            'item_id', 'store_id', 'date', 'week', 'category_encoded',
            'sales_log', 'sales_ma7', 'sales_growth', 'sales_volatility',
            'sales_normalized', 'sales_trend',
            'cost', 'cost_ratio',
            'inventory', 'inventory_ratio', 'inventory_status',
            'has_promo', 'promo_intensity', 'promo_factor', 'promo_frequency',
            'seasonality_index', 'seasonality_strength',
            'elasticity', 'elasticity_magnitude', 'elasticity_confidence', 'elasticity_category',
            'competitor_price', 'price_gap', 'price_gap_ratio', 'competitive_position',
            'price'
        ]

        result_df = sales_long[required_cols].copy()

        # Handle any remaining NaN - FIXED: Use fillna with specific values
        # For numeric columns, fill with 0
        numeric_cols = result_df.select_dtypes(include=[np.number]).columns
        result_df[numeric_cols] = result_df[numeric_cols].fillna(0)

        # For object/string columns, fill with 'unknown'
        object_cols = result_df.select_dtypes(include=['object']).columns
        for col in object_cols:
            result_df[col] = result_df[col].fillna('unknown')

        # Save encoders
        import joblib
        import json
        from src.utils.constants import PROCESSED_DATA_PATH

        # Save category encoder
        if 'category' in self.label_encoders:
            joblib.dump(self.label_encoders['category'], PROCESSED_DATA_PATH / 'category_encoder.pkl')

            # Save category mapping
            category_mapping = {
                idx: category for idx, category in enumerate(self.label_encoders['category'].classes_)
            }
            with open(PROCESSED_DATA_PATH / 'category_mapping.json', 'w') as f:
                json.dump(category_mapping, f, indent=2)

        print(f"\nPreprocessing complete!")
        print(f"  Shape: {result_df.shape}")
        print(f"  Features: {result_df.columns.tolist()}")
        print(f"  Unique items: {result_df['item_id'].nunique()}")
        print(f"  Unique stores: {result_df['store_id'].nunique()}")
        print(f"  Unique categories: {result_df['category_encoded'].nunique()}")

        return result_df