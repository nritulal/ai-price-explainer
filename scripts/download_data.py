"""Check and verify existing M5 data - NO DOWNLOAD"""

import os
from pathlib import Path


def verify_m5_data():
    """Verify that M5 data exists in the raw folder"""
    data_path = Path("data/raw/m5_walmart")

    required_files = [
        'calendar.csv',
        'sales_train_evaluation.csv',
        'sales_train_validation.csv',
        'sell_prices.csv'
    ]

    print("Checking for M5 dataset files...")
    missing_files = []

    for file in required_files:
        file_path = data_path / file
        if file_path.exists():
            size = file_path.stat().st_size / (1024 * 1024)  # MB
            print(f"✓ Found {file} ({size:.2f} MB)")
        else:
            print(f"✗ Missing {file}")
            missing_files.append(file)

    if missing_files:
        print("\n⚠️ WARNING: Missing files detected!")
        print("Please ensure your M5 data is in: data/raw/m5_walmart/")
        print("Required files:", required_files)
        return False
    else:
        print("\n✅ All M5 data files verified!")
        return True


def get_data_summary():
    """Print summary of available data"""
    data_path = Path("data/raw/m5_walmart")

    import pandas as pd

    # Load and display basic info
    print("\n📊 Data Summary:")
    print("=" * 50)

    # Calendar
    calendar = pd.read_csv(data_path / 'calendar.csv')
    print(f"Calendar: {len(calendar)} rows, {calendar.shape[1]} columns")
    print(f"  Date range: {calendar['date'].min()} to {calendar['date'].max()}")

    # Sales
    sales = pd.read_csv(data_path / 'sales_train_evaluation.csv')
    print(f"Sales: {sales.shape[0]} items/stores, {sales.shape[1]} columns")

    # Prices
    prices = pd.read_csv(data_path / 'sell_prices.csv')
    print(f"Prices: {len(prices)} unique price records")
    print(f"  Unique items: {prices['item_id'].nunique()}")
    print(f"  Unique stores: {prices['store_id'].nunique()}")


if __name__ == "__main__":
    if verify_m5_data():
        get_data_summary()
    else:
        print("\n📌 Please place your downloaded M5 files in: data/raw/m5_walmart/")
        print("   You can download from: https://www.kaggle.com/competitions/m5-forecasting-accuracy/data")