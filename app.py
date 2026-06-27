"""Flask web application - Without Confidence Module"""

from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import json
import sys
import traceback

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

# Import modules
from src.explainability.shap_explainer import SHAPExplainer
from src.explainability.lime_explainer import LIMExplainer
from src.explainability.comparator import ExplanationComparator
from src.business.rules_engine import BusinessRulesEngine
from src.utils.constants import PROCESSED_DATA_PATH, EXPERIMENTS_PATH

app = Flask(__name__)

# Global variables
model = None
feature_names = None
shap_explainer = None
lime_explainer = None
comparator = None
rules_engine = None
training_data = None
scaler = None

# Item and Store mappings
item_mapping = {}
store_mapping = {}
reverse_item_mapping = {}
reverse_store_mapping = {}
available_items = []
available_stores = []

# Feature display names
FEATURE_DISPLAY_NAMES = {
    'sales_log': 'Sales Volume (log)',
    'sales_ma7': 'Demand Trend',
    'sales_growth': 'Sales Growth Rate',
    'sales_volatility': 'Sales Stability',
    'sales_normalized': 'Relative Demand',
    'sales_trend': 'Sales Momentum',
    'cost': 'Product Cost',
    'cost_ratio': 'Cost-to-Price Ratio',
    'inventory': 'Current Inventory',
    'inventory_ratio': 'Days of Cover',
    'inventory_status': 'Inventory Level',
    'has_promo': 'Promotion Status',
    'promo_intensity': 'Promotion Intensity',
    'promo_factor': 'Promo Price Factor',
    'promo_frequency': 'Promotion Frequency',
    'seasonality_index': 'Seasonal Demand',
    'seasonality_strength': 'Seasonality Strength',
    'elasticity': 'Price Sensitivity',
    'elasticity_magnitude': 'Sensitivity Magnitude',
    'elasticity_confidence': 'Elasticity Confidence',
    'elasticity_category': 'Sensitivity Category',
    'competitor_price': 'Competitor Price',
    'price_gap': 'Price Gap',
    'price_gap_ratio': 'Price Gap Ratio',
    'competitive_position': 'Competitive Position',
    'category_encoded': 'Product Category'
}

HIDDEN_FEATURES = ['category_encoded']


def load_mappings():
    """Load item and store mappings"""
    global item_mapping, store_mapping, reverse_item_mapping, reverse_store_mapping
    global available_items, available_stores

    print("[INFO] Loading item and store mappings...")

    try:
        with open(PROCESSED_DATA_PATH / 'item_mapping.json', 'r') as f:
            item_mapping = json.load(f)
            item_mapping = {int(k): v for k, v in item_mapping.items()}
            available_items = list(item_mapping.values())[:50]
            reverse_item_mapping = {v: k for k, v in item_mapping.items()}
            print(f"[OK] Item mapping loaded: {len(available_items)} items")
    except FileNotFoundError:
        print("[WARNING] item_mapping.json not found. Creating defaults...")
        available_items = [
            'FOODS_1_001', 'FOODS_1_002', 'FOODS_1_003', 'FOODS_1_004', 'FOODS_1_005',
            'FOODS_1_006', 'FOODS_1_007', 'FOODS_1_008', 'FOODS_1_009', 'FOODS_1_010',
            'FOODS_2_001', 'FOODS_2_002', 'FOODS_2_003', 'FOODS_2_004', 'FOODS_2_005',
            'HOBBIES_1_001', 'HOBBIES_1_002', 'HOBBIES_1_003', 'HOBBIES_1_004', 'HOBBIES_1_005',
            'HOBBIES_2_001', 'HOBBIES_2_002', 'HOBBIES_2_003', 'HOBBIES_2_004', 'HOBBIES_2_005',
            'HOUSEHOLD_1_001', 'HOUSEHOLD_1_002', 'HOUSEHOLD_1_003', 'HOUSEHOLD_1_004', 'HOUSEHOLD_1_005'
        ]
        item_mapping = {i: item for i, item in enumerate(available_items)}
        reverse_item_mapping = {item: i for i, item in enumerate(available_items)}

    try:
        with open(PROCESSED_DATA_PATH / 'store_mapping.json', 'r') as f:
            store_mapping = json.load(f)
            store_mapping = {int(k): v for k, v in store_mapping.items()}
            available_stores = list(store_mapping.values())
            reverse_store_mapping = {v: k for k, v in store_mapping.items()}
            print(f"[OK] Store mapping loaded: {len(available_stores)} stores")
    except FileNotFoundError:
        print("[WARNING] store_mapping.json not found. Creating defaults...")
        available_stores = ['CA_1', 'CA_2', 'CA_3', 'CA_4', 'TX_1', 'TX_2', 'TX_3', 'WI_1', 'WI_2', 'WI_3']
        store_mapping = {i: store for i, store in enumerate(available_stores)}
        reverse_store_mapping = {store: i for i, store in enumerate(available_stores)}

    return True


def load_models():
    """Load all required models and components"""
    global model, feature_names, shap_explainer, lime_explainer, comparator
    global rules_engine, training_data, scaler

    print("\n" + "=" * 60)
    print("Loading AI-Powered Price Explanations System")
    print("=" * 60)

    if not load_mappings():
        print("[ERROR] Failed to load mappings")
        return False

    # Load feature names
    feature_path = PROCESSED_DATA_PATH / 'feature_names.json'
    if feature_path.exists():
        try:
            with open(feature_path, 'r') as f:
                feature_names = json.load(f)
            print(f"[OK] Feature names loaded: {len(feature_names)} features")
        except Exception as e:
            print(f"[ERROR] Could not load feature names: {e}")
            return False
    else:
        print("[ERROR] feature_names.json not found!")
        return False

    # Load trained model
    model_path = EXPERIMENTS_PATH / 'model_xgboost.pkl'
    if model_path.exists():
        try:
            model = joblib.load(model_path)
            print("[OK] XGBoost model loaded")
        except Exception as e:
            print(f"[ERROR] Could not load model: {e}")
            return False
    else:
        print("[ERROR] Trained model not found!")
        return False

    # Load scaler if available
    try:
        scaler = joblib.load(PROCESSED_DATA_PATH / 'scaler.pkl')
        print("[OK] Scaler loaded")
    except:
        print("[WARNING] Scaler not found, using raw features")
        scaler = None

    # Load training data
    data_path = PROCESSED_DATA_PATH / 'processed_data.csv'
    if data_path.exists():
        try:
            df = pd.read_csv(data_path, nrows=1000)
            feature_cols = [col for col in feature_names if col in df.columns]
            if feature_cols:
                training_data = df[feature_cols].values
                print(f"[OK] Training data loaded: {len(training_data)} samples")
            else:
                training_data = None
        except Exception as e:
            print(f"[WARNING] Could not load training data: {e}")
            training_data = None

    # Initialize SHAP explainer
    try:
        if training_data is not None and len(training_data) > 0:
            background = training_data[:min(100, len(training_data))]
        else:
            background = np.random.randn(100, len(feature_names))

        shap_explainer = SHAPExplainer(model, background)
        print("[OK] SHAP explainer initialized")
    except Exception as e:
        print(f"[WARNING] Could not initialize SHAP: {e}")
        shap_explainer = None

    # Initialize LIME explainer
    try:
        if training_data is not None and len(training_data) > 0:
            lime_explainer = LIMExplainer(model, training_data[:min(100, len(training_data))], feature_names)
        else:
            lime_explainer = LIMExplainer(model, np.random.randn(100, len(feature_names)), feature_names)
        print("[OK] LIME explainer initialized")
    except Exception as e:
        print(f"[WARNING] Could not initialize LIME: {e}")
        lime_explainer = None

    # Initialize comparator
    try:
        comparator = ExplanationComparator(shap_explainer, lime_explainer)
        print("[OK] Explanation comparator initialized")
    except Exception as e:
        print(f"[WARNING] Could not initialize comparator: {e}")
        comparator = None

    # Initialize business rules
    try:
        rules_engine = BusinessRulesEngine()
        print("[OK] Business rules engine initialized")
    except Exception as e:
        print(f"[ERROR] Could not initialize business rules: {e}")
        return False

    print("[OK] All components initialized successfully!")
    return True


def get_encoded_value(item_id, store_id):
    """Get encoded values for item and store IDs"""
    global reverse_item_mapping, reverse_store_mapping

    try:
        if item_id in reverse_item_mapping:
            item_encoded = reverse_item_mapping[item_id]
        else:
            item_encoded = abs(hash(item_id)) % 100

        if store_id in reverse_store_mapping:
            store_encoded = reverse_store_mapping[store_id]
        else:
            store_encoded = abs(hash(store_id)) % 10

        return int(item_encoded), int(store_encoded)
    except Exception as e:
        print(f"Error encoding: {e}")
        return abs(hash(item_id)) % 100, abs(hash(store_id)) % 10


def derive_all_features(sales, cost, inventory, has_promo, seasonality, elasticity,
                        competitor_price, current_price, category_encoded):
    """Derive all 26 features from simple inputs"""

    # Sales features
    sales_log = np.log1p(sales)
    sales_ma7 = sales
    sales_growth = 0.0
    sales_volatility = 0.0
    sales_normalized = 0.0
    sales_trend = 0.0

    # Cost features
    cost_ratio = cost / (current_price + 0.01) if current_price > 0 else 0.5
    cost_ratio = np.clip(cost_ratio, 0.05, 0.95)

    # Inventory features
    daily_sales = max(sales / 30, 1)
    inventory_ratio = inventory / (daily_sales + 1)
    inventory_ratio = np.clip(inventory_ratio, 0, 100)

    if inventory_ratio < 5:
        inventory_status = 0
    elif inventory_ratio < 20:
        inventory_status = 1
    else:
        inventory_status = 2

    # Promotion features
    promo_intensity = has_promo * 0.5
    promo_factor = 1 - (promo_intensity * 0.15)
    promo_frequency = has_promo * 0.3

    # Seasonality features
    seasonality_strength = abs(seasonality - 1.0)

    # Elasticity features
    elasticity_magnitude = abs(elasticity)
    elasticity_confidence = 0.7
    if elasticity_magnitude < 0.5:
        elasticity_category = 0
    elif elasticity_magnitude < 1.0:
        elasticity_category = 1
    else:
        elasticity_category = 2

    # Competitor features
    price_gap = competitor_price - current_price
    price_gap_ratio = price_gap / (current_price + 0.01) if current_price > 0 else 0
    price_gap_ratio = np.clip(price_gap_ratio, -0.5, 0.5)
    competitive_position = current_price / (competitor_price + 0.01) if competitor_price > 0 else 1.0
    competitive_position = np.clip(competitive_position, 0.5, 1.5)

    # Return all 26 features
    return [
        sales_log, sales_ma7, sales_growth, sales_volatility,
        sales_normalized, sales_trend,
        cost, cost_ratio,
        inventory, inventory_ratio, inventory_status,
        has_promo, promo_intensity, promo_factor, promo_frequency,
        seasonality, seasonality_strength,
        elasticity, elasticity_magnitude, elasticity_confidence, elasticity_category,
        competitor_price, price_gap, price_gap_ratio, competitive_position,
        category_encoded
    ]


@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html',
                           items=available_items,
                           stores=available_stores)


@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint for price prediction with explanation"""
    try:
        print("\n" + "=" * 60)
        print("PREDICTION REQUEST")
        print("=" * 60)

        data = request.json

        # Get Item and Store IDs
        item_id = data.get('item_id', 'FOODS_1_001')
        store_id = data.get('store_id', 'CA_1')

        # Get encoded values
        item_encoded, store_encoded = get_encoded_value(item_id, store_id)

        # Get user inputs
        sales = float(data.get('sales', 1000))
        cost = float(data.get('cost', 5.00))
        inventory = float(data.get('inventory', 5000))
        has_promo = int(data.get('has_promo', 0))
        seasonality = float(data.get('seasonality_index', 1.0))
        elasticity = float(data.get('elasticity', -1.2))
        competitor_price = float(data.get('competitor_price', 8.50))
        current_price = float(data.get('current_price', 7.00))

        # Derive all features
        category_encoded = item_encoded % 10
        features = derive_all_features(
            sales, cost, inventory, has_promo, seasonality, elasticity,
            competitor_price, current_price, category_encoded
        )

        features_array = np.array([features])

        # Apply scaling if available
        if scaler is not None:
            try:
                features_array = scaler.transform(features_array)
            except Exception as e:
                print(f"[WARNING] Scaling failed: {e}")

        # Predict
        predicted_price = float(model.predict(features_array)[0])

        # Apply business rules
        adjusted = rules_engine.apply_rules(
            predicted_price,
            current_price=current_price,
            inventory_level=inventory,
            is_holiday=data.get('is_holiday', False)
        )

        final_price = float(adjusted['adjusted_price'])

        # Generate SHAP explanation
        shap_exp = shap_explainer.explain_prediction(features_array, feature_names)
        shap_features = dict(list(shap_exp['features'].items())[:5])
        base_value = float(shap_exp['base_value'])

        # Generate LIME explanation
        lime_exp = lime_explainer.explain_prediction(features_array)
        lime_features = dict(list(lime_exp['features'].items())[:5])

        # Compare SHAP and LIME
        comparison_result = comparator.compare_explanations(features_array[0], feature_names)

        # Build display features
        display_features = {}
        for feat, details in shap_features.items():
            if feat not in HIDDEN_FEATURES:
                display_name = FEATURE_DISPLAY_NAMES.get(feat, feat.replace('_', ' ').title())
                display_features[display_name] = details

        # Create explanation
        top_list = list(display_features.items())[:4]
        parts = []
        for feature, details in top_list:
            impact = "+" if details.get('impact') == 'positive' else "-"
            val = details.get('value', 0)
            shap_val = abs(details.get('shap_value', 0))
            if isinstance(val, str):
                parts.append(f"  * {impact} {feature}: {val} ({impact} price by ${shap_val:.2f})")
            else:
                parts.append(f"  * {impact} {feature}: {val:.2f} ({impact} price by ${shap_val:.2f})")

        price_change_pct = ((final_price - current_price) / current_price) * 100 if current_price > 0 else 0

        explanation_text = f"""
PRICE RECOMMENDATION: ${final_price:.2f}

Item: {item_id} | Store: {store_id}
Current Price: ${current_price:.2f} | Change: {price_change_pct:+.1f}%

Why this price?

The recommended price is based on the following key factors:

{chr(10).join(parts)}

Business Insight:
The base price would have been ${base_value:.2f}. The adjustments above reflect current market conditions, demand patterns, and competitive landscape.

Recommendation:
This price aligns with market dynamics and maximizes expected revenue.

Actionable Next Steps:
- Monitor competitor prices for elastic items
- Consider promotional bundling if inventory is high
- Review pricing strategy for items with high seasonality
        """

        # Build LIME display features
        lime_display_features = {}
        for feat, details in lime_features.items():
            if feat not in HIDDEN_FEATURES:
                display_name = FEATURE_DISPLAY_NAMES.get(feat, feat.replace('_', ' ').title())
                lime_display_features[display_name] = details

        # ====== NO CONFIDENCE MODULE - REMOVED COMPLETELY ======
        # Just return the prediction and explanations without confidence scores

        response = {
            'success': True,
            'item_id': item_id,
            'store_id': store_id,
            'predicted_price': round(final_price, 2),
            'original_prediction': round(float(adjusted['original_price']), 2),
            'current_price': round(current_price, 2),
            'price_change_pct': round(price_change_pct, 1),
            'price_change_abs': round(final_price - current_price, 2),
            'explanation': explanation_text.strip(),
            'shap_factors': display_features,
            'lime_factors': lime_display_features,
            'business_rules_applied': adjusted['rules_applied'],
            'base_value': round(base_value, 2),
            'item_encoded': int(item_encoded),
            'store_encoded': int(store_encoded),
            'comparison': {
                'agreement_score': round(comparison_result['agreement_score'] * 100, 1),
                'correlation': round(comparison_result['correlation'], 3),
                'consistent': comparison_result['consistent']
            }
        }

        print(f"✅ Success! Returning response")
        return jsonify(response)

    except Exception as e:
        print("❌ ERROR:")
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)})


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy' if model is not None else 'unhealthy',
        'model_loaded': model is not None,
        'features_loaded': feature_names is not None,
        'items_available': len(available_items),
        'stores_available': len(available_stores)
    })


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("AI-Powered Price Explanations System")
    print("=" * 60 + "\n")

    if load_models():
        print("\n" + "=" * 60)
        print("✅ SYSTEM READY!")
        print("=" * 60)
        print(f"📦 Items available: {len(available_items)}")
        print(f"🏪 Stores available: {len(available_stores)}")
        print(f"🌐 Open: http://localhost:5000")
        print("=" * 60 + "\n")
        app.run(debug=True, port=5000, threaded=True)
    else:
        print("\n" + "=" * 60)
        print("❌ SYSTEM NOT READY")
        print("=" * 60)
        print("\nPlease run the following steps:")
        print("1. python run_experiments.py")
        print("2. python app.py")
        print("=" * 60 + "\n")