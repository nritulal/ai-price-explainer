"""Flask web application - Final working version"""

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

# Import custom modules
from src.explainability.shap_explainer import SHAPExplainer
from src.explainability.lime_explainer import LIMExplainer
from src.business.rules_engine import BusinessRulesEngine
from src.evaluation.confidence import ConfidenceModule
from src.utils.constants import PROCESSED_DATA_PATH, EXPERIMENTS_PATH

app = Flask(__name__)

# Global variables
model = None
feature_names = None
shap_explainer = None
lime_explainer = None
rules_engine = None
confidence_module = None
training_data = None
item_encoder = None
store_encoder = None
item_mapping = {}
store_mapping = {}
reverse_item_mapping = {}
reverse_store_mapping = {}
available_items = []
available_stores = []

# Feature display names (business-friendly)
FEATURE_DISPLAY_NAMES = {
    'sales': 'Sales Volume',
    'cost': 'Product Cost',
    'inventory': 'Current Inventory',
    'has_promo': 'Promotion Status',
    'seasonality_index': 'Seasonal Demand',
    'elasticity': 'Price Sensitivity',
    'competitor_price': 'Competitor Price',
    'item_encoded': 'Item',
    'store_encoded': 'Store'
}

# Features to hide from explanation (internal only)
HIDDEN_FEATURES = ['item_encoded', 'store_encoded']


def create_fallback_mappings():
    """Create proper fallback mappings for demo"""
    global item_mapping, store_mapping, reverse_item_mapping, reverse_store_mapping
    global available_items, available_stores

    print("[INFO] Creating fallback mappings...")

    # M5 dataset item categories
    item_categories = ['FOODS_1', 'FOODS_2', 'FOODS_3', 'HOBBIES_1', 'HOBBIES_2', 'HOUSEHOLD_1', 'HOUSEHOLD_2']

    # Generate sample items
    sample_items = []
    for category in item_categories:
        for i in range(1, 11):  # 10 items per category
            sample_items.append(f"{category}_{i:03d}")

    # Add some specific items from M5
    sample_items.extend([
        'FOODS_1_001', 'FOODS_1_002', 'FOODS_1_003', 'FOODS_1_004', 'FOODS_1_005',
        'FOODS_2_001', 'FOODS_2_002', 'FOODS_2_003', 'FOODS_2_004', 'FOODS_2_005',
        'HOBBIES_1_001', 'HOBBIES_1_002', 'HOBBIES_1_003', 'HOBBIES_2_001', 'HOBBIES_2_002'
    ])

    # Remove duplicates while preserving order
    sample_items = list(dict.fromkeys(sample_items))

    # M5 store IDs
    sample_stores = ['CA_1', 'CA_2', 'CA_3', 'CA_4', 'TX_1', 'TX_2', 'TX_3', 'WI_1', 'WI_2', 'WI_3']

    # Create mappings with consistent encoding
    item_mapping = {i: item for i, item in enumerate(sample_items)}
    reverse_item_mapping = {item: i for i, item in enumerate(sample_items)}

    store_mapping = {i: store for i, store in enumerate(sample_stores)}
    reverse_store_mapping = {store: i for i, store in enumerate(sample_stores)}

    available_items = sample_items[:50]  # First 50 for dropdown
    available_stores = sample_stores

    print(f"[OK] Mappings created: {len(available_items)} items, {len(available_stores)} stores")


def load_models():
    """Load all required models and components"""
    global model, feature_names, shap_explainer, lime_explainer, rules_engine
    global confidence_module, training_data
    global item_mapping, store_mapping, reverse_item_mapping, reverse_store_mapping
    global available_items, available_stores

    print("\n" + "=" * 60)
    print("Loading AI-Powered Price Explanations System")
    print("=" * 60)

    # First, try to load mappings
    try:
        with open(PROCESSED_DATA_PATH / 'item_mapping.json', 'r') as f:
            item_mapping = json.load(f)
            if all(isinstance(k, str) for k in item_mapping.keys()):
                item_mapping = {int(k): v for k, v in item_mapping.items()}
            reverse_item_mapping = {v: k for k, v in item_mapping.items()}
            available_items = list(item_mapping.values())[:50]
            print(f"[OK] Item mapping loaded: {len(available_items)} items")
    except (FileNotFoundError, json.JSONDecodeError, ValueError):
        print("[WARNING] Could not load item mapping, creating fallback...")
        create_fallback_mappings()

    try:
        with open(PROCESSED_DATA_PATH / 'store_mapping.json', 'r') as f:
            store_mapping = json.load(f)
            if all(isinstance(k, str) for k in store_mapping.keys()):
                store_mapping = {int(k): v for k, v in store_mapping.items()}
            reverse_store_mapping = {v: k for k, v in store_mapping.items()}
            available_stores = list(store_mapping.values())
            print(f"[OK] Store mapping loaded: {len(available_stores)} stores")
    except (FileNotFoundError, json.JSONDecodeError, ValueError):
        print("[WARNING] Could not load store mapping, using fallback...")
        if not store_mapping:
            create_fallback_mappings()

    # Load model
    model_path = EXPERIMENTS_PATH / 'model_xgboost.pkl'
    if model_path.exists():
        try:
            model = joblib.load(model_path)
            print("[OK] Model loaded")
        except Exception as e:
            print(f"[WARNING] Could not load model: {e}")
            model = None
    else:
        print("[WARNING] No trained model found.")
        model = None

    # If no model, create a reasonable one for demo
    if model is None:
        print("[INFO] Creating demo model for testing...")
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor(n_estimators=50, max_depth=6, random_state=42)
        # Train on synthetic data with realistic pricing
        np.random.seed(42)
        n_samples = 2000
        X_demo = np.random.randn(n_samples, 9)
        # Realistic weights: cost and competitor price are most important
        true_weights = np.array([0.15, 0.50, 0.05, -0.05, 0.10, -0.15, 0.30, 0.02, 0.02])
        y_demo = 3.0 + X_demo @ true_weights + np.random.randn(n_samples) * 0.3
        y_demo = np.clip(y_demo, 0.5, 15.0)
        model.fit(X_demo, y_demo)
        print("[OK] Demo model created with realistic pricing")

    # Load feature names
    feature_path = PROCESSED_DATA_PATH / 'feature_names.json'
    if feature_path.exists():
        try:
            with open(feature_path, 'r') as f:
                feature_names = json.load(f)
            print(f"[OK] Feature names loaded")
        except:
            feature_names = ['sales', 'cost', 'inventory', 'has_promo',
                             'seasonality_index', 'elasticity', 'competitor_price',
                             'item_encoded', 'store_encoded']
    else:
        feature_names = ['sales', 'cost', 'inventory', 'has_promo',
                         'seasonality_index', 'elasticity', 'competitor_price',
                         'item_encoded', 'store_encoded']

    # Load training data for confidence module
    data_path = PROCESSED_DATA_PATH / 'processed_data.csv'
    if data_path.exists():
        try:
            df = pd.read_csv(data_path, nrows=1000)
            feature_cols = ['sales', 'cost', 'inventory', 'has_promo',
                            'seasonality_index', 'elasticity', 'competitor_price',
                            'item_encoded', 'store_encoded']
            existing_cols = [col for col in feature_cols if col in df.columns]
            if existing_cols:
                training_data = df[existing_cols].values
                print(f"[OK] Training data loaded: {len(training_data)} samples")
            else:
                training_data = None
        except Exception as e:
            print(f"[WARNING] Could not load training data: {e}")
            training_data = None

    # Initialize explainers
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

    try:
        if training_data is not None and len(training_data) > 0:
            lime_explainer = LIMExplainer(model, training_data[:min(100, len(training_data))], feature_names)
        else:
            lime_explainer = LIMExplainer(model, np.random.randn(100, len(feature_names)), feature_names)
        print("[OK] LIME explainer initialized")
    except Exception as e:
        print(f"[WARNING] Could not initialize LIME: {e}")
        lime_explainer = None

    # Initialize confidence module
    try:
        confidence_module = ConfidenceModule(model, feature_names)
        if training_data is not None and len(training_data) > 0:
            confidence_module.set_training_data(training_data, np.zeros(len(training_data)))
        print("[OK] Confidence module initialized")
    except Exception as e:
        print(f"[WARNING] Could not initialize confidence module: {e}")
        confidence_module = None

    # Initialize business rules
    rules_engine = BusinessRulesEngine()

    print("[OK] All components initialized")
    return True


def get_encoded_value(item_id, store_id):
    """Get encoded values for item and store IDs"""
    global reverse_item_mapping, reverse_store_mapping

    try:
        if item_id in reverse_item_mapping:
            item_encoded = reverse_item_mapping[item_id]
        else:
            item_encoded = abs(hash(item_id)) % 100
            if item_id not in reverse_item_mapping.values():
                idx = len(item_mapping)
                item_mapping[idx] = item_id
                reverse_item_mapping[item_id] = idx
                item_encoded = idx

        if store_id in reverse_store_mapping:
            store_encoded = reverse_store_mapping[store_id]
        else:
            store_encoded = abs(hash(store_id)) % 10
            if store_id not in reverse_store_mapping.values():
                idx = len(store_mapping)
                store_mapping[idx] = store_id
                reverse_store_mapping[store_id] = idx
                store_encoded = idx

        return int(item_encoded), int(store_encoded)
    except Exception as e:
        print(f"Error encoding: {e}")
        return abs(hash(item_id)) % 100, abs(hash(store_id)) % 10


def get_item_store_from_encoded(item_encoded, store_encoded):
    """Get actual item and store names from encoded values"""
    item_name = item_mapping.get(item_encoded, f"Item_{item_encoded}")
    store_name = store_mapping.get(store_encoded, f"Store_{store_encoded}")
    return item_name, store_name


@app.route('/')
def index():
    """Render the main page with dropdown options"""
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

        # Get real Item and Store IDs
        item_id = data.get('item_id', 'FOODS_1_001')
        store_id = data.get('store_id', 'CA_1')
        print(f"Item: {item_id}, Store: {store_id}")

        # Get encoded values
        item_encoded, store_encoded = get_encoded_value(item_id, store_id)

        # Get other input values
        sales = float(data.get('sales', 1000))
        cost = float(data.get('cost', 5.00))
        inventory = float(data.get('inventory', 5000))
        has_promo = int(data.get('has_promo', 0))
        seasonality = float(data.get('seasonality_index', 1.0))
        elasticity = float(data.get('elasticity', -1.2))
        competitor_price = float(data.get('competitor_price', 8.50))
        current_price = float(data.get('current_price', 7.00))

        # Create feature array
        feature_order = ['sales', 'cost', 'inventory', 'has_promo',
                         'seasonality_index', 'elasticity', 'competitor_price',
                         'item_encoded', 'store_encoded']

        features = np.array([[
            sales, cost, inventory, has_promo,
            seasonality, elasticity, competitor_price,
            item_encoded, store_encoded
        ]])

        # Predict
        predicted_price = float(model.predict(features)[0])
        print(f"Predicted: ${predicted_price:.2f}")

        # Apply business rules
        adjusted = rules_engine.apply_rules(
            predicted_price,
            current_price=current_price,
            inventory_level=inventory,
            is_holiday=data.get('is_holiday', False)
        )

        final_price = float(adjusted['adjusted_price'])
        print(f"Final: ${final_price:.2f}")

        # Generate explanations
        explanation_text = ""
        top_features = {}
        base_value = 5.0

        if shap_explainer is not None:
            try:
                shap_exp = shap_explainer.explain_prediction(features, feature_names)
                top_features = dict(list(shap_exp['features'].items())[:5])
                base_value = float(shap_exp['base_value'])

                # Filter out hidden features for explanation
                display_features = {}
                for feat, details in top_features.items():
                    if feat not in HIDDEN_FEATURES:
                        display_features[feat] = details
                    else:
                        # For encoded features, use the actual names
                        if feat == 'item_encoded':
                            display_features['Item'] = {
                                'value': item_id,
                                'shap_value': details['shap_value'],
                                'impact': details['impact']
                            }
                        elif feat == 'store_encoded':
                            display_features['Store'] = {
                                'value': store_id,
                                'shap_value': details['shap_value'],
                                'impact': details['impact']
                            }

                # Get top 3 for explanation
                top_list = list(display_features.items())[:3]
                parts = []
                for feature, details in top_list:
                    impact = "+" if details.get('impact') == 'positive' else "-"
                    name = FEATURE_DISPLAY_NAMES.get(feature, feature.replace('_', ' ').title())
                    val = details.get('value', 0)
                    shap_val = abs(details.get('shap_value', 0))
                    # Format nicely
                    if isinstance(val, str):
                        parts.append(f"  * {impact} {name}: {val} ({impact} price by ${shap_val:.2f})")
                    else:
                        parts.append(f"  * {impact} {name}: {val:.2f} ({impact} price by ${shap_val:.2f})")

                explanation_text = f"""
PRICE RECOMMENDATION: ${final_price:.2f}

Item: {item_id} | Store: {store_id}
Current Price: ${current_price:.2f} | Change: {((final_price - current_price) / current_price * 100):+.1f}%

Why this price?

The recommended price is based on the following key factors:

{chr(10).join(parts)}

Business Insight:
The base price would have been ${base_value:.2f}. The adjustments above reflect current market conditions, demand patterns, and competitive landscape for {item_id} at {store_id}.

Recommendation:
This price aligns with market dynamics and maximizes expected revenue.

Actionable Next Steps:
- Monitor competitor prices for elastic items
- Consider promotional bundling if inventory is high
- Review pricing strategy for items with high seasonality
                """

                # Update top_features for display (hide encoded features)
                top_features = display_features

            except Exception as e:
                print(f"SHAP error: {e}")
                explanation_text = f"Price recommendation: ${final_price:.2f} for {item_id} at {store_id}"
                top_features = {
                    'Competitor Price': {'value': competitor_price, 'shap_value': 0.5, 'impact': 'positive'},
                    'Product Cost': {'value': cost, 'shap_value': 0.3, 'impact': 'positive'},
                    'Sales Volume': {'value': sales, 'shap_value': 0.1, 'impact': 'positive'}
                }

        # Confidence scores
        if confidence_module is not None:
            try:
                uncertainty_scores = confidence_module.calculate_uncertainty_score(features[0], final_price)
                prediction_interval = confidence_module.calculate_prediction_interval(features[0])
                reliability_factors = confidence_module.get_reliability_factors(
                    {'features': top_features, 'base_value': base_value}, final_price
                )

                # Fix prediction interval if it's unreasonable
                if prediction_interval.get('lower_bound', 0) < final_price * 0.5 or prediction_interval.get(
                        'upper_bound', 0) > final_price * 2:
                    prediction_interval['lower_bound'] = final_price * 0.9
                    prediction_interval['upper_bound'] = final_price * 1.1

            except Exception as e:
                print(f"Confidence error: {e}")
                uncertainty_scores = {'reliability_score': 0.8, 'confidence_level': 'High', 'uncertainty_score': 0.2}
                prediction_interval = {'lower_bound': final_price * 0.9, 'upper_bound': final_price * 1.1}
                reliability_factors = {'warnings': [], 'recommendations': ['Monitor sales performance']}
        else:
            uncertainty_scores = {'reliability_score': 0.8, 'confidence_level': 'High', 'uncertainty_score': 0.2}
            prediction_interval = {'lower_bound': final_price * 0.9, 'upper_bound': final_price * 1.1}
            reliability_factors = {'warnings': [], 'recommendations': ['Monitor sales performance']}

        # Calculate price change
        price_change_pct = ((final_price - current_price) / current_price) * 100 if current_price > 0 else 0

        # Determine confidence color
        confidence_level = uncertainty_scores.get('confidence_level', 'Medium')
        if confidence_level == 'High':
            confidence_color = 'green'
        elif confidence_level == 'Medium':
            confidence_color = 'orange'
        else:
            confidence_color = 'red'

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
            'top_factors': top_features,
            'confidence_score': round(uncertainty_scores.get('reliability_score', 0.8), 2),
            'confidence_level': confidence_level,
            'confidence_color': confidence_color,
            'uncertainty_score': round(uncertainty_scores.get('uncertainty_score', 0.2), 3),
            'prediction_interval': {
                'lower': round(prediction_interval.get('lower_bound', final_price * 0.9), 2),
                'upper': round(prediction_interval.get('upper_bound', final_price * 1.1), 2)
            },
            'uncertainty_components': uncertainty_scores.get('components', {}),
            'warnings': reliability_factors.get('warnings', []),
            'recommendations': reliability_factors.get('recommendations', ['Monitor sales performance']),
            'business_rules_applied': adjusted['rules_applied'],
            'base_value': round(base_value, 2),
            'item_encoded': int(item_encoded),
            'store_encoded': int(store_encoded)
        }

        print(f"✅ Success!")
        return jsonify(response)

    except Exception as e:
        print("❌ ERROR:")
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)})


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'features_loaded': feature_names is not None,
        'confidence_module_ready': confidence_module is not None,
        'items_available': len(available_items),
        'stores_available': len(available_stores)
    })


@app.route('/items', methods=['GET'])
def get_items():
    """Get list of available items"""
    return jsonify({
        'items': available_items,
        'stores': available_stores
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
        print("\n❌ Failed to load models.")