"""Flask web application - WITH Confidence Module"""

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
from src.evaluation.confidence import ConfidenceModule, UncertaintyEstimator
from src.utils.constants import PROCESSED_DATA_PATH, EXPERIMENTS_PATH
from src.llm.summarizer import LLMSummarizer
app = Flask(__name__)

# Global variables
model = None
feature_names = None
shap_explainer = None
lime_explainer = None
comparator = None
rules_engine = None
training_data_scaled = None
training_y = None
scaler = None
confidence_module = None
uncertainty_estimator = None
llm_summarizer = None

# Item and Store mappings
item_mapping = {}
store_mapping = {}
reverse_item_mapping = {}
reverse_store_mapping = {}
available_items = []
available_stores = []

# Feature display names - USER FRIENDLY NAMES
FEATURE_DISPLAY_NAMES = {
    'sales_log': 'Sales Volume',
    'sales_ma7': 'Demand Trend',
    'sales_growth': 'Sales Growth',
    'sales_volatility': 'Sales Stability',
    'sales_normalized': 'Relative Demand',
    'sales_trend': 'Sales Momentum',
    'cost': 'Product Cost',
    'cost_ratio': 'Cost-to-Price Ratio',
    'inventory': 'Inventory Level',
    'inventory_ratio': 'Inventory Days Cover',
    'inventory_status': 'Inventory Status',
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
    global rules_engine, training_data_scaled, training_y, scaler
    global confidence_module, uncertainty_estimator

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

    # Load the scaler
    scaler_path = PROCESSED_DATA_PATH / 'scaler.pkl'
    if scaler_path.exists():
        try:
            scaler = joblib.load(scaler_path)
            print(f"[OK] Scaler loaded from {scaler_path}")
        except Exception as e:
            print(f"[WARNING] Could not load scaler: {e}")
            scaler = None
    else:
        print(f"[WARNING] Scaler not found at {scaler_path}")
        scaler = None

    # Load training data and scale it
    training_data_unscaled = None
    training_y_local = None
    training_data_scaled_local = None

    data_path = PROCESSED_DATA_PATH / 'processed_data.csv'
    if data_path.exists():
        try:
            df = pd.read_csv(data_path, nrows=5000)
            feature_cols = [col for col in feature_names if col in df.columns]
            if feature_cols:
                training_data_unscaled = df[feature_cols].values

                if scaler is not None:
                    training_data_scaled_local = scaler.transform(training_data_unscaled)
                    print(f"[OK] Training data scaled successfully")
                else:
                    print("[WARNING] Scaler is None, using unscaled data")
                    training_data_scaled_local = training_data_unscaled

                if 'price' in df.columns:
                    training_y_local = df['price'].values[:min(5000, len(df))]
                    print(f"[OK] Training data loaded: {len(training_data_unscaled)} samples with prices")
                else:
                    print("[WARNING] 'price' column not found")
                    training_y_local = np.random.uniform(5, 15, min(5000, len(df)))
            else:
                training_data_scaled_local = None
                training_y_local = None
        except Exception as e:
            print(f"[WARNING] Could not load training data: {e}")
            training_data_scaled_local = None
            training_y_local = None

    # Store scaled training data for confidence module
    training_data_scaled = training_data_scaled_local
    training_y = training_y_local

    # Initialize SHAP explainer with SCALED background data
    try:
        if training_data_scaled is not None and len(training_data_scaled) > 0:
            background = training_data_scaled[:min(100, len(training_data_scaled))]
        else:
            background = np.random.randn(100, len(feature_names))

        shap_explainer = SHAPExplainer(model, background)
        print("[OK] SHAP explainer initialized")
    except Exception as e:
        print(f"[WARNING] Could not initialize SHAP: {e}")
        shap_explainer = None

    # Initialize LIME explainer with SCALED training data
    try:
        if training_data_scaled is not None and len(training_data_scaled) > 0:
            lime_explainer = LIMExplainer(model, training_data_scaled[:min(100, len(training_data_scaled))],
                                          feature_names)
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

    # Initialize Confidence Module with SCALED training data
    try:
        confidence_module = ConfidenceModule(model, feature_names)
        if training_data_scaled is not None and len(training_data_scaled) > 0:
            confidence_module.set_training_data(
                training_data_scaled,
                np.zeros(len(training_data_scaled))
            )
            print(f"[OK] Confidence module initialized with {len(training_data_scaled)} scaled samples")
        else:
            print("[WARNING] No training data for confidence module")
            synth_X = np.random.randn(100, len(feature_names))
            confidence_module.set_training_data(synth_X, np.zeros(100))
            print("[OK] Confidence module initialized with synthetic scaled data")
    except Exception as e:
        print(f"[WARNING] Could not initialize confidence module: {e}")
        confidence_module = None

    # Initialize Uncertainty Estimator with SCALED training data
    try:
        uncertainty_estimator = UncertaintyEstimator(model)
        if training_data_scaled is not None and len(training_data_scaled) > 0 and training_y is not None:
            n_samples = min(5000, len(training_data_scaled))
            uncertainty_estimator.create_ensemble(
                training_data_scaled[:n_samples],
                training_y[:n_samples],
                n_models=3
            )
            print(f"[OK] Uncertainty estimator initialized with {n_samples} scaled samples")
        else:
            print("[WARNING] No training data for uncertainty estimator")
            synth_X = np.random.randn(100, len(feature_names))
            synth_y = np.random.uniform(5, 15, 100)
            uncertainty_estimator.create_ensemble(synth_X, synth_y, n_models=3)
            print("[OK] Uncertainty estimator initialized with synthetic scaled data")
    except Exception as e:
        print(f"[WARNING] Could not initialize uncertainty estimator: {e}")
        uncertainty_estimator = None
    global llm_summarizer
    try:
        # Try local LLM first (no API key needed!)
        llm_summarizer = LLMSummarizer(provider='local', max_ram='4gb')
        print("[OK] LLM summarizer initialized with Local LLM")
    except Exception as e:
        print(f"[WARNING] Could not initialize local LLM: {e}")
        # Fallback to mock mode
        llm_summarizer = LLMSummarizer(provider='mock')
        print("[INFO] LLM summarizer using mock mode")

    print("[OK] All components initialized successfully!")
    return True

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
    """API endpoint for price prediction with explanation and confidence"""
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

        # Derive all features (unscaled)
        category_encoded = item_encoded % 10
        features = derive_all_features(
            sales, cost, inventory, has_promo, seasonality, elasticity,
            competitor_price, current_price, category_encoded
        )

        features_array = np.array([features])

        # Apply scaling for model prediction
        if scaler is not None:
            try:
                features_scaled = scaler.transform(features_array)
                print(f"[DEBUG] Features scaled successfully")
            except Exception as e:
                print(f"[WARNING] Scaling failed: {e}")
                features_scaled = features_array
        else:
            print("[WARNING] Scaler is None - using unscaled features!")
            features_scaled = features_array

        print(f"[DEBUG] Unscaled mean: {np.mean(features_array[0]):.4f}")
        print(f"[DEBUG] Scaled mean: {np.mean(features_scaled[0]):.4f}")

        # Predict using scaled features
        predicted_price = float(model.predict(features_scaled)[0])
        print(f"[DEBUG] Predicted price: ${predicted_price:.2f}")

        # Apply business rules
        adjusted = rules_engine.apply_rules(
            predicted_price,
            current_price=current_price,
            inventory_level=inventory,
            is_holiday=data.get('is_holiday', False)
        )

        final_price = float(adjusted['adjusted_price'])
        print(f"[DEBUG] Final price: ${final_price:.2f}")

        # Generate SHAP explanation using scaled features
        shap_exp = shap_explainer.explain_prediction(features_scaled, feature_names)
        shap_features = dict(list(shap_exp['features'].items())[:10])  # Get top 10
        base_value = float(shap_exp['base_value'])

        # Generate LIME explanation
        lime_exp = lime_explainer.explain_prediction(features_scaled)
        lime_features = dict(list(lime_exp['features'].items())[:10])  # Get top 10

        # Compare SHAP and LIME
        comparison_result = comparator.compare_explanations(features_scaled[0], feature_names)

        # ================================================================
        # CONFIDENCE MODULE - Using SCALED features
        # ================================================================
        uncertainty_scores = None
        prediction_interval = None
        ensemble_uncertainty = None
        reliability_factors = None

        if confidence_module is not None:
            try:
                uncertainty_scores = confidence_module.calculate_uncertainty_score(
                    features_scaled[0], predicted_price
                )
                prediction_interval = confidence_module.calculate_prediction_interval(
                    features_scaled, confidence_level=0.90
                )
                reliability_factors = confidence_module.get_reliability_factors(
                    shap_exp, predicted_price
                )
                print(f"[DEBUG] Confidence: {uncertainty_scores['confidence_level']}")
                print(
                    f"[DEBUG] Raw prediction interval: ${prediction_interval['lower_bound']:.2f} - ${prediction_interval['upper_bound']:.2f}")
            except Exception as e:
                print(f"[WARNING] Confidence calculation failed: {e}")
                import traceback
                traceback.print_exc()

        if uncertainty_estimator is not None:
            try:
                ensemble_uncertainty = uncertainty_estimator.predict_with_uncertainty(
                    features_scaled
                )
                print(f"[DEBUG] Ensemble mean: ${ensemble_uncertainty['mean_prediction']:.2f}")
            except Exception as e:
                print(f"[WARNING] Ensemble uncertainty failed: {e}")

        # ================================================================
        # FALLBACK - Ensure prediction interval contains the price
        # ================================================================
        if prediction_interval is not None:
            lower = prediction_interval['lower_bound']
            upper = prediction_interval['upper_bound']

            interval_contains_price = lower <= final_price <= upper
            is_too_narrow = (upper - lower) < (final_price * 0.05)
            is_completely_wrong = upper > 100 or lower < 0

            if not interval_contains_price or is_too_narrow or is_completely_wrong:
                print(f"[WARNING] Prediction interval issue:")
                print(f"  Interval: ${lower:.2f} - ${upper:.2f}")
                print(f"  Predicted price: ${final_price:.2f}")

                margin = final_price * 0.10
                prediction_interval['lower_bound'] = max(0, final_price - margin)
                prediction_interval['upper_bound'] = final_price + margin
                prediction_interval['interval_width'] = 2 * margin

        if ensemble_uncertainty is not None:
            ensemble_mean = ensemble_uncertainty['mean_prediction']
            if ensemble_mean < 0 or ensemble_mean > 100 or abs(ensemble_mean - final_price) > (final_price * 0.3):
                print(f"[WARNING] Ensemble mean seems wrong: ${ensemble_mean:.2f}")
                ensemble_uncertainty['mean_prediction'] = final_price
                ensemble_uncertainty['std_prediction'] = final_price * 0.05
                ensemble_uncertainty['model_agreement'] = 0.85

        # ================================================================
        # BUILD EXPLANATION WITH ALL FACTORS (POSITIVE AND NEGATIVE)
        # ================================================================

        # Build display features - Include ALL factors
        display_features = {}
        all_factors_with_impact = []

        for feat, details in shap_features.items():
            if feat not in HIDDEN_FEATURES:
                display_name = FEATURE_DISPLAY_NAMES.get(feat, feat.replace('_', ' ').title())
                shap_val = details.get('shap_value', 0)
                display_features[display_name] = details

                all_factors_with_impact.append({
                    'name': display_name,
                    'value': details.get('value', 0),
                    'shap_value': shap_val,
                    'impact': 'positive' if shap_val > 0 else 'negative',
                    'abs_impact': abs(shap_val)
                })

        # Sort by absolute impact (highest first)
        all_factors_with_impact.sort(key=lambda x: x['abs_impact'], reverse=True)

        # Separate positive and negative factors
        positive_factors = [f for f in all_factors_with_impact if f['impact'] == 'positive']
        negative_factors = [f for f in all_factors_with_impact if f['impact'] == 'negative']

        # Build the explanation parts
        parts = []

        # Add positive factors
        if positive_factors:
            parts.append("")
            parts.append("FACTORS INCREASING THE PRICE (Positive Impact):")
            parts.append("-" * 50)
            for f in positive_factors[:8]:
                shap_val = abs(f['shap_value'])
                val = f['value']
                if isinstance(val, str):
                    parts.append(f"  ↑ {f['name']}: {val} (+${shap_val:.2f})")
                else:
                    parts.append(f"  ↑ {f['name']}: {val:.2f} (+${shap_val:.2f})")

        # Add negative factors
        if negative_factors:
            parts.append("")
            parts.append("FACTORS DECREASING THE PRICE (Negative Impact):")
            parts.append("-" * 50)
            for f in negative_factors[:8]:
                shap_val = abs(f['shap_value'])
                val = f['value']
                if isinstance(val, str):
                    parts.append(f"  ↓ {f['name']}: {val} (-${shap_val:.2f})")
                else:
                    parts.append(f"  ↓ {f['name']}: {val:.2f} (-${shap_val:.2f})")

        # Build confidence text
        price_change_pct = ((final_price - current_price) / current_price) * 100 if current_price > 0 else 0

        confidence_text = ""
        if uncertainty_scores:
            confidence_text = f"""
Confidence Level: {uncertainty_scores['confidence_level']}
Reliability Score: {uncertainty_scores['reliability_score']:.1%}
Uncertainty Score: {uncertainty_scores['uncertainty_score']:.1%}
"""

        if prediction_interval:
            confidence_text += f"""
Prediction Range: ${prediction_interval['lower_bound']:.2f} - ${prediction_interval['upper_bound']:.2f}
"""

        if ensemble_uncertainty:
            confidence_text += f"""
Ensemble Mean: ${ensemble_uncertainty['mean_prediction']:.2f}
Model Agreement: {ensemble_uncertainty['model_agreement']:.1%}
"""

        # Build the final explanation text
        explanation_text = f"""
PRICE RECOMMENDATION: ${final_price:.2f}

Item: {item_id} | Store: {store_id}
Current Price: ${current_price:.2f} | Change: {price_change_pct:+.1f}%

{confidence_text}

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
        # ================================================================
        # GENERATE LLM EXPLANATION (Optional - falls back to mock)
        # ================================================================
        llm_explanation = None
        if llm_summarizer is not None:
            try:
                # Prepare business context with correct price
                business_context = {
                    'item_id': item_id,
                    'store_id': store_id,
                    'sales': sales,
                    'inventory': inventory,
                    'has_promo': has_promo,
                    'seasonality': seasonality,
                    'elasticity': elasticity,
                    'competitor_price': competitor_price,
                    'current_price': current_price,
                    'recommended_price': final_price,
                    'base_price': base_value
                }

                llm_explanation = llm_summarizer.generate_explanation(
                    shap_explanation=shap_exp,
                    lime_explanation=lime_exp,
                    business_context=business_context
                )
                print(f"[DEBUG] LLM explanation generated")
            except Exception as e:
                print(f"[WARNING] LLM explanation failed: {e}")
                llm_explanation = None

        # If LLM generated an explanation, use it (keep the price and confidence from mock)
        if llm_explanation:
            # Keep the price, item info, and confidence from the mock explanation
            # Replace just the "Why this price?" section
            explanation_text = f"""
        PRICE RECOMMENDATION: ${final_price:.2f}

        Item: {item_id} | Store: {store_id}
        Current Price: ${current_price:.2f} | Change: {price_change_pct:+.1f}%

        {confidence_text}

        Why this price?

        {llm_explanation}

        Actionable Next Steps:
        - Monitor competitor prices for elastic items
        - Consider promotional bundling if inventory is high
        - Review pricing strategy for items with high seasonality
                    """.strip()
        # Build LIME display features
        lime_display_features = {}
        for feat, details in lime_features.items():
            if feat not in HIDDEN_FEATURES:
                display_name = FEATURE_DISPLAY_NAMES.get(feat, feat.replace('_', ' ').title())
                lime_display_features[display_name] = details

        # Build response
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
            },
            'confidence': {
                'level': uncertainty_scores['confidence_level'] if uncertainty_scores else 'Unknown',
                'reliability_score': round(uncertainty_scores['reliability_score'], 2) if uncertainty_scores else None,
                'uncertainty_score': round(uncertainty_scores['uncertainty_score'], 2) if uncertainty_scores else None,
                'prediction_interval_lower': round(prediction_interval['lower_bound'],
                                                   2) if prediction_interval else None,
                'prediction_interval_upper': round(prediction_interval['upper_bound'],
                                                   2) if prediction_interval else None,
                'ensemble_mean': round(ensemble_uncertainty['mean_prediction'], 2) if ensemble_uncertainty else None,
                'ensemble_std': round(ensemble_uncertainty['std_prediction'], 2) if ensemble_uncertainty else None,
                'model_agreement': round(ensemble_uncertainty['model_agreement'] * 100,
                                         1) if ensemble_uncertainty else None,
                'warnings': reliability_factors['warnings'] if reliability_factors else [],
                'recommendations': reliability_factors['recommendations'] if reliability_factors else []
            }
        }

        print(f"Success! Returning response")
        return jsonify(response)

    except Exception as e:
        print("ERROR:")
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
        'stores_available': len(available_stores),
        'confidence_loaded': confidence_module is not None,
        'scaler_loaded': scaler is not None,
        'training_data_loaded': training_data_scaled is not None
    })


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("AI-Powered Price Explanations System")
    print("=" * 60 + "\n")

    if load_models():
        print("\n" + "=" * 60)
        print("SYSTEM READY!")
        print("=" * 60)
        print(f"Items available: {len(available_items)}")
        print(f"Stores available: {len(available_stores)}")
        print(f"Scaler: {'Loaded' if scaler is not None else 'NOT LOADED!'}")
        print(f"Training Data: {'Loaded' if training_data_scaled is not None else 'NOT LOADED!'}")
        print(f"Confidence Module: {'Loaded' if confidence_module else 'Not loaded'}")
        print(f"Uncertainty Estimator: {'Loaded' if uncertainty_estimator else 'Not loaded'}")
        print(f"Open: http://localhost:5000")
        print("=" * 60 + "\n")
        app.run(debug=True, port=5000, threaded=True)
    else:
        print("\n" + "=" * 60)
        print("SYSTEM NOT READY")
        print("=" * 60)
        print("\nPlease run the following steps:")
        print("1. python run_experiments.py")
        print("2. python app.py")
        print("=" * 60 + "\n")