"""Flask web application"""

from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import json
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

# Import custom modules
from src.explainability.shap_explainer import SHAPExplainer
from src.explainability.lime_explainer import LIMExplainer
from src.business.rules_engine import BusinessRulesEngine
from src.evaluation.confidence import ConfidenceModule
from src.utils.constants import PROCESSED_DATA_PATH, EXPERIMENTS_PATH

app = Flask(__name__)

# Global variables for loaded models
model = None
feature_names = None
shap_explainer = None
lime_explainer = None
rules_engine = None
confidence_module = None
training_data = None


def load_models():
    """Load all required models and components"""
    global model, feature_names, shap_explainer, lime_explainer, rules_engine, confidence_module, training_data

    print("\n" + "=" * 60)
    print("Loading AI-Powered Price Explanations System")
    print("=" * 60)

    # Load model
    model_path = EXPERIMENTS_PATH / 'model_xgboost.pkl'
    if model_path.exists():
        model = joblib.load(model_path)
        print("[OK] Model loaded")
    else:
        print("[WARNING] No trained model found. Please run run_experiments.py first")
        return False

    # Load feature names
    feature_path = PROCESSED_DATA_PATH / 'feature_names.json'
    if feature_path.exists():
        with open(feature_path, 'r') as f:
            feature_names = json.load(f)
        print(f"[OK] Feature names loaded: {feature_names}")

    # Load training data for confidence module (if available)
    data_path = PROCESSED_DATA_PATH / 'processed_data.csv'
    if data_path.exists():
        df = pd.read_csv(data_path)
        # Create feature matrix
        feature_cols = ['sales', 'cost', 'inventory', 'has_promo',
                        'seasonality_index', 'elasticity', 'competitor_price',
                        'item_encoded', 'store_encoded']
        training_data = df[feature_cols].values[:1000]  # Use subset for memory
        print("[OK] Training data loaded for confidence module")

    # Initialize explainers (with dummy background data)
    dummy_background = np.random.randn(100, len(feature_names))
    shap_explainer = SHAPExplainer(model, dummy_background)
    lime_explainer = LIMExplainer(model, dummy_background, feature_names)

    # Initialize confidence module
    confidence_module = ConfidenceModule(model, feature_names)
    if training_data is not None:
        confidence_module.set_training_data(training_data, np.zeros(len(training_data)))

    # Initialize business rules
    rules_engine = BusinessRulesEngine()

    print("[OK] All components initialized")
    return True


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint for price prediction with explanation"""
    try:
        data = request.json

        # Prepare input features
        feature_order = ['sales', 'cost', 'inventory', 'has_promo',
                         'seasonality_index', 'elasticity', 'competitor_price',
                         'item_encoded', 'store_encoded']

        # Get input values
        sales = float(data.get('sales', 1000))
        cost = float(data.get('cost', 5.00))
        inventory = float(data.get('inventory', 5000))
        has_promo = int(data.get('has_promo', 0))
        seasonality = float(data.get('seasonality_index', 1.0))
        elasticity = float(data.get('elasticity', -1.2))
        competitor_price = float(data.get('competitor_price', 8.50))
        item_encoded = int(data.get('item_encoded', 10))
        store_encoded = int(data.get('store_encoded', 2))

        # Create feature array
        features = np.array([[
            sales, cost, inventory, has_promo,
            seasonality, elasticity, competitor_price,
            item_encoded, store_encoded
        ]])

        # Predict
        predicted_price = model.predict(features)[0]

        # Apply business rules
        current_price = data.get('current_price')
        adjusted = rules_engine.apply_rules(
            predicted_price,
            current_price=current_price,
            inventory_level=inventory,
            is_holiday=data.get('is_holiday', False)
        )

        # Generate explanations
        shap_exp = shap_explainer.explain_prediction(features, feature_names)
        lime_exp = lime_explainer.explain_prediction(features)

        # Calculate confidence scores
        final_price = adjusted['adjusted_price']
        base_value = shap_exp['base_value']

        # Confidence Module
        uncertainty_scores = confidence_module.calculate_uncertainty_score(features[0], final_price)
        prediction_interval = confidence_module.calculate_prediction_interval(features[0])
        reliability_factors = confidence_module.get_reliability_factors(shap_exp, final_price)

        # Get top factors
        top_features = dict(list(shap_exp['features'].items())[:5])

        # Create custom explanation
        top_features_list = list(shap_exp['features'].items())[:3]
        explanation_parts = []
        for feature, details in top_features_list:
            impact_symbol = "+" if details['impact'] == 'positive' else "-"
            feature_name = feature.replace('_', ' ').title()
            explanation_parts.append(
                f"  * {impact_symbol} {feature_name}: {details['value']:.2f} "
                f"({impact_symbol} price by ${abs(details['shap_value']):.2f})"
            )

        custom_explanation = f"""
PRICE RECOMMENDATION: ${final_price:.2f}

Why this price?

The recommended price is based on the following key factors:

{chr(10).join(explanation_parts)}

Business Insight:
The base price would have been ${base_value:.2f}. The adjustments above reflect current market conditions, demand patterns, and competitive landscape.

Recommendation:
This price aligns with market dynamics and maximizes expected revenue.

Actionable Next Steps:
- Monitor competitor prices for elastic items
- Consider promotional bundling if inventory is high
- Review pricing strategy for items with high seasonality
        """

        # Determine confidence color
        confidence_level = uncertainty_scores['confidence_level']
        if confidence_level == 'High':
            confidence_color = 'green'
        elif confidence_level == 'Medium':
            confidence_color = 'orange'
        else:
            confidence_color = 'red'

        return jsonify({
            'success': True,
            'predicted_price': round(float(final_price), 2),
            'original_prediction': round(float(adjusted['original_price']), 2),
            'price_change_pct': round(adjusted['price_change_pct'] * 100, 1),
            'explanation': custom_explanation.strip(),
            'top_factors': top_features,
            'confidence_score': round(uncertainty_scores['reliability_score'], 2),
            'confidence_level': confidence_level,
            'confidence_color': confidence_color,
            'uncertainty_score': round(uncertainty_scores['uncertainty_score'], 3),
            'prediction_interval': {
                'lower': round(prediction_interval['lower_bound'], 2),
                'upper': round(prediction_interval['upper_bound'], 2)
            },
            'uncertainty_components': uncertainty_scores['components'],
            'warnings': reliability_factors['warnings'],
            'recommendations': reliability_factors['recommendations'],
            'business_rules_applied': adjusted['rules_applied'],
            'base_value': round(float(base_value), 2)
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)})


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'features_loaded': feature_names is not None,
        'confidence_module_ready': confidence_module is not None
    })


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("AI-Powered Price Explanations System")
    print("=" * 60 + "\n")

    if load_models():
        print("\n[READY] System ready!")
        print("[URL] Open http://localhost:5000 in your browser\n")
        app.run(debug=True, port=5000)
    else:
        print("\n[ERROR] Failed to load models. Please run: python run_experiments.py")