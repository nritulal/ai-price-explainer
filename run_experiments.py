"""Main execution script"""

import sys
import warnings

warnings.filterwarnings('ignore')

from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data.dataloader import DataLoader
from src.model.train_model import PricePredictor
from src.explainability.shap_explainer import SHAPExplainer
from src.explainability.lime_explainer import LIMExplainer
from src.explainability.comparator import ExplanationComparator
from src.llm.summarizer import LLMSummarizer
from src.business.rules_engine import BusinessRulesEngine
from src.evaluation.confidence import ConfidenceModule, UncertaintyEstimator
from src.utils.logger import setup_logger
from src.utils.constants import CONFIG, PROCESSED_DATA_PATH
import numpy as np
import json
import pandas as pd
import joblib


def print_data_preview(df, title="Data Preview"):
    """Print sample records and feature information"""
    print("\n" + "=" * 80)
    print(f"{title}")
    print("=" * 80)

    # Print feature names
    print("\n[FEATURES IN DATASET]")
    print("-" * 40)
    all_features = [col for col in df.columns if col not in ['date', 'item_id', 'store_id']]
    for i, col in enumerate(all_features, 1):
        print(f"  {i:2d}. {col}")

    # Print sample records
    print("\n[SAMPLE RECORDS - First 10 rows]")
    print("-" * 40)

    # Select columns to display
    display_cols = ['item_id', 'store_id', 'date', 'sales', 'cost', 'inventory',
                    'has_promo', 'seasonality_index', 'elasticity', 'competitor_price', 'price']

    existing_cols = [col for col in display_cols if col in df.columns]
    sample_df = df[existing_cols].head(10).copy()

    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    print(sample_df.to_string(index=True))

    # Print statistics
    print("\n[STATISTICS SUMMARY]")
    print("-" * 40)
    numeric_cols = ['sales', 'cost', 'inventory', 'seasonality_index', 'elasticity', 'competitor_price', 'price']
    existing_numeric = [col for col in numeric_cols if col in df.columns]
    if existing_numeric:
        stats = df[existing_numeric].describe()
        print(stats.round(2))


def main():
    logger = setup_logger('main')
    logger.info("=" * 60)
    logger.info("AI-Powered Price Explanations System")
    logger.info("=" * 60)

    # 1. Load and prepare data
    logger.info("Step 1: Loading and preparing data")
    dataloader = DataLoader(CONFIG)

    try:
        X_train, X_val, X_test, y_train, y_val, y_test, train_df, val_df, test_df = dataloader.process_pipeline()
    except FileNotFoundError as e:
        logger.error(f"Data files not found: {e}")
        logger.info("Please ensure M5 data is in: data/raw/m5_walmart/")
        return
    except Exception as e:
        logger.error(f"Error processing data: {e}")
        import traceback
        traceback.print_exc()
        return

    # Print data preview
    print_data_preview(train_df, "TRAINING DATA PREVIEW")

    # Check if we have data
    if X_train is None or len(X_train) == 0:
        logger.error("No training data available")
        return

    # 2. Train model
    logger.info("\nStep 2: Training price prediction model")
    predictor = PricePredictor()
    metrics = predictor.train(X_train, y_train, X_val, y_val)
    predictor.save_model()

    # Print feature importance
    print("\n" + "=" * 80)
    print("MODEL FEATURE IMPORTANCE")
    print("=" * 80)
    feature_importance = predictor.get_feature_importance(X_train.columns.tolist())
    for i, (feature, importance) in enumerate(feature_importance.items(), 1):
        print(f"  {i:2d}. {feature:20s}: {importance:.4f}")

    # 3. Setup explainers
    logger.info("\nStep 3: Setting up explainability layer")

    background_size = min(100, len(X_val))
    shap_explainer = SHAPExplainer(predictor.model, X_val[:background_size].values)
    lime_explainer = LIMExplainer(
        predictor.model,
        X_val[:background_size].values,
        X_train.columns.tolist()
    )
    comparator = ExplanationComparator(shap_explainer, lime_explainer)

    # 4. Setup Confidence Module
    logger.info("\nStep 4: Setting up Confidence Module")

    confidence_module = ConfidenceModule(predictor.model, X_train.columns.tolist())
    confidence_module.set_training_data(X_val.values, y_val.values)

    uncertainty_estimator = UncertaintyEstimator(predictor.model)
    uncertainty_estimator.create_ensemble(X_train.values, y_train.values, n_models=3)

    # 5. Test on sample
    logger.info("\nStep 5: Generating sample explanations")
    sample_idx = 0
    sample_instance = X_test.iloc[[sample_idx]].values
    sample_actual_price = y_test.iloc[sample_idx]

    # Print sample input features
    print("\n" + "=" * 80)
    print("SAMPLE PREDICTION - INPUT FEATURES")
    print("=" * 80)
    for i, col in enumerate(X_test.columns):
        print(f"  {col:20s}: {sample_instance[0, i]:.4f}")

    # SHAP explanation
    shap_exp = shap_explainer.explain_prediction(sample_instance, X_test.columns.tolist())

    # LIME explanation
    lime_exp = lime_explainer.explain_prediction(sample_instance)

    # Compare explanations
    comparison = comparator.compare_explanations(sample_instance, X_test.columns.tolist())

    # LLM summarization
    llm = LLMSummarizer(provider='mock')
    business_exp = llm.generate_explanation(shap_exp, lime_exp)

    # Business rules
    rules_engine = BusinessRulesEngine()
    adjusted_price = rules_engine.apply_rules(shap_exp['prediction'])

    # CONFIDENCE MODULE CALCULATIONS
    sample_prediction = shap_exp['prediction']
    uncertainty_scores = confidence_module.calculate_uncertainty_score(sample_instance[0], sample_prediction)
    prediction_interval = confidence_module.calculate_prediction_interval(sample_instance[0])
    reliability_factors = confidence_module.get_reliability_factors(shap_exp, sample_prediction)

    # Ensemble uncertainty
    ensemble_uncertainty = uncertainty_estimator.predict_with_uncertainty(sample_instance)

    # Calculate prediction error
    prediction_error = abs(sample_prediction - sample_actual_price)
    error_pct = (prediction_error / sample_actual_price) * 100 if sample_actual_price > 0 else 0

    # Print results
    print("\n" + "=" * 60)
    print("SAMPLE PRICE EXPLANATION")
    print("=" * 60)
    print(f"\nActual Price:     ${sample_actual_price:.2f}")
    print(f"Predicted Price:  ${sample_prediction:.2f}")
    print(f"Error:            ${prediction_error:.2f} ({error_pct:.1f}%)")

    print("\n[SHAP EXPLANATION - Top 5 Features]")
    print("-" * 40)
    for i, (feature, details) in enumerate(list(shap_exp['features'].items())[:5], 1):
        impact = "+" if details['impact'] == 'positive' else "-"
        print(f"  {i}. {feature:20s}: value={details['value']:.4f}, "
              f"impact={impact}{abs(details['shap_value']):.4f}")

    print(f"\n{business_exp}")

    print("\n[Business Rules Applied]")
    print("-" * 40)
    print(f"  Original Price: ${adjusted_price['original_price']:.2f}")
    print(f"  Adjusted Price: ${adjusted_price['adjusted_price']:.2f}")
    print(f"  Price Change:   {adjusted_price['price_change_pct'] * 100:.1f}%")
    if adjusted_price['rules_applied']:
        print(f"  Rules Applied:  {', '.join(adjusted_price['rules_applied'])}")

    # CONFIDENCE MODULE OUTPUT
    print("\n" + "=" * 60)
    print("CONFIDENCE MODULE OUTPUT")
    print("=" * 60)

    print(f"\n[Uncertainty & Reliability]")
    print("-" * 40)
    print(f"  Uncertainty Score:  {uncertainty_scores['uncertainty_score']:.3f} (0=certain, 1=uncertain)")
    print(f"  Reliability Score:  {uncertainty_scores['reliability_score']:.3f} (0=unreliable, 1=reliable)")
    print(f"  Confidence Level:   {uncertainty_scores['confidence_level']}")

    print(f"\n[Prediction Interval]")
    print("-" * 40)
    print(f"  Lower Bound (2.5%):  ${prediction_interval['lower_bound']:.2f}")
    print(f"  Upper Bound (97.5%): ${prediction_interval['upper_bound']:.2f}")
    print(f"  Interval Width:      ${prediction_interval['interval_width']:.2f}")

    print(f"\n[Ensemble Uncertainty]")
    print("-" * 40)
    print(f"  Mean Prediction:     ${ensemble_uncertainty['mean_prediction']:.2f}")
    print(f"  Std Deviation:       ${ensemble_uncertainty['std_prediction']:.2f}")
    print(f"  Model Agreement:     {ensemble_uncertainty['model_agreement']:.3f}")

    print(f"\n[Uncertainty Components]")
    print("-" * 40)
    for component, value in uncertainty_scores['components'].items():
        print(f"  {component:30s}: {value:.3f}")

    if reliability_factors['warnings']:
        print(f"\n[Warnings]")
        print("-" * 40)
        for warning in reliability_factors['warnings']:
            print(f"  WARNING: {warning}")

    print(f"\n[Recommendations]")
    print("-" * 40)
    for rec in reliability_factors['recommendations']:
        print(f"  RECOMMENDATION: {rec}")

    # Explanation agreement
    print(f"\n[Explanation Agreement]")
    print("-" * 40)
    print(f"  SHAP-LIME Agreement: {comparison['agreement_score'] * 100:.1f}%")
    print(f"  Correlation:         {comparison['correlation']:.3f}")

    # Save results
    results = {
        'model_metrics': metrics,
        'feature_importance': feature_importance,
        'sample_explanation': shap_exp,
        'lime_explanation': lime_exp,
        'business_explanation': business_exp,
        'adjusted_price': adjusted_price,
        'explanation_agreement': comparison['agreement_score'],
        'prediction_error': {
            'absolute': float(prediction_error),
            'percentage': float(error_pct)
        },
        'confidence_module': {
            'uncertainty_scores': uncertainty_scores,
            'prediction_interval': prediction_interval,
            'ensemble_uncertainty': ensemble_uncertainty,
            'reliability_factors': reliability_factors
        }
    }

    # Ensure directory exists
    Path('experiments').mkdir(exist_ok=True)

    with open('experiments/results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, default=str)

    logger.info("\nExperiment completed successfully!")
    logger.info(f"Model RMSE: {metrics['val_rmse']:.4f}")
    logger.info(f"Model R2:   {metrics['val_r2']:.4f}")
    logger.info(f"Confidence Level: {uncertainty_scores['confidence_level']}")
    logger.info("Results saved to experiments/results.json")

    # Print final summary
    print("\n" + "=" * 80)
    print("EXPERIMENT SUMMARY")
    print("=" * 80)
    print(f"\n  Total Training Samples: {len(X_train):,}")
    print(f"  Total Validation Samples: {len(X_val):,}")
    print(f"  Total Test Samples: {len(X_test):,}")
    print(f"  Model RMSE: ${metrics['val_rmse']:.4f}")
    print(f"  Model R2 Score: {metrics['val_r2']:.4f}")
    print(f"  Training Time: {metrics['training_time_seconds']:.2f} seconds")
    print(f"  Prediction Reliability: {uncertainty_scores['reliability_score']:.1%}")
    print(f"  Confidence Level: {uncertainty_scores['confidence_level']}")

    # ================================================================
    # SAVE THE SCALER - THIS IS THE CRITICAL FIX
    # ================================================================
    print("\n" + "=" * 80)
    print("SAVING ARTIFACTS FOR WEB APP")
    print("=" * 80)

    # Save scaler
    try:
        scaler = dataloader.scaler
        if scaler is not None:
            scaler_path = PROCESSED_DATA_PATH / 'scaler.pkl'
            joblib.dump(scaler, scaler_path)
            print(f"[OK] Scaler saved to {scaler_path}")
        else:
            print("[WARNING] Scaler is None, not saving")
    except Exception as e:
        print(f"[ERROR] Failed to save scaler: {e}")

    # Save encoders from FeatureEngineer
    if hasattr(dataloader.engineer, 'label_encoders'):
        le_item = dataloader.engineer.label_encoders.get('item')
        le_store = dataloader.engineer.label_encoders.get('store')

        if le_item:
            joblib.dump(le_item, PROCESSED_DATA_PATH / 'item_encoder.pkl')
            print("[OK] Saved item encoder")
        if le_store:
            joblib.dump(le_store, PROCESSED_DATA_PATH / 'store_encoder.pkl')
            print("[OK] Saved store encoder")

        # Save mappings
        if le_item:
            item_mapping = {idx: item for idx, item in enumerate(le_item.classes_)}
            with open(PROCESSED_DATA_PATH / 'item_mapping.json', 'w') as f:
                json.dump(item_mapping, f, indent=2)
            print("[OK] Saved item mapping")
        if le_store:
            store_mapping = {idx: store for idx, store in enumerate(le_store.classes_)}
            with open(PROCESSED_DATA_PATH / 'store_mapping.json', 'w') as f:
                json.dump(store_mapping, f, indent=2)
            print("[OK] Saved store mapping")

    # Save item and store mappings for the web app
    print("\n[INFO] Saving item and store mappings for web app...")

    # Get unique items and stores from training data
    unique_items = train_df['item_id'].unique().tolist()
    unique_stores = train_df['store_id'].unique().tolist()

    # Create mappings
    item_mapping = {i: item for i, item in enumerate(unique_items)}
    store_mapping = {i: store for i, store in enumerate(unique_stores)}

    # Save mappings
    with open(PROCESSED_DATA_PATH / 'item_mapping.json', 'w') as f:
        json.dump(item_mapping, f, indent=2)

    with open(PROCESSED_DATA_PATH / 'store_mapping.json', 'w') as f:
        json.dump(store_mapping, f, indent=2)

    # Save reverse mappings for lookup
    reverse_item_mapping = {item: i for i, item in enumerate(unique_items)}
    reverse_store_mapping = {store: i for i, store in enumerate(unique_stores)}

    with open(PROCESSED_DATA_PATH / 'reverse_item_mapping.json', 'w') as f:
        json.dump(reverse_item_mapping, f, indent=2)

    with open(PROCESSED_DATA_PATH / 'reverse_store_mapping.json', 'w') as f:
        json.dump(reverse_store_mapping, f, indent=2)

    print(f"[OK] Saved {len(unique_items)} items and {len(unique_stores)} stores for web app")

    # Save feature names
    with open(PROCESSED_DATA_PATH / 'feature_names.json', 'w') as f:
        json.dump(X_train.columns.tolist(), f, indent=2)
    print("[OK] Saved feature names")

    print("\n" + "=" * 80)
    print("ALL ARTIFACTS SAVED SUCCESSFULLY")
    print("=" * 80)
    print("\nYou can now run: python app.py")


if __name__ == "__main__":
    main()