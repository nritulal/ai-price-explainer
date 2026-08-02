"""Test Local LLM integration"""

from src.llm.summarizer import LLMSummarizer

print("Testing Local LLM...")

# Initialize local LLM
llm = LLMSummarizer(provider='local', max_ram='4gb')

# Test with sample data
sample_shap = {
    'prediction': 8.40,
    'base_value': 4.42,
    'features': {
        'cost': {'value': 5.00, 'shap_value': 11.68, 'impact': 'positive'},
        'competitor_price': {'value': 8.50, 'shap_value': 3.60, 'impact': 'positive'},
        'inventory': {'value': 5000, 'shap_value': -4.50, 'impact': 'negative'},
        'seasonality_index': {'value': 1.2, 'shap_value': 0.05, 'impact': 'positive'}
    }
}

business_context = {
    'item_id': 'FOODS_1_001',
    'store_id': 'CA_1',
    'sales': 1000,
    'inventory': 5000,
    'competitor_price': 8.50,
    'current_price': 7.00
}

explanation = llm.generate_explanation(
    shap_explanation=sample_shap,
    business_context=business_context
)

print("\n" + "="*60)
print("LOCAL LLM EXPLANATION:")
print("="*60)
print(explanation)