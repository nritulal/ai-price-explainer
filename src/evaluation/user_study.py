"""User study framework for explanation quality"""

import pandas as pd
import numpy as np
from typing import List, Dict


class UserStudy:
    def __init__(self):
        self.results = []

    def create_survey_question(self, scenario: Dict) -> Dict:
        """Create a survey question for user study"""

        return {
            'scenario_id': scenario.get('id'),
            'product': scenario.get('product', 'Unknown'),
            'current_price': scenario.get('current_price'),
            'recommended_price': scenario.get('recommended_price'),
            'explanation': scenario.get('explanation'),
            'questions': [
                {
                    'text': 'How clear is this explanation? (1-5)',
                    'type': 'rating',
                    'scale': (1, 5)
                },
                {
                    'text': 'Do you trust this price recommendation? (1-5)',
                    'type': 'rating',
                    'scale': (1, 5)
                },
                {
                    'text': 'Would you use this system for pricing decisions? (Yes/No)',
                    'type': 'binary'
                }
            ]
        }

    def record_response(self, response: Dict):
        """Record user response"""
        self.results.append(response)

    def analyze_results(self) -> Dict:
        """Analyze user study results"""

        if not self.results:
            return {'error': 'No results recorded'}

        df = pd.DataFrame(self.results)

        return {
            'total_responses': len(df),
            'avg_clarity': df['clarity_score'].mean() if 'clarity_score' in df else None,
            'avg_trust': df['trust_score'].mean() if 'trust_score' in df else None,
            'adoption_rate': df['would_use'].mean() * 100 if 'would_use' in df else None,
            'qualitative_feedback': df['feedback'].tolist() if 'feedback' in df else []
        }