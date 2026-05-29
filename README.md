# AI-Powered Price Explanations

## Bringing Transparency to Retail Pricing Decisions

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.7+-orange.svg)](https://xgboost.ai/)
[![Flask](https://img.shields.io/badge/Flask-2.3+-green.svg)](https://flask.palletsprojects.com/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## Overview

This project implements an explainable AI system for retail price recommendations. It combines XGBoost price prediction with SHAP and LIME explanations, includes a confidence module for uncertainty estimation, and provides a web interface for interactive pricing decisions.

## Features

- **Price Prediction**: XGBoost model trained on M5 Walmart data
- **Explainability**: SHAP and LIME for transparent decisions
- **Confidence Module**: Uncertainty estimation and reliability scores
- **Business Rules**: Price caps, change limits, inventory adjustments
- **Web Interface**: Interactive demo application
- **API Endpoints**: RESTful API for integration

## Project Structure
.
├── config/ # Configuration files
├── data/ # Data directory (gitignored)
├── experiments/ # Model checkpoints (gitignored)
├── notebooks/ # Jupyter notebooks
├── scripts/ # Utility scripts
├── src/ # Source code
│ ├── business/ # Business rules engine
│ ├── data/ # Data loading & preprocessing
│ ├── evaluation/ # Metrics & confidence module
│ ├── explainability/ # SHAP & LIME explainers
│ ├── llm/ # LLM summarizer
│ ├── model/ # Model training & prediction
│ └── utils/ # Helpers & constants
├── templates/ # HTML templates
├── tests/ # Unit tests
├── app.py # Flask web application
└── run_experiments.py # Main execution script



## Installation

### Prerequisites
- Python 3.9 or higher
- M5 Forecasting dataset (place in `data/raw/m5_walmart/`)

### Setup

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/ai-price-explainer.git
cd ai-price-explainer

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Place M5 data in data/raw/m5_walmart/
# Then run the pipeline
python run_experiments.py

# Start the web application
python app.py