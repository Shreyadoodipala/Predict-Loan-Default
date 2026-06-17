"""
Flask API for Loan Default Prediction
Provides endpoints to predict loan default risk using the trained Decision Tree model.
"""

from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Use __file__ to determine the project root directory
# app.py is in src/, so parent.parent gives us the project root
PARENT = Path(__file__).parent.parent

# Get model directory from env or use default
MODEL_DIR = Path(os.getenv("MODEL_DIR", "models"))
if not MODEL_DIR.is_absolute():
    MODEL_DIR = PARENT / MODEL_DIR

# Initialize Flask app
app = Flask(__name__)

# Load the trained pipeline and preprocessor
try:
    pipeline = joblib.load(MODEL_DIR / "decision_tree_pipeline.pkl")
    preprocessor = joblib.load(MODEL_DIR / "preprocessor.pkl")
    print(f"✓ Models loaded successfully")
except FileNotFoundError as e:
    print(f"✗ Error: Could not find model files. {e}")
    pipeline = None
    preprocessor = None


# ==================== Helper Functions ====================

def validate_input(data):
    """
    Validate that all required features are present in the input data.
    
    Returns:
        tuple: (is_valid: bool, error_message: str or None)
    """
    required_features = [
        'credit_policy', 'purpose', 'interest_rate', 'installment',
        'log_annual_income', 'debt_income_ratio', 'fico', 'days_with_credit_line',
        'revolve_balance', 'revolve_utilized', 'inquiries_last_6_mon',
        'delinquent_2_yrs', 'public_recs'
    ]
    
    missing_features = [f for f in required_features if f not in data]
    if missing_features:
        return False, f"Missing required features: {', '.join(missing_features)}"
    
    return True, None


def preprocess_input(data):
    """
    Preprocess raw input data for prediction.
    
    Parameters:
        data (dict): Raw input features
        
    Returns:
        pd.DataFrame: Preprocessed data ready for model prediction
    """
    from src.feature_engineering import feature_engineering
    from src.preprocessing import FrequencyEncoder
    
    # Convert to DataFrame
    df = pd.DataFrame([data])
    
    # Apply feature engineering
    df = feature_engineering(df)
    
    # Apply frequency encoding for 'purpose' column
    freq_encoder = FrequencyEncoder()
    # Simple frequency encoding for demonstration - in production, this should be fitted on training data
    freq_encoder.fit(df, ['purpose'])
    df = freq_encoder.transform(df)
    
    return df


def make_prediction(input_data):
    """
    Make a prediction on the input data.
    
    Parameters:
        input_data (dict): Raw input features
        
    Returns:
        dict: Prediction result with label and probability
    """
    try:
        # Validate input
        is_valid, error_msg = validate_input(input_data)
        if not is_valid:
            return None, error_msg
        
        # Preprocess the input
        X = preprocess_input(input_data)
        
        # Make prediction
        prediction = pipeline.predict(X)[0]
        probability = pipeline.predict_proba(X)[0]
        
        return {
            'prediction': int(prediction),
            'probability_no_default': float(probability[0]),
            'probability_default': float(probability[1]),
            'default_risk': 'High' if prediction == 1 else 'Low'
        }, None
        
    except Exception as e:
        return None, f"Error during prediction: {str(e)}"


# ==================== API Routes ====================

@app.route('/', methods=['GET'])
def home():
    """Home endpoint - provides API documentation."""
    return jsonify({
        'message': 'Loan Default Prediction API',
        'version': '1.0.0',
        'endpoints': {
            'POST /predict': 'Make a prediction on a single loan application',
            'POST /predict-batch': 'Make predictions on multiple loan applications',
            'GET /health': 'Check API health status'
        }
    }), 200


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    if pipeline is None or preprocessor is None:
        return jsonify({
            'status': 'unhealthy',
            'message': 'Models not loaded'
        }), 503
    
    return jsonify({
        'status': 'healthy',
        'message': 'API is running and models are loaded'
    }), 200


@app.route('/predict', methods=['POST'])
def predict():
    """
    Make a prediction on a single loan application.
    
    Expected JSON format:
    {
        "credit_policy": int (0 or 1),
        "purpose": str (e.g., "debt_consolidation", "credit_card"),
        "interest_rate": float,
        "installment": float,
        "log_annual_income": float,
        "debt_income_ratio": float,
        "fico": int,
        "days_with_credit_line": float,
        "revolve_balance": float,
        "revolve_utilized": float,
        "inquiries_last_6_mon": int,
        "delinquent_2_yrs": int,
        "public_recs": int
    }
    """
    if pipeline is None:
        return jsonify({
            'error': 'Models not loaded. Cannot make predictions.'
        }), 503
    
    # Get JSON data
    data = request.get_json()
    if not data:
        return jsonify({
            'error': 'No JSON data provided'
        }), 400
    
    # Make prediction
    result, error = make_prediction(data)
    
    if error:
        return jsonify({
            'error': error
        }), 400
    
    return jsonify({
        'success': True,
        'input': data,
        'prediction': result
    }), 200


@app.route('/predict-batch', methods=['POST'])
def predict_batch():
    """
    Make predictions on multiple loan applications.
    
    Expected JSON format:
    {
        "records": [
            {loan application 1},
            {loan application 2},
            ...
        ]
    }
    """
    if pipeline is None:
        return jsonify({
            'error': 'Models not loaded. Cannot make predictions.'
        }), 503
    
    data = request.get_json()
    if not data or 'records' not in data:
        return jsonify({
            'error': 'Expected JSON with "records" key containing list of loan applications'
        }), 400
    
    records = data['records']
    if not isinstance(records, list):
        return jsonify({
            'error': '"records" must be a list'
        }), 400
    
    predictions = []
    errors = []
    
    for idx, record in enumerate(records):
        result, error = make_prediction(record)
        
        if error:
            errors.append({
                'record_index': idx,
                'error': error
            })
        else:
            predictions.append({
                'record_index': idx,
                'prediction': result
            })
    
    return jsonify({
        'success': len(errors) == 0,
        'total_records': len(records),
        'successful_predictions': len(predictions),
        'failed_predictions': len(errors),
        'predictions': predictions,
        'errors': errors if errors else None
    }), 200


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({
        'error': 'Endpoint not found',
        'message': 'Use GET / to see available endpoints'
    }), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    return jsonify({
        'error': 'Internal server error',
        'message': str(error)
    }), 500


# ==================== Main ====================

if __name__ == '__main__':
    # Run the Flask app
    print("Starting Loan Default Prediction API...")
    app.run(debug=True, host='0.0.0.0', port=5000)
