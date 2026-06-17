# Loan Default Prediction API

A Flask-based REST API for predicting loan default risk using a trained Decision Tree model.

## Features

- **Single Prediction**: Make predictions on individual loan applications
- **Batch Predictions**: Process multiple loan applications in a single request
- **Input Validation**: Automatic validation of required fields
- **Data Preprocessing**: Automatic handling of feature engineering and scaling
- **Health Checks**: Monitor API status and model availability
- **Error Handling**: Comprehensive error messages for debugging


## Running the API

```bash
python src/app.py
```

The API will start on `http://localhost:5000` by default.

### Running with Production Server

For production deployment, use Gunicorn:

```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 src.app:app
```

## API Endpoints

### 1. Home Endpoint
**GET** `/`

Returns API information and available endpoints.

**Response:**
```json
{
    "message": "Loan Default Prediction API",
    "version": "1.0.0",
    "endpoints": {
        "POST /predict": "Make a prediction on a single loan application",
        "POST /predict-batch": "Make predictions on multiple loan applications",
        "GET /health": "Check API health status"
    }
}
```

---

### 2. Health Check
**GET** `/health`

Check if the API is running and models are loaded.

**Response (Healthy):**
```json
{
    "status": "healthy",
    "message": "API is running and models are loaded"
}
```

**Status Codes:**
- `200` - API is healthy
- `503` - Models not loaded

---

### 3. Single Prediction
**POST** `/predict`

Make a prediction on a single loan application.

**Request Body:**
```json
{
    "credit_policy": 1,
    "purpose": "debt_consolidation",
    "interest_rate": 0.1189,
    "installment": 829.1,
    "log_annual_income": 11.35040654,
    "debt_income_ratio": 19.48,
    "fico": 737,
    "days_with_credit_line": 5639.958333,
    "revolve_balance": 28854,
    "revolve_utilized": 52.1,
    "inquiries_last_6_mon": 0,
    "delinquent_2_yrs": 0,
    "public_recs": 0
}
```

**Input Field Descriptions:**
| Field | Type | Description |
|-------|------|-------------|
| `credit_policy` | int (0/1) | Credit policy indicator |
| `purpose` | string | Loan purpose (e.g., "debt_consolidation", "credit_card", "all_other") |
| `interest_rate` | float | Annual interest rate (0-1) |
| `installment` | float | Monthly installment amount ($) |
| `log_annual_income` | float | Log of annual income |
| `debt_income_ratio` | float | Debt-to-income ratio (%) |
| `fico` | int | FICO credit score (300-850) |
| `days_with_credit_line` | float | Days with credit line |
| `revolve_balance` | float | Revolving balance ($) |
| `revolve_utilized` | float | Revolving utilization (%) |
| `inquiries_last_6_mon` | int | Credit inquiries in last 6 months |
| `delinquent_2_yrs` | int | Delinquent accounts in 2 years |
| `public_recs` | int | Public records count |

**Response:**
```json
{
    "success": true,
    "input": {
        "credit_policy": 1,
        "purpose": "debt_consolidation",
        ...
    },
    "prediction": {
        "prediction": 0,
        "probability_no_default": 0.85,
        "probability_default": 0.15,
        "default_risk": "Low"
    }
}
```

**Prediction Output:**
| Field | Description |
|-------|-------------|
| `prediction` | 0 = Will not default, 1 = Will default |
| `probability_no_default` | Probability of NOT defaulting (0-1) |
| `probability_default` | Probability of defaulting (0-1) |
| `default_risk` | Risk level (Low = prediction 0, High = prediction 1) |

**Status Codes:**
- `200` - Prediction successful
- `400` - Bad request (missing/invalid fields)
- `503` - Models not loaded

---

### 4. Batch Predictions
**POST** `/predict-batch`

Make predictions on multiple loan applications.

**Request Body:**
```json
{
    "records": [
        {
            "credit_policy": 1,
            "purpose": "debt_consolidation",
            "interest_rate": 0.1189,
            "installment": 829.1,
            "log_annual_income": 11.35040654,
            "debt_income_ratio": 19.48,
            "fico": 737,
            "days_with_credit_line": 5639.958333,
            "revolve_balance": 28854,
            "revolve_utilized": 52.1,
            "inquiries_last_6_mon": 0,
            "delinquent_2_yrs": 0,
            "public_recs": 0
        },
        {
            "credit_policy": 1,
            "purpose": "credit_card",
            ...
        }
    ]
}
```

**Response:**
```json
{
    "success": true,
    "total_records": 2,
    "successful_predictions": 2,
    "failed_predictions": 0,
    "predictions": [
        {
            "record_index": 0,
            "prediction": {
                "prediction": 0,
                "probability_no_default": 0.85,
                "probability_default": 0.15,
                "default_risk": "Low"
            }
        },
        {
            "record_index": 1,
            "prediction": {
                "prediction": 1,
                "probability_no_default": 0.45,
                "probability_default": 0.55,
                "default_risk": "High"
            }
        }
    ],
    "errors": null
}
```

**Status Codes:**
- `200` - Predictions processed
- `400` - Bad request format
- `503` - Models not loaded

---

## Usage Examples

### Python (using requests)

```python
import requests
import json

BASE_URL = "http://localhost:5000"

# Single prediction
loan_data = {
    "credit_policy": 1,
    "purpose": "debt_consolidation",
    "interest_rate": 0.1189,
    "installment": 829.1,
    "log_annual_income": 11.35040654,
    "debt_income_ratio": 19.48,
    "fico": 737,
    "days_with_credit_line": 5639.958333,
    "revolve_balance": 28854,
    "revolve_utilized": 52.1,
    "inquiries_last_6_mon": 0,
    "delinquent_2_yrs": 0,
    "public_recs": 0
}

response = requests.post(f"{BASE_URL}/predict", json=loan_data)
print(json.dumps(response.json(), indent=2))
```

### cURL

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "credit_policy": 1,
    "purpose": "debt_consolidation",
    "interest_rate": 0.1189,
    "installment": 829.1,
    "log_annual_income": 11.35040654,
    "debt_income_ratio": 19.48,
    "fico": 737,
    "days_with_credit_line": 5639.958333,
    "revolve_balance": 28854,
    "revolve_utilized": 52.1,
    "inquiries_last_6_mon": 0,
    "delinquent_2_yrs": 0,
    "public_recs": 0
  }'
```

---

## Preprocessing Pipeline

The API automatically applies the following preprocessing steps:

1. **Feature Engineering**
   - `income_installment_ratio`: Annual income / (installment × 12)
   - `debt_to_credit_ratio`: Revolving balance / Annual income

2. **Categorical Encoding**
   - Frequency encoding for `purpose` column

3. **Numerical Scaling**
   - Standard scaling for normally distributed features
   - Power transformation + Standard scaling for mildly skewed features
   - Power transformation + Robust scaling for heavily skewed features

These transformations are learned from the training data and stored in `preprocessor.pkl`.


