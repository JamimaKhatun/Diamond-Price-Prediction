#!/usr/bin/env python3
"""
DiamondAI - Flask Web Application
Serves the frontend and provides API endpoints for diamond price prediction
"""

from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
import json

# Ensure src is importable
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE_DIR, 'src')
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# Lazy import of backend service
try:
    from backend_service import PredictionService
except Exception:
    PredictionService = None

app = Flask(
    __name__,
    static_folder='frontend',
    template_folder='frontend'
)
CORS(app)

# Mock user database (in production, use a real database)
users_db = {
    "demo@diamondai.com": {
        "id": 1,
        "name": "Demo User",
        "email": "demo@diamondai.com",
        "password": "demo123",  # In production, hash passwords!
        "company": "DiamondAI Demo",
        "plan": "Professional",
        "predictions_used": 0,
        "predictions_limit": 10000,
        "created_at": "2024-01-01T00:00:00Z"
    }
}

# Prediction history storage (in production, use a database)
prediction_history = []

@app.route('/')
def index():
    """Serve the main frontend application"""
    return send_from_directory('frontend', 'index.html')

@app.route('/<path:filename>')
def static_files(filename):
    """Serve static files"""
    return send_from_directory('frontend', filename)

# Initialize prediction service (with graceful fallback)
prediction_service = PredictionService() if PredictionService else None

@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "ml_available": bool(prediction_service and prediction_service.is_ready() and not prediction_service.using_mock),
        "using_mock": bool(prediction_service and prediction_service.using_mock),
        "model": getattr(prediction_service, 'model_name', 'Mock Predictor'),
        "timestamp": datetime.now().isoformat()
    })

@app.route('/api/auth/login', methods=['POST'])
def login():
    """Handle user login"""
    data = request.get_json()
    email = data.get('email')
    password = data.get('password')
    
    # Check credentials (in production, use proper authentication)
    user = users_db.get(email)
    if user and user['password'] == password:
        # Return user data (exclude password)
        user_data = {k: v for k, v in user.items() if k != 'password'}
        return jsonify({
            "success": True,
            "user": user_data,
            "token": f"mock_token_{user['id']}"  # In production, use JWT
        })
    else:
        return jsonify({
            "success": False,
            "error": "Invalid email or password"
        }), 401

@app.route('/api/auth/signup', methods=['POST'])
def signup():
    """Handle user registration"""
    data = request.get_json()
    
    # Extract user data
    first_name = data.get('firstName', '')
    last_name = data.get('lastName', '')
    email = data.get('email', '')
    company = data.get('company', 'Individual')
    password = data.get('password', '')
    
    # Check if user already exists
    if email in users_db:
        return jsonify({
            "success": False,
            "error": "User with this email already exists"
        }), 400
    
    # Create new user
    user_id = len(users_db) + 1
    new_user = {
        "id": user_id,
        "name": f"{first_name} {last_name}".strip(),
        "email": email,
        "password": password,  # In production, hash this!
        "company": company,
        "plan": "Starter",
        "predictions_used": 0,
        "predictions_limit": 1000,
        "created_at": datetime.now().isoformat()
    }
    
    users_db[email] = new_user
    
    # Return user data (exclude password)
    user_data = {k: v for k, v in new_user.items() if k != 'password'}
    return jsonify({
        "success": True,
        "user": user_data,
        "token": f"mock_token_{user_id}"
    })

@app.route('/api/predict', methods=['POST'])
def predict_diamond_price():
    """Predict diamond price using ML models (backend_service) with mock fallback"""
    try:
        data = request.get_json() or {}
        
        # Extract diamond characteristics
        diamond_data = {
            'carat': float(data.get('carat', 1.0)),
            'cut': data.get('cut', 'Good'),
            'color': data.get('color', 'G'),
            'clarity': data.get('clarity', 'VS1'),
            'depth': float(data.get('depth', 61.5)),
            'table': float(data.get('table', 57.0)),
            'x': float(data.get('x', 6.0)),
            'y': float(data.get('y', 6.0)),
            'z': float(data.get('z', 3.7))
        }
        
        # Predict via service (returns camelCase keys)
        if prediction_service and prediction_service.is_ready():
            service_pred = prediction_service.predict(diamond_data)
            prediction = {
                'price': service_pred['price'],
                'confidence': service_pred['confidence'],
                'price_range': {
                    'min': service_pred['priceRange']['min'],
                    'max': service_pred['priceRange']['max']
                },
                'price_per_carat': service_pred['pricePerCarat'],
                'category': service_pred['category'],
                'category_icon': service_pred['categoryIcon'],
                'model': service_pred.get('model', 'Predictor')
            }
        else:
            prediction = make_mock_prediction(diamond_data)
        
        # Store prediction in history
        prediction_record = {
            "id": len(prediction_history) + 1,
            "timestamp": datetime.now().isoformat(),
            "input": diamond_data,
            "prediction": prediction,
            "user_id": data.get('user_id', 'anonymous')
        }
        prediction_history.append(prediction_record)
        
        return jsonify({
            "success": True,
            "prediction": prediction,
            "timestamp": prediction_record["timestamp"]
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Prediction failed: {str(e)}"
        }), 500

def make_mock_prediction(diamond_data):
    """Make mock prediction for demonstration"""
    # Calculate base price using realistic diamond pricing logic
    base_price = 3000
    
    # Carat weight (most important factor - exponential relationship)
    base_price *= (diamond_data['carat'] ** 2.5)
    
    # Cut quality multiplier
    cut_multipliers = {
        'Fair': 0.8,
        'Good': 0.9,
        'Very Good': 1.0,
        'Premium': 1.1,
        'Ideal': 1.2
    }
    base_price *= cut_multipliers.get(diamond_data['cut'], 1.0)
    
    # Color grade multiplier (D is best, J is worst in our range)
    color_multipliers = {
        'D': 1.3, 'E': 1.2, 'F': 1.1, 'G': 1.0,
        'H': 0.9, 'I': 0.8, 'J': 0.7
    }
    base_price *= color_multipliers.get(diamond_data['color'], 1.0)
    
    # Clarity multiplier (FL is best, I1 is worst)
    clarity_multipliers = {
        'FL': 2.0, 'IF': 1.8, 'VVS1': 1.6, 'VVS2': 1.4,
        'VS1': 1.2, 'VS2': 1.0, 'SI1': 0.8, 'SI2': 0.6, 'I1': 0.4
    }
    base_price *= clarity_multipliers.get(diamond_data['clarity'], 1.0)
    
    # Ensure minimum price
    base_price = max(base_price, 300)
    
    # Add some realistic variance
    variance = np.random.normal(1.0, 0.05)  # 5% standard deviation
    predicted_price = int(base_price * variance)
    
    # Calculate confidence (higher for typical diamonds)
    confidence = 90 + np.random.randint(0, 8)  # 90-97%
    
    # Price range (confidence interval)
    price_range = {
        'min': int(predicted_price * 0.85),
        'max': int(predicted_price * 1.15)
    }
    
    # Determine category
    if predicted_price < 2000:
        category = 'Budget-Friendly'
        category_icon = '💚'
    elif predicted_price < 5000:
        category = 'Mid-Range'
        category_icon = '💛'
    elif predicted_price < 10000:
        category = 'Premium'
        category_icon = '🧡'
    else:
        category = 'Luxury'
        category_icon = '❤️'
    
    return {
        'price': predicted_price,
        'confidence': confidence,
        'price_range': price_range,
        'price_per_carat': int(predicted_price / diamond_data['carat']),
        'category': category,
        'category_icon': category_icon,
        'model': 'Mock Predictor',
        'features_importance': {
            'carat': 0.8615,
            'clarity': 0.0452,
            'color': 0.0274,
            'cut': 0.0067,
            'dimensions': 0.0301,
            'proportions': 0.0291
        }
    }

@app.route('/api/history')
def get_prediction_history():
    """Get user's prediction history"""
    user_id = request.args.get('user_id', 'anonymous')
    
    # Filter history by user (in production, use proper authentication)
    user_history = [
        record for record in prediction_history 
        if record.get('user_id') == user_id
    ]
    
    return jsonify({
        "success": True,
        "history": user_history[-50:]  # Return last 50 predictions
    })

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors by serving the main app (for SPA routing)"""
    return send_from_directory('frontend', 'index.html')

if __name__ == '__main__':
    print("🚀 Starting DiamondAI Web Application...")
    print("=" * 50)
    print("📊 ML Models Available: ❌ No (using mock predictions)")
    print("🌐 Frontend: Serving from /frontend directory")
    print("🔗 API Endpoints: /api/*")
    print("=" * 50)
    print("🎯 Demo Login Credentials:")
    print("   Email: demo@diamondai.com")
    print("   Password: demo123")
    print("=" * 50)
    print("📱 Access the application at: http://localhost:5000")
    print("🔧 API Health Check: http://localhost:5000/api/health")
    print("=" * 50)
    
    # Run the application
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True,
        threaded=True
    )