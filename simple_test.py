#!/usr/bin/env python3
"""
Diamond Price Predictor - Simple Test
A basic test to verify the system works without heavy dependencies.
"""

import sys
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

def create_sample_data(n_samples=1000):
    """Create sample diamond data for testing"""
    np.random.seed(42)
    
    # Generate features
    carat = np.random.exponential(0.5, n_samples) + 0.2
    carat = np.clip(carat, 0.2, 5.0)
    
    cut_options = ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal']
    cut = np.random.choice(cut_options, n_samples, p=[0.05, 0.15, 0.25, 0.35, 0.20])
    
    color_options = ['D', 'E', 'F', 'G', 'H', 'I', 'J']
    color = np.random.choice(color_options, n_samples, p=[0.05, 0.10, 0.15, 0.20, 0.25, 0.15, 0.10])
    
    clarity_options = ['FL', 'IF', 'VVS1', 'VVS2', 'VS1', 'VS2', 'SI1', 'SI2', 'I1']
    clarity = np.random.choice(clarity_options, n_samples, p=[0.02, 0.03, 0.05, 0.08, 0.15, 0.20, 0.25, 0.20, 0.02])
    
    depth = np.random.normal(61.5, 2.0, n_samples)
    depth = np.clip(depth, 55, 70)
    
    table = np.random.normal(57.5, 3.0, n_samples)
    table = np.clip(table, 50, 70)
    
    # Calculate dimensions based on carat
    x = (carat * 6.5) ** (1/3) + np.random.normal(0, 0.1, n_samples)
    y = x + np.random.normal(0, 0.05, n_samples)
    z = x * depth / 100 + np.random.normal(0, 0.05, n_samples)
    
    x = np.clip(x, 3.0, 12.0)
    y = np.clip(y, 3.0, 12.0)
    z = np.clip(z, 1.5, 8.0)
    
    # Create realistic price based on features
    cut_multiplier = {'Fair': 0.8, 'Good': 0.9, 'Very Good': 1.0, 'Premium': 1.1, 'Ideal': 1.2}
    color_multiplier = {'D': 1.3, 'E': 1.2, 'F': 1.1, 'G': 1.0, 'H': 0.9, 'I': 0.8, 'J': 0.7}
    clarity_multiplier = {'FL': 2.0, 'IF': 1.8, 'VVS1': 1.6, 'VVS2': 1.4, 'VS1': 1.2, 'VS2': 1.0, 'SI1': 0.8, 'SI2': 0.6, 'I1': 0.4}
    
    base_price = 3000 * (carat ** 2.5)
    
    price = base_price * np.array([cut_multiplier[c] for c in cut]) * \
            np.array([color_multiplier[c] for c in color]) * \
            np.array([clarity_multiplier[c] for c in clarity])
    
    price = price * np.random.normal(1.0, 0.1, n_samples)
    price = np.clip(price, 300, 50000)
    price = np.round(price).astype(int)
    
    # Create DataFrame
    df = pd.DataFrame({
        'carat': np.round(carat, 2),
        'cut': cut,
        'color': color,
        'clarity': clarity,
        'depth': np.round(depth, 1),
        'table': np.round(table, 1),
        'price': price,
        'x': np.round(x, 2),
        'y': np.round(y, 2),
        'z': np.round(z, 2)
    })
    
    return df

def preprocess_data(df):
    """Preprocess the data for machine learning"""
    # Separate features and target
    X = df.drop('price', axis=1)
    y = df['price']
    
    # Encode categorical variables
    label_encoders = {}
    categorical_features = ['cut', 'color', 'clarity']
    
    for feature in categorical_features:
        le = LabelEncoder()
        X[feature] = le.fit_transform(X[feature])
        label_encoders[feature] = le
    
    return X, y, label_encoders

def train_model(X, y):
    """Train a Random Forest model"""
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train Random Forest
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    return model, r2, rmse, X_test, y_test, y_pred

def predict_diamond_price(model, label_encoders, diamond_features):
    """Predict price for a single diamond"""
    # Create DataFrame from input
    df = pd.DataFrame([diamond_features])
    
    # Encode categorical features
    for feature, encoder in label_encoders.items():
        if feature in df.columns:
            df[feature] = encoder.transform(df[feature])
    
    # Make prediction
    prediction = model.predict(df)[0]
    return prediction

def main():
    print("💎 Diamond Price Predictor - Simple Test")
    print("=" * 50)
    
    # Create sample data
    print("1. Creating sample diamond dataset...")
    df = create_sample_data(1000)
    print(f"   Dataset created with {len(df)} diamonds")
    print(f"   Price range: ${df['price'].min():,} - ${df['price'].max():,}")
    
    # Show sample data
    print("\n2. Sample data:")
    print(df.head())
    
    # Preprocess data
    print("\n3. Preprocessing data...")
    X, y, label_encoders = preprocess_data(df)
    print(f"   Features: {list(X.columns)}")
    
    # Train model
    print("\n4. Training Random Forest model...")
    model, r2, rmse, X_test, y_test, y_pred = train_model(X, y)
    print(f"   Model trained successfully!")
    print(f"   R² Score: {r2:.4f}")
    print(f"   RMSE: ${rmse:,.2f}")
    
    # Feature importance
    print("\n5. Feature importance:")
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    for _, row in feature_importance.iterrows():
        print(f"   {row['feature']}: {row['importance']:.4f}")
    
    # Example predictions
    print("\n6. Example predictions:")
    print("-" * 30)
    
    # High-quality diamond
    diamond1 = {
        'carat': 2.0,
        'cut': 'Ideal',
        'color': 'D',
        'clarity': 'VVS1',
        'depth': 61.5,
        'table': 57.0,
        'x': 8.1,
        'y': 8.1,
        'z': 5.0
    }
    
    pred1 = predict_diamond_price(model, label_encoders, diamond1)
    print(f"\nPremium Diamond (2.0ct, Ideal, D, VVS1): ${pred1:,.2f}")
    
    # Medium-quality diamond
    diamond2 = {
        'carat': 1.0,
        'cut': 'Good',
        'color': 'G',
        'clarity': 'SI1',
        'depth': 62.0,
        'table': 58.0,
        'x': 6.2,
        'y': 6.2,
        'z': 3.8
    }
    
    pred2 = predict_diamond_price(model, label_encoders, diamond2)
    print(f"Mid-Range Diamond (1.0ct, Good, G, SI1): ${pred2:,.2f}")
    
    # Budget diamond
    diamond3 = {
        'carat': 0.5,
        'cut': 'Fair',
        'color': 'J',
        'clarity': 'SI2',
        'depth': 64.0,
        'table': 60.0,
        'x': 5.0,
        'y': 5.0,
        'z': 3.2
    }
    
    pred3 = predict_diamond_price(model, label_encoders, diamond3)
    print(f"Budget Diamond (0.5ct, Fair, J, SI2): ${pred3:,.2f}")
    
    # Model performance visualization (text-based)
    print("\n7. Model Performance Analysis:")
    print("-" * 30)
    
    # Prediction accuracy by price range
    price_ranges = [(0, 2000), (2000, 5000), (5000, 10000), (10000, float('inf'))]
    range_names = ['Budget', 'Mid-Range', 'Premium', 'Luxury']
    
    for i, (min_price, max_price) in enumerate(price_ranges):
        mask = (y_test >= min_price) & (y_test < max_price)
        if mask.sum() > 0:
            range_r2 = r2_score(y_test[mask], y_pred[mask])
            range_rmse = np.sqrt(mean_squared_error(y_test[mask], y_pred[mask]))
            print(f"   {range_names[i]} (${min_price:,}-${max_price:,}): R² = {range_r2:.4f}, RMSE = ${range_rmse:,.0f}")
    
    # Summary
    print("\n" + "=" * 50)
    print("✅ SIMPLE TEST COMPLETED SUCCESSFULLY!")
    print("=" * 50)
    print(f"📊 Model Performance: R² = {r2:.4f}, RMSE = ${rmse:,.2f}")
    print(f"🎯 The model can predict diamond prices with {r2*100:.1f}% accuracy")
    print("\nNext steps:")
    print("1. 🌐 Try the full web interface: streamlit run streamlit_app.py")
    print("2. 📓 Open the Jupyter notebook for detailed analysis")
    print("3. 💻 Use the CLI: python cli.py --help")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        print("Please ensure you have pandas, numpy, and scikit-learn installed:")
        print("pip install pandas numpy scikit-learn")