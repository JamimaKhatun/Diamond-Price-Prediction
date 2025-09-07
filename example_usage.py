#!/usr/bin/env python3
"""
Diamond Price Predictor - Example Usage
This script demonstrates how to use the Diamond Price Predictor system.
"""

import sys
import os
import pandas as pd
import numpy as np

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from diamond_predictor import DiamondPricePredictor

def main():
    print("💎 Diamond Price Predictor - Example Usage")
    print("=" * 60)
    
    # Initialize the predictor
    print("\n1. Initializing Diamond Price Predictor...")
    predictor = DiamondPricePredictor()
    
    # Load and prepare data
    print("\n2. Loading and preparing data...")
    data = predictor.load_and_prepare_data(file_path=None, explore=False, visualize=False)
    print(f"   Dataset shape: {data.shape}")
    print(f"   Price range: ${data['price'].min():.0f} - ${data['price'].max():.0f}")
    
    # Train models (reduced epochs for faster demo)
    print("\n3. Training machine learning models...")
    print("   This may take a few minutes...")
    
    # Train traditional ML models
    print("   Training traditional ML models...")
    ml_results = predictor.train_traditional_ml_models(optimize_best=False)
    
    # Train deep learning models (fewer epochs for demo)
    print("   Training deep learning models...")
    dl_results = predictor.train_deep_learning_models(epochs=20, batch_size=32)
    
    # Compare all models
    print("\n4. Comparing model performance...")
    comparison_results = predictor.compare_all_models()
    
    print("\nTop 5 Models:")
    print("-" * 50)
    for i, (_, row) in enumerate(comparison_results.head().iterrows()):
        medal = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else "🏅"
        print(f"{medal} {row['Model']}: R² = {row['R² Score']:.4f}, RMSE = ${row['RMSE']:.2f}")
    
    # Example predictions
    print("\n5. Example diamond price predictions...")
    print("=" * 50)
    
    # Example 1: High-quality diamond
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
    
    print("\nExample 1: Premium Diamond")
    print("Characteristics:")
    for key, value in diamond1.items():
        print(f"  {key.title()}: {value}")
    
    pred1 = predictor.predict_price(diamond1, 'best')
    if isinstance(pred1, np.ndarray):
        pred1 = pred1[0]
    print(f"💰 Predicted Price: ${pred1:,.2f}")
    
    # Example 2: Medium-quality diamond
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
    
    print("\nExample 2: Mid-Range Diamond")
    print("Characteristics:")
    for key, value in diamond2.items():
        print(f"  {key.title()}: {value}")
    
    pred2 = predictor.predict_price(diamond2, 'best')
    if isinstance(pred2, np.ndarray):
        pred2 = pred2[0]
    print(f"💰 Predicted Price: ${pred2:,.2f}")
    
    # Example 3: Budget diamond
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
    
    print("\nExample 3: Budget Diamond")
    print("Characteristics:")
    for key, value in diamond3.items():
        print(f"  {key.title()}: {value}")
    
    pred3 = predictor.predict_price(diamond3, 'best')
    if isinstance(pred3, np.ndarray):
        pred3 = pred3[0]
    print(f"💰 Predicted Price: ${pred3:,.2f}")
    
    # Batch prediction example
    print("\n6. Batch prediction example...")
    print("=" * 50)
    
    # Create sample batch data
    batch_diamonds = pd.DataFrame([
        {'carat': 1.5, 'cut': 'Premium', 'color': 'E', 'clarity': 'VS1', 'depth': 61.0, 'table': 56.0, 'x': 7.0, 'y': 7.0, 'z': 4.3},
        {'carat': 0.8, 'cut': 'Very Good', 'color': 'F', 'clarity': 'VS2', 'depth': 62.5, 'table': 57.5, 'x': 5.8, 'y': 5.8, 'z': 3.6},
        {'carat': 2.5, 'cut': 'Ideal', 'color': 'D', 'clarity': 'IF', 'depth': 61.2, 'table': 56.5, 'x': 8.8, 'y': 8.8, 'z': 5.4},
    ])
    
    print("Batch diamonds:")
    print(batch_diamonds.to_string(index=False))
    
    batch_predictions = []
    for _, row in batch_diamonds.iterrows():
        pred = predictor.predict_price(row.to_dict(), 'best')
        if isinstance(pred, np.ndarray):
            pred = pred[0]
        batch_predictions.append(pred)
    
    batch_diamonds['predicted_price'] = batch_predictions
    
    print("\nBatch predictions:")
    print(batch_diamonds[['carat', 'cut', 'color', 'clarity', 'predicted_price']].to_string(index=False))
    
    # Feature importance analysis
    print("\n7. Feature importance analysis...")
    print("=" * 50)
    
    importance_df = predictor.get_feature_importance_analysis()
    if importance_df is not None:
        print("Top 5 most important features:")
        top_features = importance_df['Mean'].sort_values(ascending=False).head()
        for feature, importance in top_features.items():
            print(f"  {feature}: {importance:.4f}")
    
    # Model performance summary
    print("\n8. Model performance summary...")
    print("=" * 50)
    
    best_model = comparison_results.iloc[0]
    print(f"🏆 Best Overall Model: {best_model['Model']}")
    print(f"   Type: {best_model['Type']}")
    print(f"   R² Score: {best_model['R² Score']:.4f}")
    print(f"   RMSE: ${best_model['RMSE']:.2f}")
    print(f"   MAE: ${best_model['MAE']:.2f}")
    
    # Performance by model type
    type_performance = comparison_results.groupby('Type').agg({
        'R² Score': 'mean',
        'RMSE': 'mean',
        'MAE': 'mean'
    }).round(4)
    
    print(f"\nAverage performance by model type:")
    for model_type, metrics in type_performance.iterrows():
        print(f"  {model_type}:")
        print(f"    R² Score: {metrics['R² Score']:.4f}")
        print(f"    RMSE: ${metrics['RMSE']:.2f}")
        print(f"    MAE: ${metrics['MAE']:.2f}")
    
    # Save models
    print("\n9. Saving trained models...")
    predictor.save_all_models("example_saved_models")
    print("   Models saved to 'example_saved_models' directory")
    
    # Summary
    print("\n" + "=" * 60)
    print("EXAMPLE COMPLETED SUCCESSFULLY! 🎉")
    print("=" * 60)
    print("\nWhat you can do next:")
    print("1. 🌐 Run the web interface: streamlit run streamlit_app.py")
    print("2. 📓 Open the Jupyter notebook for detailed analysis")
    print("3. 💻 Use the CLI for automation: python cli.py --help")
    print("4. 🔧 Modify the code to add your own models or features")
    print("\nFor more information, check the README.md file!")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Example interrupted by user. Goodbye!")
    except Exception as e:
        print(f"\n❌ Error running example: {str(e)}")
        print("Please check that all dependencies are installed correctly.")