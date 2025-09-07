#!/usr/bin/env python3
"""
Diamond Price Predictor - Command Line Interface
A comprehensive CLI for diamond price prediction using ML and Deep Learning.
"""

import argparse
import sys
import os
import json
import pandas as pd
import numpy as np
from pathlib import Path

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from diamond_predictor import DiamondPricePredictor

class DiamondPredictorCLI:
    def __init__(self):
        self.predictor = DiamondPricePredictor()
        self.models_trained = False
        
    def load_data(self, file_path=None, explore=False):
        """Load and prepare data"""
        print("🔄 Loading diamond dataset...")
        try:
            data = self.predictor.load_and_prepare_data(
                file_path=file_path, 
                explore=explore, 
                visualize=False
            )
            print(f"✅ Data loaded successfully! Shape: {data.shape}")
            return data
        except Exception as e:
            print(f"❌ Error loading data: {str(e)}")
            return None
    
    def train_models(self, ml_only=False, dl_only=False, epochs=50, batch_size=32):
        """Train machine learning models"""
        print("\n🚀 Starting model training...")
        
        try:
            if not ml_only:
                print("🧠 Training Deep Learning models...")
                self.predictor.train_deep_learning_models(epochs=epochs, batch_size=batch_size)
                print("✅ Deep Learning models trained!")
            
            if not dl_only:
                print("🤖 Training Traditional ML models...")
                self.predictor.train_traditional_ml_models(optimize_best=True)
                print("✅ Traditional ML models trained!")
            
            if not ml_only and not dl_only:
                print("📊 Comparing all models...")
                comparison = self.predictor.compare_all_models()
                print("✅ Model comparison completed!")
                
                # Display top 5 models
                print("\n🏆 Top 5 Models:")
                print("-" * 60)
                for i, (_, row) in enumerate(comparison.head().iterrows()):
                    medal = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else "🏅"
                    print(f"{medal} {row['Model']}: R² = {row['R² Score']:.4f}, RMSE = ${row['RMSE']:.2f}")
            
            self.models_trained = True
            print("\n✅ All models trained successfully!")
            
        except Exception as e:
            print(f"❌ Error training models: {str(e)}")
    
    def predict_single(self, diamond_features, model_type='best'):
        """Predict price for a single diamond"""
        if not self.models_trained:
            print("❌ No models trained. Please train models first.")
            return None
        
        try:
            print("\n💎 Diamond Price Prediction")
            print("=" * 50)
            
            # Display input features
            print("Input characteristics:")
            for key, value in diamond_features.items():
                print(f"  {key.title()}: {value}")
            
            # Make prediction
            prediction = self.predictor.predict_price(diamond_features, model_type)
            if isinstance(prediction, np.ndarray):
                prediction = prediction[0]
            
            print(f"\n💰 Predicted Price: ${prediction:,.2f}")
            
            # Price category
            if prediction < 1000:
                category = "Budget-Friendly 💚"
            elif prediction < 5000:
                category = "Mid-Range 💛"
            elif prediction < 15000:
                category = "Premium 🧡"
            else:
                category = "Luxury ❤️"
            
            print(f"📊 Price Category: {category}")
            
            return prediction
            
        except Exception as e:
            print(f"❌ Error making prediction: {str(e)}")
            return None
    
    def predict_batch(self, input_file, output_file=None, model_type='best'):
        """Predict prices for multiple diamonds from CSV file"""
        if not self.models_trained:
            print("❌ No models trained. Please train models first.")
            return
        
        try:
            print(f"\n📊 Batch prediction from {input_file}")
            
            # Load data
            df = pd.read_csv(input_file)
            print(f"Loaded {len(df)} diamonds for prediction")
            
            # Make predictions
            predictions = []
            print("🔄 Making predictions...")
            
            for i, (_, row) in enumerate(df.iterrows()):
                if i % 100 == 0:
                    print(f"  Processed {i}/{len(df)} diamonds...")
                
                pred = self.predictor.predict_price(row.to_dict(), model_type)
                if isinstance(pred, np.ndarray):
                    pred = pred[0]
                predictions.append(pred)
            
            # Add predictions to dataframe
            df['predicted_price'] = predictions
            
            # Save results
            if output_file is None:
                output_file = input_file.replace('.csv', '_predictions.csv')
            
            df.to_csv(output_file, index=False)
            print(f"✅ Predictions saved to {output_file}")
            
            # Display statistics
            print(f"\n📈 Prediction Statistics:")
            print(f"  Mean Price: ${np.mean(predictions):,.2f}")
            print(f"  Median Price: ${np.median(predictions):,.2f}")
            print(f"  Min Price: ${np.min(predictions):,.2f}")
            print(f"  Max Price: ${np.max(predictions):,.2f}")
            
        except Exception as e:
            print(f"❌ Error in batch prediction: {str(e)}")
    
    def save_models(self, directory="saved_models"):
        """Save trained models"""
        if not self.models_trained:
            print("❌ No models to save. Please train models first.")
            return
        
        try:
            print(f"💾 Saving models to {directory}...")
            self.predictor.save_all_models(directory)
            print("✅ Models saved successfully!")
        except Exception as e:
            print(f"❌ Error saving models: {str(e)}")
    
    def load_models(self, directory="saved_models"):
        """Load saved models"""
        try:
            print(f"📂 Loading models from {directory}...")
            self.predictor.load_models(directory)
            self.models_trained = True
            print("✅ Models loaded successfully!")
        except Exception as e:
            print(f"❌ Error loading models: {str(e)}")
    
    def show_model_summary(self):
        """Display model performance summary"""
        if not self.models_trained:
            print("❌ No models trained. Please train models first.")
            return
        
        print("\n📊 Model Performance Summary")
        print("=" * 60)
        
        # ML models summary
        if hasattr(self.predictor, 'ml_results') and self.predictor.ml_results:
            print(self.predictor.ml_models.get_model_summary())
        
        # DL models summary
        if hasattr(self.predictor, 'dl_results') and self.predictor.dl_results:
            print(self.predictor.dl_models.get_model_summary())
        
        # Overall comparison
        if hasattr(self.predictor, 'comparison_results') and not self.predictor.comparison_results.empty:
            best_model = self.predictor.comparison_results.iloc[0]
            print(f"\n🏆 OVERALL BEST MODEL: {best_model['Model']}")
            print(f"   R² Score: {best_model['R² Score']:.4f}")
            print(f"   RMSE: ${best_model['RMSE']:.2f}")
            print(f"   MAE: ${best_model['MAE']:.2f}")
    
    def interactive_prediction(self):
        """Interactive prediction mode"""
        if not self.models_trained:
            print("❌ No models trained. Please train models first.")
            return
        
        print("\n🎯 Interactive Diamond Price Prediction")
        print("=" * 50)
        print("Enter diamond characteristics (press Enter for default values):")
        
        try:
            # Get input from user
            carat = float(input("Carat (default 1.0): ") or "1.0")
            
            cut_options = ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal']
            print(f"Cut options: {', '.join(cut_options)}")
            cut = input("Cut (default Good): ") or "Good"
            if cut not in cut_options:
                print(f"Invalid cut. Using 'Good'")
                cut = "Good"
            
            color_options = ['D', 'E', 'F', 'G', 'H', 'I', 'J']
            print(f"Color options: {', '.join(color_options)}")
            color = input("Color (default G): ") or "G"
            if color not in color_options:
                print(f"Invalid color. Using 'G'")
                color = "G"
            
            clarity_options = ['FL', 'IF', 'VVS1', 'VVS2', 'VS1', 'VS2', 'SI1', 'SI2', 'I1']
            print(f"Clarity options: {', '.join(clarity_options)}")
            clarity = input("Clarity (default VS1): ") or "VS1"
            if clarity not in clarity_options:
                print(f"Invalid clarity. Using 'VS1'")
                clarity = "VS1"
            
            depth = float(input("Depth % (default 61.5): ") or "61.5")
            table = float(input("Table % (default 57.0): ") or "57.0")
            x = float(input("Length mm (default 6.0): ") or "6.0")
            y = float(input("Width mm (default 6.0): ") or "6.0")
            z = float(input("Height mm (default 3.7): ") or "3.7")
            
            diamond_features = {
                'carat': carat,
                'cut': cut,
                'color': color,
                'clarity': clarity,
                'depth': depth,
                'table': table,
                'x': x,
                'y': y,
                'z': z
            }
            
            # Make prediction
            self.predict_single(diamond_features)
            
        except ValueError as e:
            print(f"❌ Invalid input: {str(e)}")
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
        except Exception as e:
            print(f"❌ Error: {str(e)}")

def main():
    parser = argparse.ArgumentParser(
        description="Diamond Price Predictor CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Complete analysis with synthetic data
  python cli.py --complete-analysis
  
  # Train models only
  python cli.py --train --epochs 100
  
  # Predict single diamond price
  python cli.py --predict --carat 1.5 --cut Ideal --color D --clarity VVS1
  
  # Batch prediction
  python cli.py --batch-predict input.csv --output output.csv
  
  # Interactive mode
  python cli.py --interactive
  
  # Load saved models and predict
  python cli.py --load-models --predict --carat 2.0 --cut Premium
        """
    )
    
    # Main actions
    parser.add_argument('--complete-analysis', action='store_true',
                       help='Run complete analysis (load data, train models, compare)')
    parser.add_argument('--train', action='store_true',
                       help='Train models')
    parser.add_argument('--predict', action='store_true',
                       help='Predict diamond price')
    parser.add_argument('--batch-predict', type=str,
                       help='Batch predict from CSV file')
    parser.add_argument('--interactive', action='store_true',
                       help='Interactive prediction mode')
    
    # Data options
    parser.add_argument('--data-file', type=str,
                       help='Path to diamond dataset CSV file')
    parser.add_argument('--explore-data', action='store_true',
                       help='Show data exploration')
    
    # Training options
    parser.add_argument('--ml-only', action='store_true',
                       help='Train only traditional ML models')
    parser.add_argument('--dl-only', action='store_true',
                       help='Train only deep learning models')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of epochs for deep learning (default: 50)')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for deep learning (default: 32)')
    
    # Prediction options
    parser.add_argument('--carat', type=float, default=1.0,
                       help='Diamond carat (default: 1.0)')
    parser.add_argument('--cut', type=str, default='Good',
                       choices=['Fair', 'Good', 'Very Good', 'Premium', 'Ideal'],
                       help='Diamond cut (default: Good)')
    parser.add_argument('--color', type=str, default='G',
                       choices=['D', 'E', 'F', 'G', 'H', 'I', 'J'],
                       help='Diamond color (default: G)')
    parser.add_argument('--clarity', type=str, default='VS1',
                       choices=['FL', 'IF', 'VVS1', 'VVS2', 'VS1', 'VS2', 'SI1', 'SI2', 'I1'],
                       help='Diamond clarity (default: VS1)')
    parser.add_argument('--depth', type=float, default=61.5,
                       help='Diamond depth percentage (default: 61.5)')
    parser.add_argument('--table', type=float, default=57.0,
                       help='Diamond table percentage (default: 57.0)')
    parser.add_argument('--x', type=float, default=6.0,
                       help='Diamond length in mm (default: 6.0)')
    parser.add_argument('--y', type=float, default=6.0,
                       help='Diamond width in mm (default: 6.0)')
    parser.add_argument('--z', type=float, default=3.7,
                       help='Diamond height in mm (default: 3.7)')
    parser.add_argument('--model-type', type=str, default='best',
                       choices=['best', 'ml', 'dl', 'ensemble'],
                       help='Model type for prediction (default: best)')
    
    # File operations
    parser.add_argument('--output', type=str,
                       help='Output file for batch predictions')
    parser.add_argument('--save-models', action='store_true',
                       help='Save trained models')
    parser.add_argument('--load-models', action='store_true',
                       help='Load saved models')
    parser.add_argument('--models-dir', type=str, default='saved_models',
                       help='Directory for saving/loading models (default: saved_models)')
    
    # Information
    parser.add_argument('--summary', action='store_true',
                       help='Show model performance summary')
    
    args = parser.parse_args()
    
    # Initialize CLI
    cli = DiamondPredictorCLI()
    
    print("💎 Diamond Price Predictor CLI")
    print("=" * 50)
    
    # Load models if requested
    if args.load_models:
        cli.load_models(args.models_dir)
    
    # Complete analysis
    if args.complete_analysis:
        print("🚀 Running complete analysis...")
        data = cli.load_data(args.data_file, args.explore_data)
        if data is not None:
            cli.train_models(args.ml_only, args.dl_only, args.epochs, args.batch_size)
            cli.show_model_summary()
            if args.save_models:
                cli.save_models(args.models_dir)
        return
    
    # Load data
    if args.train or args.explore_data:
        data = cli.load_data(args.data_file, args.explore_data)
        if data is None:
            return
    
    # Train models
    if args.train:
        cli.train_models(args.ml_only, args.dl_only, args.epochs, args.batch_size)
        if args.save_models:
            cli.save_models(args.models_dir)
    
    # Show summary
    if args.summary:
        cli.show_model_summary()
    
    # Interactive mode
    if args.interactive:
        cli.interactive_prediction()
        return
    
    # Single prediction
    if args.predict:
        diamond_features = {
            'carat': args.carat,
            'cut': args.cut,
            'color': args.color,
            'clarity': args.clarity,
            'depth': args.depth,
            'table': args.table,
            'x': args.x,
            'y': args.y,
            'z': args.z
        }
        cli.predict_single(diamond_features, args.model_type)
    
    # Batch prediction
    if args.batch_predict:
        cli.predict_batch(args.batch_predict, args.output, args.model_type)
    
    # If no action specified, show help
    if not any([args.complete_analysis, args.train, args.predict, args.batch_predict, 
                args.interactive, args.summary, args.load_models]):
        parser.print_help()

if __name__ == "__main__":
    main()