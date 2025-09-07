"""
Diamond Price Predictor - Main Predictor Class
This module combines traditional ML and deep learning models for comprehensive diamond price prediction.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

from data_processor import DiamondDataProcessor
from ml_models import TraditionalMLModels
from deep_learning_models import DeepLearningModels

class DiamondPricePredictor:
    def __init__(self):
        self.data_processor = DiamondDataProcessor()
        self.ml_models = TraditionalMLModels()
        self.dl_models = DeepLearningModels()
        
        self.data = None
        self.X_train_scaled = None
        self.X_test_scaled = None
        self.y_train = None
        self.y_test = None
        self.X_train = None
        self.X_test = None
        
        self.ml_results = {}
        self.dl_results = {}
        self.comparison_results = {}
        
    def load_and_prepare_data(self, file_path=None, explore=True, visualize=True):
        """Load and prepare data for training"""
        print("=== LOADING AND PREPARING DATA ===")
        
        # Load data
        self.data = self.data_processor.load_data(file_path)
        
        # Explore data
        if explore:
            self.data_processor.explore_data(self.data)
        
        # Visualize data
        if visualize:
            self.data_processor.visualize_data(self.data)
        
        # Preprocess data
        (self.X_train_scaled, self.X_test_scaled, 
         self.y_train, self.y_test, 
         self.X_train, self.X_test) = self.data_processor.preprocess_data(self.data)
        
        print("Data preparation completed successfully!")
        return self.data
    
    def train_traditional_ml_models(self, optimize_best=True):
        """Train traditional machine learning models"""
        print("\n" + "="*50)
        print("TRAINING TRADITIONAL ML MODELS")
        print("="*50)
        
        # Initialize and train models
        self.ml_models.initialize_models()
        self.ml_results = self.ml_models.train_all_models(
            self.X_train_scaled, self.y_train, 
            self.X_test_scaled, self.y_test
        )
        
        # Optimize best model
        if optimize_best and self.ml_models.best_model_name:
            print(f"\nOptimizing {self.ml_models.best_model_name}...")
            try:
                optimized_model, best_params = self.ml_models.optimize_hyperparameters(
                    self.X_train_scaled, self.y_train, 
                    self.ml_models.best_model_name
                )
                
                # Evaluate optimized model
                y_test_pred_opt = optimized_model.predict(self.X_test_scaled)
                opt_r2 = r2_score(self.y_test, y_test_pred_opt)
                print(f"Optimized model R² score: {opt_r2:.4f}")
                
            except Exception as e:
                print(f"Optimization failed: {str(e)}")
        
        # Create ensemble
        ensemble_pred, ensemble_metrics = self.ml_models.create_ensemble_model(
            self.X_train_scaled, self.y_train,
            self.X_test_scaled, self.y_test
        )
        
        # Plot results
        self.ml_models.plot_results(self.y_test, self.data_processor.feature_names)
        
        return self.ml_results
    
    def train_deep_learning_models(self, epochs=100, batch_size=32):
        """Train deep learning models"""
        print("\n" + "="*50)
        print("TRAINING DEEP LEARNING MODELS")
        print("="*50)
        
        # Initialize and train models
        input_dim = self.X_train_scaled.shape[1]
        self.dl_models.initialize_models(input_dim)
        
        self.dl_results = self.dl_models.train_all_models(
            self.X_train_scaled, self.y_train,
            self.X_test_scaled, self.y_test,
            epochs=epochs, batch_size=batch_size
        )
        
        # Plot training history
        self.dl_models.plot_training_history()
        
        # Plot results
        self.dl_models.plot_results(self.y_test)
        
        return self.dl_results
    
    def compare_all_models(self):
        """Compare traditional ML and deep learning models"""
        print("\n" + "="*50)
        print("COMPARING ALL MODELS")
        print("="*50)
        
        # Combine results
        all_results = {}
        
        # Add ML results
        for name, results in self.ml_results.items():
            all_results[f"ML: {name}"] = results
        
        # Add DL results
        for name, results in self.dl_results.items():
            all_results[f"DL: {name}"] = results
        
        # Create comparison DataFrame
        comparison_data = []
        for name, results in all_results.items():
            comparison_data.append({
                'Model': name,
                'Type': 'Traditional ML' if name.startswith('ML:') else 'Deep Learning',
                'R² Score': results['test_r2'],
                'RMSE': results['test_rmse'],
                'MAE': results['test_mae']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('R² Score', ascending=False)
        
        print("\n=== MODEL COMPARISON RESULTS ===")
        print(comparison_df.to_string(index=False, float_format='%.4f'))
        
        # Find overall best model
        best_model_row = comparison_df.iloc[0]
        print(f"\nOverall Best Model: {best_model_row['Model']}")
        print(f"R² Score: {best_model_row['R² Score']:.4f}")
        print(f"RMSE: ${best_model_row['RMSE']:.2f}")
        print(f"MAE: ${best_model_row['MAE']:.2f}")
        
        # Visualization
        self.plot_model_comparison(comparison_df)
        
        self.comparison_results = comparison_df
        return comparison_df
    
    def plot_model_comparison(self, comparison_df):
        """Plot comprehensive model comparison"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. R² Score comparison
        colors = ['skyblue' if 'ML:' in model else 'lightcoral' for model in comparison_df['Model']]
        axes[0, 0].barh(comparison_df['Model'], comparison_df['R² Score'], color=colors, alpha=0.7)
        axes[0, 0].set_title('Model Comparison - R² Scores')
        axes[0, 0].set_xlabel('R² Score')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. RMSE comparison
        axes[0, 1].barh(comparison_df['Model'], comparison_df['RMSE'], color=colors, alpha=0.7)
        axes[0, 1].set_title('Model Comparison - RMSE')
        axes[0, 1].set_xlabel('RMSE ($)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. MAE comparison
        axes[1, 0].barh(comparison_df['Model'], comparison_df['MAE'], color=colors, alpha=0.7)
        axes[1, 0].set_title('Model Comparison - MAE')
        axes[1, 0].set_xlabel('MAE ($)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Model type distribution
        type_counts = comparison_df['Type'].value_counts()
        axes[1, 1].pie(type_counts.values, labels=type_counts.index, autopct='%1.1f%%',
                      colors=['skyblue', 'lightcoral'], alpha=0.7)
        axes[1, 1].set_title('Model Type Distribution')
        
        plt.tight_layout()
        plt.show()
        
        # Performance by type
        plt.figure(figsize=(12, 8))
        
        # Box plot of R² scores by model type
        plt.subplot(2, 2, 1)
        comparison_df.boxplot(column='R² Score', by='Type', ax=plt.gca())
        plt.title('R² Score Distribution by Model Type')
        plt.suptitle('')
        
        # Box plot of RMSE by model type
        plt.subplot(2, 2, 2)
        comparison_df.boxplot(column='RMSE', by='Type', ax=plt.gca())
        plt.title('RMSE Distribution by Model Type')
        plt.suptitle('')
        
        # Scatter plot: R² vs RMSE
        plt.subplot(2, 2, 3)
        ml_data = comparison_df[comparison_df['Type'] == 'Traditional ML']
        dl_data = comparison_df[comparison_df['Type'] == 'Deep Learning']
        
        plt.scatter(ml_data['R² Score'], ml_data['RMSE'], 
                   label='Traditional ML', alpha=0.7, s=100, color='skyblue')
        plt.scatter(dl_data['R² Score'], dl_data['RMSE'], 
                   label='Deep Learning', alpha=0.7, s=100, color='lightcoral')
        plt.xlabel('R² Score')
        plt.ylabel('RMSE ($)')
        plt.title('R² Score vs RMSE')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Model performance ranking
        plt.subplot(2, 2, 4)
        top_5 = comparison_df.head(5)
        plt.barh(range(len(top_5)), top_5['R² Score'], 
                color=['gold', 'silver', '#CD7F32', 'lightblue', 'lightgreen'])
        plt.yticks(range(len(top_5)), [model.replace('ML: ', '').replace('DL: ', '') 
                                      for model in top_5['Model']])
        plt.xlabel('R² Score')
        plt.title('Top 5 Models')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def predict_price(self, diamond_features, model_type='best', model_name=None):
        """Predict diamond price for new data"""
        # Prepare input data
        if isinstance(diamond_features, dict):
            input_df = pd.DataFrame([diamond_features])
        else:
            input_df = diamond_features.copy()
        
        # Process the input
        processed_input = self.data_processor.prepare_prediction_data(input_df)
        
        # Make prediction based on model type
        if model_type == 'best':
            # Use the overall best model
            if not hasattr(self, 'comparison_results') or self.comparison_results.empty:
                raise ValueError("No models trained yet. Train models first.")
            
            best_model_name = self.comparison_results.iloc[0]['Model']
            
            if best_model_name.startswith('ML:'):
                model_name_clean = best_model_name.replace('ML: ', '')
                prediction = self.ml_models.predict(processed_input, model_name_clean)
            else:
                model_name_clean = best_model_name.replace('DL: ', '')
                prediction = self.dl_models.predict(processed_input, model_name_clean)
        
        elif model_type == 'ml':
            prediction = self.ml_models.predict(processed_input, model_name)
        
        elif model_type == 'dl':
            prediction = self.dl_models.predict(processed_input, model_name)
        
        elif model_type == 'ensemble':
            # Create ensemble prediction from top models
            ml_pred = self.ml_models.predict(processed_input)
            dl_pred = self.dl_models.predict(processed_input)
            prediction = (ml_pred + dl_pred) / 2
        
        else:
            raise ValueError("Invalid model_type. Choose from: 'best', 'ml', 'dl', 'ensemble'")
        
        return prediction
    
    def save_all_models(self, directory="saved_models"):
        """Save all trained models"""
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        # Save data processor
        processor_path = os.path.join(directory, "data_processor.joblib")
        joblib.dump(self.data_processor, processor_path)
        print(f"Saved data processor to {processor_path}")
        
        # Save ML models
        ml_prefix = os.path.join(directory, "ml_model")
        self.ml_models.save_models(ml_prefix)
        
        # Save DL models
        dl_prefix = os.path.join(directory, "dl_model")
        self.dl_models.save_models(dl_prefix)
        
        # Save comparison results
        if hasattr(self, 'comparison_results') and not self.comparison_results.empty:
            comparison_path = os.path.join(directory, "model_comparison.csv")
            self.comparison_results.to_csv(comparison_path, index=False)
            print(f"Saved model comparison to {comparison_path}")
    
    def load_models(self, directory="saved_models"):
        """Load saved models"""
        try:
            # Load data processor
            processor_path = os.path.join(directory, "data_processor.joblib")
            self.data_processor = joblib.load(processor_path)
            print(f"Loaded data processor from {processor_path}")
            
            # Load comparison results
            comparison_path = os.path.join(directory, "model_comparison.csv")
            if os.path.exists(comparison_path):
                self.comparison_results = pd.read_csv(comparison_path)
                print(f"Loaded model comparison from {comparison_path}")
            
            print("Models loaded successfully!")
            
        except Exception as e:
            print(f"Error loading models: {str(e)}")
    
    def generate_prediction_report(self, diamond_features):
        """Generate a comprehensive prediction report"""
        print("\n" + "="*50)
        print("DIAMOND PRICE PREDICTION REPORT")
        print("="*50)
        
        # Display input features
        print("\nInput Diamond Characteristics:")
        if isinstance(diamond_features, dict):
            for key, value in diamond_features.items():
                print(f"  {key.title()}: {value}")
        
        # Get predictions from different model types
        predictions = {}
        
        try:
            predictions['Best Model'] = self.predict_price(diamond_features, 'best')
            predictions['ML Ensemble'] = self.predict_price(diamond_features, 'ml')
            predictions['DL Best'] = self.predict_price(diamond_features, 'dl')
            predictions['Combined Ensemble'] = self.predict_price(diamond_features, 'ensemble')
        except Exception as e:
            print(f"Error making predictions: {str(e)}")
            return
        
        # Display predictions
        print("\nPrice Predictions:")
        for model_type, pred in predictions.items():
            if isinstance(pred, np.ndarray):
                pred = pred[0]
            print(f"  {model_type}: ${pred:,.2f}")
        
        # Calculate statistics
        pred_values = [pred[0] if isinstance(pred, np.ndarray) else pred 
                      for pred in predictions.values()]
        
        print(f"\nPrediction Statistics:")
        print(f"  Mean: ${np.mean(pred_values):,.2f}")
        print(f"  Median: ${np.median(pred_values):,.2f}")
        print(f"  Std Dev: ${np.std(pred_values):,.2f}")
        print(f"  Min: ${np.min(pred_values):,.2f}")
        print(f"  Max: ${np.max(pred_values):,.2f}")
        
        # Confidence assessment
        std_dev = np.std(pred_values)
        mean_pred = np.mean(pred_values)
        confidence = "High" if std_dev / mean_pred < 0.1 else "Medium" if std_dev / mean_pred < 0.2 else "Low"
        
        print(f"\nPrediction Confidence: {confidence}")
        print(f"  (Based on prediction variance: {std_dev/mean_pred:.1%})")
        
        return predictions
    
    def get_feature_importance_analysis(self):
        """Analyze feature importance across models"""
        print("\n" + "="*50)
        print("FEATURE IMPORTANCE ANALYSIS")
        print("="*50)
        
        feature_names = self.data_processor.feature_names
        
        # Get feature importance from tree-based ML models
        ml_importances = {}
        for name, results in self.ml_results.items():
            model = results['model']
            if hasattr(model, 'feature_importances_'):
                ml_importances[name] = model.feature_importances_
        
        if ml_importances:
            # Create feature importance DataFrame
            importance_df = pd.DataFrame(ml_importances, index=feature_names)
            
            # Calculate mean importance
            importance_df['Mean'] = importance_df.mean(axis=1)
            importance_df = importance_df.sort_values('Mean', ascending=False)
            
            print("\nFeature Importance Rankings:")
            print(importance_df['Mean'].to_string(float_format='%.4f'))
            
            # Plot feature importance
            plt.figure(figsize=(12, 8))
            
            # Mean importance
            plt.subplot(1, 2, 1)
            plt.barh(importance_df.index, importance_df['Mean'], color='lightgreen', alpha=0.7)
            plt.title('Mean Feature Importance')
            plt.xlabel('Importance')
            plt.grid(True, alpha=0.3)
            
            # Importance by model
            plt.subplot(1, 2, 2)
            importance_df.drop('Mean', axis=1).plot(kind='barh', ax=plt.gca(), alpha=0.7)
            plt.title('Feature Importance by Model')
            plt.xlabel('Importance')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
            
            return importance_df
        else:
            print("No tree-based models available for feature importance analysis.")
            return None
    
    def run_complete_analysis(self, file_path=None, save_models=True):
        """Run complete diamond price prediction analysis"""
        print("="*60)
        print("DIAMOND PRICE PREDICTOR - COMPLETE ANALYSIS")
        print("="*60)
        
        # Step 1: Load and prepare data
        self.load_and_prepare_data(file_path)
        
        # Step 2: Train traditional ML models
        self.train_traditional_ml_models()
        
        # Step 3: Train deep learning models
        self.train_deep_learning_models()
        
        # Step 4: Compare all models
        comparison_results = self.compare_all_models()
        
        # Step 5: Feature importance analysis
        self.get_feature_importance_analysis()
        
        # Step 6: Save models
        if save_models:
            self.save_all_models()
        
        # Step 7: Print final summary
        print("\n" + "="*60)
        print("ANALYSIS COMPLETE - SUMMARY")
        print("="*60)
        
        print(self.ml_models.get_model_summary())
        print(self.dl_models.get_model_summary())
        
        best_model = comparison_results.iloc[0]
        print(f"\nOVERALL BEST MODEL: {best_model['Model']}")
        print(f"Performance: R² = {best_model['R² Score']:.4f}, RMSE = ${best_model['RMSE']:.2f}")
        
        return comparison_results