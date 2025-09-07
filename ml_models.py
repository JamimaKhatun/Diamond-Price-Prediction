"""
Diamond Price Predictor - Machine Learning Models Module
This module contains traditional ML models for diamond price prediction.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score, GridSearchCV
import optuna
import warnings
warnings.filterwarnings('ignore')

class TraditionalMLModels:
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.results = {}
        
    def initialize_models(self):
        """Initialize all traditional ML models"""
        self.models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0),
            'Lasso Regression': Lasso(alpha=1.0),
            'Random Forest': RandomForestRegressor(
                n_estimators=100,
                random_state=42,
                n_jobs=-1
            ),
            'Gradient Boosting': GradientBoostingRegressor(
                n_estimators=100,
                random_state=42
            ),
            'XGBoost': xgb.XGBRegressor(
                n_estimators=100,
                random_state=42,
                n_jobs=-1
            ),
            'Support Vector Regression': SVR(
                kernel='rbf',
                C=1.0,
                gamma='scale'
            )
        }
        print("Traditional ML models initialized successfully!")
        
    def train_all_models(self, X_train, y_train, X_test, y_test):
        """Train all models and evaluate performance"""
        print("=== TRAINING TRADITIONAL ML MODELS ===")
        
        self.results = {}
        best_score = float('-inf')
        
        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            
            try:
                # Train the model
                model.fit(X_train, y_train)
                
                # Make predictions
                y_train_pred = model.predict(X_train)
                y_test_pred = model.predict(X_test)
                
                # Calculate metrics
                train_r2 = r2_score(y_train, y_train_pred)
                test_r2 = r2_score(y_test, y_test_pred)
                train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
                test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
                train_mae = mean_absolute_error(y_train, y_train_pred)
                test_mae = mean_absolute_error(y_test, y_test_pred)
                
                # Cross-validation score
                cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
                cv_mean = cv_scores.mean()
                cv_std = cv_scores.std()
                
                # Store results
                self.results[name] = {
                    'model': model,
                    'train_r2': train_r2,
                    'test_r2': test_r2,
                    'train_rmse': train_rmse,
                    'test_rmse': test_rmse,
                    'train_mae': train_mae,
                    'test_mae': test_mae,
                    'cv_mean': cv_mean,
                    'cv_std': cv_std,
                    'predictions': y_test_pred
                }
                
                # Update best model
                if test_r2 > best_score:
                    best_score = test_r2
                    self.best_model = model
                    self.best_model_name = name
                
                print(f"  R² Score: {test_r2:.4f}")
                print(f"  RMSE: {test_rmse:.2f}")
                print(f"  MAE: {test_mae:.2f}")
                print(f"  CV Score: {cv_mean:.4f} (±{cv_std:.4f})")
                
            except Exception as e:
                print(f"  Error training {name}: {str(e)}")
                continue
        
        print(f"\nBest model: {self.best_model_name} with R² = {best_score:.4f}")
        return self.results
    
    def optimize_hyperparameters(self, X_train, y_train, model_name='Random Forest', n_trials=50):
        """Optimize hyperparameters using Optuna"""
        print(f"=== OPTIMIZING {model_name.upper()} HYPERPARAMETERS ===")
        
        def objective(trial):
            if model_name == 'Random Forest':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'max_depth': trial.suggest_int('max_depth', 3, 20),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                    'random_state': 42,
                    'n_jobs': -1
                }
                model = RandomForestRegressor(**params)
                
            elif model_name == 'XGBoost':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'random_state': 42,
                    'n_jobs': -1
                }
                model = xgb.XGBRegressor(**params)
                
            elif model_name == 'Gradient Boosting':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'random_state': 42
                }
                model = GradientBoostingRegressor(**params)
            
            else:
                raise ValueError(f"Optimization not implemented for {model_name}")
            
            # Cross-validation
            scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
            return scores.mean()
        
        # Create study and optimize
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        print(f"Best parameters: {study.best_params}")
        print(f"Best CV score: {study.best_value:.4f}")
        
        # Train model with best parameters
        if model_name == 'Random Forest':
            optimized_model = RandomForestRegressor(**study.best_params)
        elif model_name == 'XGBoost':
            optimized_model = xgb.XGBRegressor(**study.best_params)
        elif model_name == 'Gradient Boosting':
            optimized_model = GradientBoostingRegressor(**study.best_params)
        
        optimized_model.fit(X_train, y_train)
        self.models[f'{model_name} (Optimized)'] = optimized_model
        
        return optimized_model, study.best_params
    
    def evaluate_model_performance(self, y_true, y_pred, model_name="Model"):
        """Evaluate model performance with detailed metrics"""
        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        
        # Mean Absolute Percentage Error
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        # Explained Variance Score
        from sklearn.metrics import explained_variance_score
        evs = explained_variance_score(y_true, y_pred)
        
        print(f"\n=== {model_name.upper()} PERFORMANCE ===")
        print(f"R² Score: {r2:.4f}")
        print(f"RMSE: ${rmse:.2f}")
        print(f"MAE: ${mae:.2f}")
        print(f"MAPE: {mape:.2f}%")
        print(f"Explained Variance: {evs:.4f}")
        
        return {
            'r2': r2,
            'rmse': rmse,
            'mae': mae,
            'mape': mape,
            'explained_variance': evs
        }
    
    def plot_results(self, y_test, feature_names=None):
        """Plot comprehensive results"""
        if not self.results:
            print("No results to plot. Train models first.")
            return
        
        # 1. Model comparison
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # R² scores comparison
        model_names = list(self.results.keys())
        r2_scores = [self.results[name]['test_r2'] for name in model_names]
        
        axes[0, 0].bar(model_names, r2_scores, color='skyblue', alpha=0.7)
        axes[0, 0].set_title('Model Comparison - R² Scores')
        axes[0, 0].set_ylabel('R² Score')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].grid(True, alpha=0.3)
        
        # RMSE comparison
        rmse_scores = [self.results[name]['test_rmse'] for name in model_names]
        axes[0, 1].bar(model_names, rmse_scores, color='coral', alpha=0.7)
        axes[0, 1].set_title('Model Comparison - RMSE')
        axes[0, 1].set_ylabel('RMSE ($)')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(True, alpha=0.3)
        
        # Best model predictions vs actual
        best_predictions = self.results[self.best_model_name]['predictions']
        axes[1, 0].scatter(y_test, best_predictions, alpha=0.6, color='green')
        axes[1, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        axes[1, 0].set_title(f'{self.best_model_name} - Predictions vs Actual')
        axes[1, 0].set_xlabel('Actual Price ($)')
        axes[1, 0].set_ylabel('Predicted Price ($)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Residuals plot
        residuals = y_test - best_predictions
        axes[1, 1].scatter(best_predictions, residuals, alpha=0.6, color='purple')
        axes[1, 1].axhline(y=0, color='r', linestyle='--')
        axes[1, 1].set_title(f'{self.best_model_name} - Residuals Plot')
        axes[1, 1].set_xlabel('Predicted Price ($)')
        axes[1, 1].set_ylabel('Residuals ($)')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # 2. Feature importance (for tree-based models)
        if hasattr(self.best_model, 'feature_importances_') and feature_names:
            plt.figure(figsize=(10, 6))
            importances = self.best_model.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            plt.bar(range(len(importances)), importances[indices], color='lightgreen', alpha=0.7)
            plt.title(f'{self.best_model_name} - Feature Importance')
            plt.xlabel('Features')
            plt.ylabel('Importance')
            plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
    
    def create_ensemble_model(self, X_train, y_train, X_test, y_test):
        """Create an ensemble model combining top performers"""
        print("=== CREATING ENSEMBLE MODEL ===")
        
        # Select top 3 models based on test R² score
        sorted_models = sorted(self.results.items(), 
                             key=lambda x: x[1]['test_r2'], 
                             reverse=True)[:3]
        
        print("Top 3 models for ensemble:")
        for name, results in sorted_models:
            print(f"  {name}: R² = {results['test_r2']:.4f}")
        
        # Create ensemble predictions
        ensemble_train_pred = np.zeros(len(y_train))
        ensemble_test_pred = np.zeros(len(y_test))
        
        weights = []
        for name, results in sorted_models:
            weight = results['test_r2']  # Use R² as weight
            weights.append(weight)
            
            model = results['model']
            ensemble_train_pred += weight * model.predict(X_train)
            ensemble_test_pred += weight * model.predict(X_test)
        
        # Normalize by total weight
        total_weight = sum(weights)
        ensemble_train_pred /= total_weight
        ensemble_test_pred /= total_weight
        
        # Evaluate ensemble
        ensemble_metrics = self.evaluate_model_performance(
            y_test, ensemble_test_pred, "Ensemble Model"
        )
        
        # Store ensemble results
        self.results['Ensemble'] = {
            'model': 'ensemble',
            'train_r2': r2_score(y_train, ensemble_train_pred),
            'test_r2': ensemble_metrics['r2'],
            'train_rmse': np.sqrt(mean_squared_error(y_train, ensemble_train_pred)),
            'test_rmse': ensemble_metrics['rmse'],
            'train_mae': mean_absolute_error(y_train, ensemble_train_pred),
            'test_mae': ensemble_metrics['mae'],
            'predictions': ensemble_test_pred,
            'weights': dict(zip([name for name, _ in sorted_models], weights))
        }
        
        return ensemble_test_pred, ensemble_metrics
    
    def save_models(self, filepath_prefix="diamond_ml_models"):
        """Save trained models"""
        for name, results in self.results.items():
            if results['model'] != 'ensemble':
                model_filename = f"{filepath_prefix}_{name.replace(' ', '_').lower()}.joblib"
                joblib.dump(results['model'], model_filename)
                print(f"Saved {name} to {model_filename}")
        
        # Save best model separately
        if self.best_model:
            best_model_filename = f"{filepath_prefix}_best_model.joblib"
            joblib.dump(self.best_model, best_model_filename)
            print(f"Saved best model ({self.best_model_name}) to {best_model_filename}")
    
    def load_model(self, filepath):
        """Load a saved model"""
        try:
            model = joblib.load(filepath)
            print(f"Model loaded successfully from {filepath}")
            return model
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return None
    
    def predict(self, X, model_name=None):
        """Make predictions using specified model or best model"""
        if model_name and model_name in self.results:
            model = self.results[model_name]['model']
        elif self.best_model:
            model = self.best_model
        else:
            raise ValueError("No trained model available for prediction")
        
        return model.predict(X)
    
    def get_model_summary(self):
        """Get summary of all trained models"""
        if not self.results:
            return "No models trained yet."
        
        summary = "\n=== MODEL SUMMARY ===\n"
        summary += f"{'Model':<25} {'R²':<8} {'RMSE':<10} {'MAE':<10} {'CV Score':<12}\n"
        summary += "-" * 70 + "\n"
        
        for name, results in self.results.items():
            summary += f"{name:<25} {results['test_r2']:<8.4f} {results['test_rmse']:<10.2f} "
            summary += f"{results['test_mae']:<10.2f} "
            if 'cv_mean' in results:
                summary += f"{results['cv_mean']:<8.4f}±{results['cv_std']:<.3f}\n"
            else:
                summary += "N/A\n"
        
        summary += f"\nBest Model: {self.best_model_name}\n"
        return summary