"""
Diamond Price Predictor - Deep Learning Models Module
This module contains neural network models for diamond price prediction.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks, optimizers
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

class DeepLearningModels:
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.results = {}
        self.history = {}
        
    def create_basic_nn(self, input_dim, name="Basic_NN"):
        """Create a basic neural network"""
        model = keras.Sequential([
            layers.Dense(128, activation='relu', input_shape=(input_dim,)),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(32, activation='relu'),
            layers.Dense(1)
        ], name=name)
        
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def create_deep_nn(self, input_dim, name="Deep_NN"):
        """Create a deeper neural network"""
        model = keras.Sequential([
            layers.Dense(256, activation='relu', input_shape=(input_dim,)),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            layers.Dense(64, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.2),
            
            layers.Dense(16, activation='relu'),
            layers.Dense(1)
        ], name=name)
        
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def create_wide_nn(self, input_dim, name="Wide_NN"):
        """Create a wide neural network"""
        model = keras.Sequential([
            layers.Dense(512, activation='relu', input_shape=(input_dim,)),
            layers.Dropout(0.4),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(1)
        ], name=name)
        
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def create_residual_nn(self, input_dim, name="Residual_NN"):
        """Create a neural network with residual connections"""
        inputs = layers.Input(shape=(input_dim,))
        
        # First block
        x = layers.Dense(128, activation='relu')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        
        # Residual block 1
        residual = x
        x = layers.Dense(128, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Add()([x, residual])  # Residual connection
        
        # Residual block 2
        x = layers.Dense(64, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
        residual2 = x
        x = layers.Dense(64, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Add()([x, residual2])  # Residual connection
        
        # Output
        x = layers.Dense(32, activation='relu')(x)
        x = layers.Dropout(0.1)(x)
        outputs = layers.Dense(1)(x)
        
        model = keras.Model(inputs=inputs, outputs=outputs, name=name)
        
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def create_ensemble_nn(self, input_dim, name="Ensemble_NN"):
        """Create an ensemble neural network with multiple branches"""
        inputs = layers.Input(shape=(input_dim,))
        
        # Branch 1: Deep and narrow
        branch1 = layers.Dense(64, activation='relu')(inputs)
        branch1 = layers.Dropout(0.2)(branch1)
        branch1 = layers.Dense(32, activation='relu')(branch1)
        branch1 = layers.Dense(16, activation='relu')(branch1)
        branch1 = layers.Dense(8, activation='relu')(branch1)
        
        # Branch 2: Wide and shallow
        branch2 = layers.Dense(256, activation='relu')(inputs)
        branch2 = layers.Dropout(0.3)(branch2)
        branch2 = layers.Dense(128, activation='relu')(branch2)
        
        # Branch 3: Medium
        branch3 = layers.Dense(128, activation='relu')(inputs)
        branch3 = layers.BatchNormalization()(branch3)
        branch3 = layers.Dropout(0.2)(branch3)
        branch3 = layers.Dense(64, activation='relu')(branch3)
        branch3 = layers.Dense(32, activation='relu')(branch3)
        
        # Combine branches
        combined = layers.Concatenate()([branch1, branch2, branch3])
        combined = layers.Dense(64, activation='relu')(combined)
        combined = layers.Dropout(0.2)(combined)
        combined = layers.Dense(32, activation='relu')(combined)
        outputs = layers.Dense(1)(combined)
        
        model = keras.Model(inputs=inputs, outputs=outputs, name=name)
        
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def initialize_models(self, input_dim):
        """Initialize all neural network models"""
        print("=== INITIALIZING DEEP LEARNING MODELS ===")
        
        self.models = {
            'Basic NN': self.create_basic_nn(input_dim),
            'Deep NN': self.create_deep_nn(input_dim),
            'Wide NN': self.create_wide_nn(input_dim),
            'Residual NN': self.create_residual_nn(input_dim),
            'Ensemble NN': self.create_ensemble_nn(input_dim)
        }
        
        print(f"Initialized {len(self.models)} neural network models")
        
        # Print model summaries
        for name, model in self.models.items():
            print(f"\n{name} Architecture:")
            print(f"  Total parameters: {model.count_params():,}")
            print(f"  Layers: {len(model.layers)}")
    
    def get_callbacks(self, model_name, patience=20):
        """Get training callbacks"""
        callbacks_list = [
            callbacks.EarlyStopping(
                monitor='val_loss',
                patience=patience,
                restore_best_weights=True,
                verbose=1
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=10,
                min_lr=1e-7,
                verbose=1
            ),
            callbacks.ModelCheckpoint(
                f'best_{model_name.replace(" ", "_").lower()}_model.h5',
                monitor='val_loss',
                save_best_only=True,
                verbose=0
            )
        ]
        
        return callbacks_list
    
    def train_all_models(self, X_train, y_train, X_test, y_test, epochs=100, batch_size=32, validation_split=0.2):
        """Train all neural network models"""
        print("=== TRAINING DEEP LEARNING MODELS ===")
        
        self.results = {}
        self.history = {}
        best_score = float('-inf')
        
        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            
            try:
                # Get callbacks
                model_callbacks = self.get_callbacks(name)
                
                # Train the model
                history = model.fit(
                    X_train, y_train,
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_split=validation_split,
                    callbacks=model_callbacks,
                    verbose=0
                )
                
                # Store training history
                self.history[name] = history
                
                # Make predictions
                y_train_pred = model.predict(X_train, verbose=0).flatten()
                y_test_pred = model.predict(X_test, verbose=0).flatten()
                
                # Calculate metrics
                train_r2 = r2_score(y_train, y_train_pred)
                test_r2 = r2_score(y_test, y_test_pred)
                train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
                test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
                train_mae = mean_absolute_error(y_train, y_train_pred)
                test_mae = mean_absolute_error(y_test, y_test_pred)
                
                # Store results
                self.results[name] = {
                    'model': model,
                    'train_r2': train_r2,
                    'test_r2': test_r2,
                    'train_rmse': train_rmse,
                    'test_rmse': test_rmse,
                    'train_mae': train_mae,
                    'test_mae': test_mae,
                    'predictions': y_test_pred,
                    'epochs_trained': len(history.history['loss'])
                }
                
                # Update best model
                if test_r2 > best_score:
                    best_score = test_r2
                    self.best_model = model
                    self.best_model_name = name
                
                print(f"  R² Score: {test_r2:.4f}")
                print(f"  RMSE: ${test_rmse:.2f}")
                print(f"  MAE: ${test_mae:.2f}")
                print(f"  Epochs trained: {len(history.history['loss'])}")
                
            except Exception as e:
                print(f"  Error training {name}: {str(e)}")
                continue
        
        print(f"\nBest model: {self.best_model_name} with R² = {best_score:.4f}")
        return self.results
    
    def cross_validate_model(self, model_creator, X, y, cv_folds=5, epochs=50, batch_size=32):
        """Perform cross-validation for a neural network model"""
        print("=== PERFORMING CROSS-VALIDATION ===")
        
        kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
        cv_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(kfold.split(X)):
            print(f"Training fold {fold + 1}/{cv_folds}...")
            
            X_train_fold, X_val_fold = X[train_idx], X[val_idx]
            y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
            
            # Create and train model
            model = model_creator(X.shape[1])
            
            # Early stopping for CV
            early_stopping = callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=0
            )
            
            model.fit(
                X_train_fold, y_train_fold,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(X_val_fold, y_val_fold),
                callbacks=[early_stopping],
                verbose=0
            )
            
            # Evaluate
            y_pred_fold = model.predict(X_val_fold, verbose=0).flatten()
            fold_score = r2_score(y_val_fold, y_pred_fold)
            cv_scores.append(fold_score)
            
            print(f"  Fold {fold + 1} R² Score: {fold_score:.4f}")
        
        mean_score = np.mean(cv_scores)
        std_score = np.std(cv_scores)
        
        print(f"\nCross-validation results:")
        print(f"  Mean R² Score: {mean_score:.4f} (±{std_score:.4f})")
        
        return cv_scores, mean_score, std_score
    
    def plot_training_history(self):
        """Plot training history for all models"""
        if not self.history:
            print("No training history available.")
            return
        
        n_models = len(self.history)
        fig, axes = plt.subplots(2, n_models, figsize=(5*n_models, 10))
        
        if n_models == 1:
            axes = axes.reshape(2, 1)
        
        for i, (name, history) in enumerate(self.history.items()):
            # Loss plot
            axes[0, i].plot(history.history['loss'], label='Training Loss', color='blue')
            axes[0, i].plot(history.history['val_loss'], label='Validation Loss', color='red')
            axes[0, i].set_title(f'{name} - Loss')
            axes[0, i].set_xlabel('Epoch')
            axes[0, i].set_ylabel('Loss')
            axes[0, i].legend()
            axes[0, i].grid(True, alpha=0.3)
            
            # MAE plot
            axes[1, i].plot(history.history['mae'], label='Training MAE', color='blue')
            axes[1, i].plot(history.history['val_mae'], label='Validation MAE', color='red')
            axes[1, i].set_title(f'{name} - MAE')
            axes[1, i].set_xlabel('Epoch')
            axes[1, i].set_ylabel('MAE')
            axes[1, i].legend()
            axes[1, i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_results(self, y_test):
        """Plot comprehensive results for neural networks"""
        if not self.results:
            print("No results to plot. Train models first.")
            return
        
        # 1. Model comparison
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # R² scores comparison
        model_names = list(self.results.keys())
        r2_scores = [self.results[name]['test_r2'] for name in model_names]
        
        axes[0, 0].bar(model_names, r2_scores, color='lightblue', alpha=0.7)
        axes[0, 0].set_title('Neural Network Comparison - R² Scores')
        axes[0, 0].set_ylabel('R² Score')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].grid(True, alpha=0.3)
        
        # RMSE comparison
        rmse_scores = [self.results[name]['test_rmse'] for name in model_names]
        axes[0, 1].bar(model_names, rmse_scores, color='lightcoral', alpha=0.7)
        axes[0, 1].set_title('Neural Network Comparison - RMSE')
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
        
        # 2. Training epochs comparison
        plt.figure(figsize=(10, 6))
        epochs_trained = [self.results[name]['epochs_trained'] for name in model_names]
        plt.bar(model_names, epochs_trained, color='gold', alpha=0.7)
        plt.title('Training Epochs by Model')
        plt.xlabel('Model')
        plt.ylabel('Epochs Trained')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def hyperparameter_tuning(self, X_train, y_train, model_type='basic', n_trials=20):
        """Perform hyperparameter tuning using Keras Tuner"""
        try:
            import keras_tuner as kt
        except ImportError:
            print("Keras Tuner not installed. Installing...")
            import subprocess
            subprocess.check_call(['pip', 'install', 'keras-tuner'])
            import keras_tuner as kt
        
        def build_model(hp):
            model = keras.Sequential()
            
            # Input layer
            model.add(layers.Dense(
                hp.Int('units_1', min_value=32, max_value=512, step=32),
                activation='relu',
                input_shape=(X_train.shape[1],)
            ))
            model.add(layers.Dropout(hp.Float('dropout_1', 0.0, 0.5, step=0.1)))
            
            # Hidden layers
            for i in range(hp.Int('num_layers', 2, 5)):
                model.add(layers.Dense(
                    hp.Int(f'units_{i+2}', min_value=16, max_value=256, step=16),
                    activation='relu'
                ))
                model.add(layers.Dropout(hp.Float(f'dropout_{i+2}', 0.0, 0.4, step=0.1)))
            
            # Output layer
            model.add(layers.Dense(1))
            
            # Compile
            model.compile(
                optimizer=optimizers.Adam(hp.Float('learning_rate', 1e-4, 1e-2, sampling='LOG')),
                loss='mse',
                metrics=['mae']
            )
            
            return model
        
        # Create tuner
        tuner = kt.RandomSearch(
            build_model,
            objective='val_loss',
            max_trials=n_trials,
            directory='diamond_tuning',
            project_name='diamond_price_nn'
        )
        
        # Search
        print(f"Starting hyperparameter tuning with {n_trials} trials...")
        tuner.search(
            X_train, y_train,
            epochs=50,
            validation_split=0.2,
            callbacks=[callbacks.EarlyStopping(patience=10)],
            verbose=0
        )
        
        # Get best model
        best_model = tuner.get_best_models(num_models=1)[0]
        best_hyperparameters = tuner.get_best_hyperparameters(num_trials=1)[0]
        
        print("Best hyperparameters:")
        for param, value in best_hyperparameters.values.items():
            print(f"  {param}: {value}")
        
        return best_model, best_hyperparameters
    
    def save_models(self, filepath_prefix="diamond_dl_models"):
        """Save trained neural network models"""
        for name, results in self.results.items():
            model_filename = f"{filepath_prefix}_{name.replace(' ', '_').lower()}.h5"
            results['model'].save(model_filename)
            print(f"Saved {name} to {model_filename}")
        
        # Save best model separately
        if self.best_model:
            best_model_filename = f"{filepath_prefix}_best_model.h5"
            self.best_model.save(best_model_filename)
            print(f"Saved best model ({self.best_model_name}) to {best_model_filename}")
    
    def load_model(self, filepath):
        """Load a saved neural network model"""
        try:
            model = keras.models.load_model(filepath)
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
        
        return model.predict(X, verbose=0).flatten()
    
    def get_model_summary(self):
        """Get summary of all trained neural network models"""
        if not self.results:
            return "No models trained yet."
        
        summary = "\n=== NEURAL NETWORK MODEL SUMMARY ===\n"
        summary += f"{'Model':<15} {'R²':<8} {'RMSE':<10} {'MAE':<10} {'Epochs':<8} {'Parameters':<12}\n"
        summary += "-" * 75 + "\n"
        
        for name, results in self.results.items():
            params = results['model'].count_params()
            summary += f"{name:<15} {results['test_r2']:<8.4f} {results['test_rmse']:<10.2f} "
            summary += f"{results['test_mae']:<10.2f} {results['epochs_trained']:<8} {params:<12,}\n"
        
        summary += f"\nBest Model: {self.best_model_name}\n"
        return summary