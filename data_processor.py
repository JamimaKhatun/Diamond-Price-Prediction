"""
Diamond Price Predictor - Data Processing Module
This module handles data loading, preprocessing, and feature engineering for diamond price prediction.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class DiamondDataProcessor:
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_names = []
        self.target_name = 'price'
        
    def load_data(self, file_path=None):
        """Load diamond dataset"""
        if file_path is None:
            # Create a more comprehensive synthetic dataset if no file provided
            return self._create_synthetic_dataset()
        
        try:
            df = pd.read_csv(file_path)
            print(f"Dataset loaded successfully with shape: {df.shape}")
            return df
        except Exception as e:
            print(f"Error loading data: {e}")
            print("Creating synthetic dataset instead...")
            return self._create_synthetic_dataset()
    
    def _create_synthetic_dataset(self, n_samples=5000):
        """Create a comprehensive synthetic diamond dataset"""
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
        
        # Calculate dimensions based on carat (approximate)
        x = (carat * 6.5) ** (1/3) + np.random.normal(0, 0.1, n_samples)
        y = x + np.random.normal(0, 0.05, n_samples)
        z = x * depth / 100 + np.random.normal(0, 0.05, n_samples)
        
        # Ensure positive dimensions
        x = np.clip(x, 3.0, 12.0)
        y = np.clip(y, 3.0, 12.0)
        z = np.clip(z, 1.5, 8.0)
        
        # Create price based on features (realistic pricing model)
        cut_multiplier = {'Fair': 0.8, 'Good': 0.9, 'Very Good': 1.0, 'Premium': 1.1, 'Ideal': 1.2}
        color_multiplier = {'D': 1.3, 'E': 1.2, 'F': 1.1, 'G': 1.0, 'H': 0.9, 'I': 0.8, 'J': 0.7}
        clarity_multiplier = {'FL': 2.0, 'IF': 1.8, 'VVS1': 1.6, 'VVS2': 1.4, 'VS1': 1.2, 'VS2': 1.0, 'SI1': 0.8, 'SI2': 0.6, 'I1': 0.4}
        
        base_price = 3000 * (carat ** 2.5)  # Exponential relationship with carat
        
        price = base_price * np.array([cut_multiplier[c] for c in cut]) * \
                np.array([color_multiplier[c] for c in color]) * \
                np.array([clarity_multiplier[c] for c in clarity])
        
        # Add some noise and ensure reasonable price range
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
        
        print(f"Synthetic dataset created with shape: {df.shape}")
        return df
    
    def explore_data(self, df):
        """Perform exploratory data analysis"""
        print("=== DIAMOND DATASET EXPLORATION ===")
        print(f"Dataset shape: {df.shape}")
        print(f"\nDataset info:")
        print(df.info())
        print(f"\nMissing values:")
        print(df.isnull().sum())
        print(f"\nBasic statistics:")
        print(df.describe())
        
        # Categorical features analysis
        categorical_features = ['cut', 'color', 'clarity']
        print(f"\nCategorical features distribution:")
        for feature in categorical_features:
            if feature in df.columns:
                print(f"\n{feature}:")
                print(df[feature].value_counts())
        
        return df
    
    def visualize_data(self, df):
        """Create comprehensive visualizations"""
        # Set style
        plt.style.use('seaborn-v0_8')
        
        # 1. Price distribution
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Price histogram
        axes[0, 0].hist(df['price'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].set_title('Price Distribution')
        axes[0, 0].set_xlabel('Price ($)')
        axes[0, 0].set_ylabel('Frequency')
        
        # Price vs Carat
        axes[0, 1].scatter(df['carat'], df['price'], alpha=0.6, color='coral')
        axes[0, 1].set_title('Price vs Carat')
        axes[0, 1].set_xlabel('Carat')
        axes[0, 1].set_ylabel('Price ($)')
        
        # Cut distribution
        cut_counts = df['cut'].value_counts()
        axes[1, 0].bar(cut_counts.index, cut_counts.values, color='lightgreen')
        axes[1, 0].set_title('Cut Distribution')
        axes[1, 0].set_xlabel('Cut')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Color distribution
        color_counts = df['color'].value_counts()
        axes[1, 1].bar(color_counts.index, color_counts.values, color='gold')
        axes[1, 1].set_title('Color Distribution')
        axes[1, 1].set_xlabel('Color')
        axes[1, 1].set_ylabel('Count')
        
        plt.tight_layout()
        plt.show()
        
        # 2. Correlation heatmap
        plt.figure(figsize=(12, 8))
        
        # Create a copy for correlation analysis
        df_corr = df.copy()
        
        # Encode categorical variables for correlation
        for col in ['cut', 'color', 'clarity']:
            if col in df_corr.columns:
                le = LabelEncoder()
                df_corr[col + '_encoded'] = le.fit_transform(df_corr[col])
        
        # Select numeric columns for correlation
        numeric_cols = df_corr.select_dtypes(include=[np.number]).columns
        correlation_matrix = df_corr[numeric_cols].corr()
        
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                   square=True, linewidths=0.5)
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        plt.show()
        
        # 3. Box plots for categorical features
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        categorical_features = ['cut', 'color', 'clarity']
        for i, feature in enumerate(categorical_features):
            if feature in df.columns:
                df.boxplot(column='price', by=feature, ax=axes[i])
                axes[i].set_title(f'Price by {feature.title()}')
                axes[i].set_xlabel(feature.title())
                axes[i].set_ylabel('Price ($)')
        
        plt.suptitle('')  # Remove default title
        plt.tight_layout()
        plt.show()
    
    def preprocess_data(self, df, test_size=0.2, random_state=42):
        """Preprocess the data for machine learning"""
        print("=== PREPROCESSING DATA ===")
        
        # Handle missing values
        df = df.dropna()
        
        # Remove outliers (optional)
        df = self._remove_outliers(df)
        
        # Feature engineering
        df = self._engineer_features(df)
        
        # Separate features and target
        X = df.drop(self.target_name, axis=1)
        y = df[self.target_name]
        
        # Encode categorical variables
        categorical_features = ['cut', 'color', 'clarity']
        for feature in categorical_features:
            if feature in X.columns:
                le = LabelEncoder()
                X[feature] = le.fit_transform(X[feature])
                self.label_encoders[feature] = le
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print(f"Training set shape: {X_train_scaled.shape}")
        print(f"Test set shape: {X_test_scaled.shape}")
        print(f"Features: {self.feature_names}")
        
        return X_train_scaled, X_test_scaled, y_train, y_test, X_train, X_test
    
    def _remove_outliers(self, df, method='iqr'):
        """Remove outliers from the dataset"""
        if method == 'iqr':
            Q1 = df['price'].quantile(0.25)
            Q3 = df['price'].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            initial_shape = df.shape[0]
            df = df[(df['price'] >= lower_bound) & (df['price'] <= upper_bound)]
            final_shape = df.shape[0]
            
            print(f"Removed {initial_shape - final_shape} outliers ({((initial_shape - final_shape) / initial_shape * 100):.2f}%)")
        
        return df
    
    def _engineer_features(self, df):
        """Create additional features"""
        # Volume
        if all(col in df.columns for col in ['x', 'y', 'z']):
            df['volume'] = df['x'] * df['y'] * df['z']
        
        # Price per carat
        df['price_per_carat'] = df['price'] / df['carat']
        
        # Carat categories
        df['carat_category'] = pd.cut(df['carat'], 
                                     bins=[0, 0.5, 1.0, 1.5, 2.0, float('inf')],
                                     labels=['Small', 'Medium', 'Large', 'Very Large', 'Exceptional'])
        
        # Drop the price_per_carat as it's derived from target
        if 'price_per_carat' in df.columns:
            df = df.drop('price_per_carat', axis=1)
        
        # Encode the new categorical feature
        if 'carat_category' in df.columns:
            le = LabelEncoder()
            df['carat_category'] = le.fit_transform(df['carat_category'])
            self.label_encoders['carat_category'] = le
        
        return df
    
    def prepare_prediction_data(self, input_data):
        """Prepare new data for prediction"""
        # Convert to DataFrame if it's a dictionary
        if isinstance(input_data, dict):
            df = pd.DataFrame([input_data])
        else:
            df = input_data.copy()
        
        # Engineer features
        df = self._engineer_features(df)
        
        # Encode categorical variables
        for feature, encoder in self.label_encoders.items():
            if feature in df.columns:
                df[feature] = encoder.transform(df[feature])
        
        # Ensure all features are present
        for feature in self.feature_names:
            if feature not in df.columns:
                df[feature] = 0  # Default value for missing features
        
        # Select and order features
        df = df[self.feature_names]
        
        # Scale features
        df_scaled = self.scaler.transform(df)
        
        return df_scaled