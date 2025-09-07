"""
Diamond Price Predictor - Streamlit Web Interface
A comprehensive web application for diamond price prediction using ML and Deep Learning.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from diamond_predictor import DiamondPricePredictor
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Diamond Price Predictor",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1f77b4;
    }
    .prediction-result {
        font-size: 2rem;
        font-weight: bold;
        color: #2e8b57;
        text-align: center;
        padding: 1rem;
        background-color: #f0f8ff;
        border-radius: 0.5rem;
        border: 2px solid #2e8b57;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'predictor' not in st.session_state:
    st.session_state.predictor = None
if 'models_trained' not in st.session_state:
    st.session_state.models_trained = False
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False

def main():
    # Header
    st.markdown('<h1 class="main-header">💎 Diamond Price Predictor</h1>', unsafe_allow_html=True)
    st.markdown("### Advanced ML & Deep Learning Diamond Valuation System")
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["🏠 Home", "📊 Data Analysis", "🤖 Model Training", "💰 Price Prediction", "📈 Model Comparison", "ℹ️ About"]
    )
    
    if page == "🏠 Home":
        show_home_page()
    elif page == "📊 Data Analysis":
        show_data_analysis_page()
    elif page == "🤖 Model Training":
        show_model_training_page()
    elif page == "💰 Price Prediction":
        show_prediction_page()
    elif page == "📈 Model Comparison":
        show_comparison_page()
    elif page == "ℹ️ About":
        show_about_page()

def show_home_page():
    st.markdown('<h2 class="sub-header">Welcome to Diamond Price Predictor</h2>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>🤖 Advanced ML Models</h3>
            <p>Utilizes multiple machine learning algorithms including Random Forest, XGBoost, and Neural Networks</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>🧠 Deep Learning</h3>
            <p>Sophisticated neural network architectures with residual connections and ensemble methods</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>📊 Comprehensive Analysis</h3>
            <p>Detailed data exploration, feature importance analysis, and model performance comparison</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Quick start guide
    st.markdown('<h3 class="sub-header">Quick Start Guide</h3>', unsafe_allow_html=True)
    
    steps = [
        "📊 **Data Analysis**: Explore the diamond dataset and understand feature distributions",
        "🤖 **Model Training**: Train both traditional ML and deep learning models",
        "💰 **Price Prediction**: Use trained models to predict diamond prices",
        "📈 **Model Comparison**: Compare performance across different model types"
    ]
    
    for i, step in enumerate(steps, 1):
        st.markdown(f"{i}. {step}")
    
    # Sample diamond data
    st.markdown('<h3 class="sub-header">Sample Diamond Data</h3>', unsafe_allow_html=True)
    
    sample_data = {
        'Carat': [0.23, 0.21, 0.23, 0.29, 0.31],
        'Cut': ['Ideal', 'Premium', 'Good', 'Premium', 'Good'],
        'Color': ['E', 'E', 'E', 'I', 'J'],
        'Clarity': ['SI2', 'SI1', 'VS1', 'VS2', 'SI2'],
        'Depth': [61.5, 59.8, 56.9, 62.4, 63.3],
        'Table': [55.0, 61.0, 65.0, 58.0, 58.0],
        'Price': [326, 326, 327, 334, 335]
    }
    
    st.dataframe(pd.DataFrame(sample_data))

def show_data_analysis_page():
    st.markdown('<h2 class="sub-header">Data Analysis & Exploration</h2>', unsafe_allow_html=True)
    
    # Initialize predictor if not exists
    if st.session_state.predictor is None:
        st.session_state.predictor = DiamondPricePredictor()
    
    # Data loading section
    st.markdown("### 📁 Data Loading")
    
    data_source = st.radio(
        "Choose data source:",
        ["Use synthetic dataset", "Upload CSV file"]
    )
    
    if data_source == "Upload CSV file":
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        if uploaded_file is not None:
            try:
                data = pd.read_csv(uploaded_file)
                st.success("File uploaded successfully!")
                st.dataframe(data.head())
            except Exception as e:
                st.error(f"Error loading file: {str(e)}")
                return
    else:
        if st.button("Load Synthetic Dataset"):
            with st.spinner("Loading and preparing data..."):
                try:
                    data = st.session_state.predictor.load_and_prepare_data(
                        file_path=None, explore=False, visualize=False
                    )
                    st.session_state.data_loaded = True
                    st.success("Synthetic dataset loaded successfully!")
                except Exception as e:
                    st.error(f"Error loading data: {str(e)}")
                    return
    
    # Data exploration
    if st.session_state.data_loaded and st.session_state.predictor.data is not None:
        data = st.session_state.predictor.data
        
        st.markdown("### 📊 Dataset Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Samples", len(data))
        with col2:
            st.metric("Features", len(data.columns) - 1)
        with col3:
            st.metric("Average Price", f"${data['price'].mean():.0f}")
        with col4:
            st.metric("Price Range", f"${data['price'].min():.0f} - ${data['price'].max():.0f}")
        
        # Data distribution plots
        st.markdown("### 📈 Data Distributions")
        
        # Price distribution
        fig_price = px.histogram(data, x='price', nbins=50, title='Price Distribution')
        fig_price.update_layout(xaxis_title='Price ($)', yaxis_title='Frequency')
        st.plotly_chart(fig_price, use_container_width=True)
        
        # Feature distributions
        col1, col2 = st.columns(2)
        
        with col1:
            # Carat vs Price
            fig_carat = px.scatter(data, x='carat', y='price', title='Price vs Carat')
            fig_carat.update_layout(xaxis_title='Carat', yaxis_title='Price ($)')
            st.plotly_chart(fig_carat, use_container_width=True)
        
        with col2:
            # Cut distribution
            cut_counts = data['cut'].value_counts()
            fig_cut = px.bar(x=cut_counts.index, y=cut_counts.values, title='Cut Distribution')
            fig_cut.update_layout(xaxis_title='Cut', yaxis_title='Count')
            st.plotly_chart(fig_cut, use_container_width=True)
        
        # Correlation heatmap
        st.markdown("### 🔥 Feature Correlations")
        
        # Prepare data for correlation
        data_corr = data.copy()
        categorical_cols = ['cut', 'color', 'clarity']
        
        for col in categorical_cols:
            if col in data_corr.columns:
                data_corr[col] = pd.Categorical(data_corr[col]).codes
        
        numeric_cols = data_corr.select_dtypes(include=[np.number]).columns
        corr_matrix = data_corr[numeric_cols].corr()
        
        fig_corr = px.imshow(corr_matrix, text_auto=True, aspect="auto", title='Feature Correlation Matrix')
        st.plotly_chart(fig_corr, use_container_width=True)
        
        # Statistical summary
        st.markdown("### 📋 Statistical Summary")
        st.dataframe(data.describe())

def show_model_training_page():
    st.markdown('<h2 class="sub-header">Model Training & Optimization</h2>', unsafe_allow_html=True)
    
    if not st.session_state.data_loaded:
        st.warning("Please load data first in the Data Analysis page.")
        return
    
    # Training options
    st.markdown("### ⚙️ Training Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        train_ml = st.checkbox("Train Traditional ML Models", value=True)
        optimize_ml = st.checkbox("Optimize Best ML Model", value=True)
    
    with col2:
        train_dl = st.checkbox("Train Deep Learning Models", value=True)
        dl_epochs = st.slider("DL Training Epochs", 10, 200, 50)
        dl_batch_size = st.selectbox("DL Batch Size", [16, 32, 64, 128], index=1)
    
    # Training button
    if st.button("🚀 Start Training", type="primary"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Train ML models
            if train_ml:
                status_text.text("Training Traditional ML Models...")
                progress_bar.progress(25)
                
                ml_results = st.session_state.predictor.train_traditional_ml_models(optimize_best=optimize_ml)
                st.success("Traditional ML models trained successfully!")
                
                # Display ML results
                st.markdown("#### 📊 ML Model Results")
                ml_summary = []
                for name, results in ml_results.items():
                    ml_summary.append({
                        'Model': name,
                        'R² Score': results['test_r2'],
                        'RMSE': results['test_rmse'],
                        'MAE': results['test_mae']
                    })
                
                ml_df = pd.DataFrame(ml_summary).sort_values('R² Score', ascending=False)
                st.dataframe(ml_df, use_container_width=True)
            
            progress_bar.progress(50)
            
            # Train DL models
            if train_dl:
                status_text.text("Training Deep Learning Models...")
                progress_bar.progress(75)
                
                dl_results = st.session_state.predictor.train_deep_learning_models(
                    epochs=dl_epochs, batch_size=dl_batch_size
                )
                st.success("Deep Learning models trained successfully!")
                
                # Display DL results
                st.markdown("#### 🧠 Deep Learning Model Results")
                dl_summary = []
                for name, results in dl_results.items():
                    dl_summary.append({
                        'Model': name,
                        'R² Score': results['test_r2'],
                        'RMSE': results['test_rmse'],
                        'MAE': results['test_mae'],
                        'Epochs': results['epochs_trained']
                    })
                
                dl_df = pd.DataFrame(dl_summary).sort_values('R² Score', ascending=False)
                st.dataframe(dl_df, use_container_width=True)
            
            progress_bar.progress(100)
            status_text.text("Training completed!")
            st.session_state.models_trained = True
            
            # Model comparison
            if train_ml and train_dl:
                st.markdown("#### 🏆 Model Comparison")
                comparison_results = st.session_state.predictor.compare_all_models()
                
                # Display top 5 models
                top_models = comparison_results.head()
                st.dataframe(top_models, use_container_width=True)
                
                # Best model highlight
                best_model = comparison_results.iloc[0]
                st.markdown(f"""
                <div class="prediction-result">
                    🏆 Best Model: {best_model['Model']}<br>
                    R² Score: {best_model['R² Score']:.4f} | RMSE: ${best_model['RMSE']:.2f}
                </div>
                """, unsafe_allow_html=True)
        
        except Exception as e:
            st.error(f"Training failed: {str(e)}")
            progress_bar.progress(0)
            status_text.text("Training failed!")

def show_prediction_page():
    st.markdown('<h2 class="sub-header">Diamond Price Prediction</h2>', unsafe_allow_html=True)
    
    if not st.session_state.models_trained:
        st.warning("Please train models first in the Model Training page.")
        return
    
    # Input form
    st.markdown("### 💎 Enter Diamond Characteristics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        carat = st.number_input("Carat", min_value=0.1, max_value=5.0, value=1.0, step=0.01)
        cut = st.selectbox("Cut", ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal'])
        color = st.selectbox("Color", ['D', 'E', 'F', 'G', 'H', 'I', 'J'])
    
    with col2:
        clarity = st.selectbox("Clarity", ['FL', 'IF', 'VVS1', 'VVS2', 'VS1', 'VS2', 'SI1', 'SI2', 'I1'])
        depth = st.number_input("Depth (%)", min_value=50.0, max_value=80.0, value=61.5, step=0.1)
        table = st.number_input("Table (%)", min_value=40.0, max_value=80.0, value=57.0, step=0.1)
    
    with col3:
        x = st.number_input("Length (mm)", min_value=1.0, max_value=15.0, value=5.0, step=0.01)
        y = st.number_input("Width (mm)", min_value=1.0, max_value=15.0, value=5.0, step=0.01)
        z = st.number_input("Height (mm)", min_value=1.0, max_value=10.0, value=3.0, step=0.01)
    
    # Prediction button
    if st.button("💰 Predict Price", type="primary"):
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
        
        try:
            with st.spinner("Calculating price predictions..."):
                predictions = st.session_state.predictor.generate_prediction_report(diamond_features)
            
            # Display predictions
            st.markdown("### 🎯 Price Predictions")
            
            pred_cols = st.columns(len(predictions))
            for i, (model_type, pred) in enumerate(predictions.items()):
                with pred_cols[i]:
                    if isinstance(pred, np.ndarray):
                        pred_value = pred[0]
                    else:
                        pred_value = pred
                    
                    st.metric(
                        label=model_type,
                        value=f"${pred_value:,.2f}"
                    )
            
            # Prediction statistics
            pred_values = [pred[0] if isinstance(pred, np.ndarray) else pred 
                          for pred in predictions.values()]
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Average Prediction", f"${np.mean(pred_values):,.2f}")
            with col2:
                st.metric("Prediction Range", f"${np.max(pred_values) - np.min(pred_values):,.2f}")
            with col3:
                confidence = "High" if np.std(pred_values) / np.mean(pred_values) < 0.1 else "Medium"
                st.metric("Confidence", confidence)
            
            # Visualization
            fig = go.Figure(data=[
                go.Bar(x=list(predictions.keys()), 
                      y=[pred[0] if isinstance(pred, np.ndarray) else pred for pred in predictions.values()],
                      marker_color='lightblue')
            ])
            fig.update_layout(
                title='Price Predictions by Model Type',
                xaxis_title='Model Type',
                yaxis_title='Predicted Price ($)',
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")
    
    # Batch prediction
    st.markdown("### 📊 Batch Prediction")
    st.markdown("Upload a CSV file with multiple diamonds for batch prediction.")
    
    uploaded_file = st.file_uploader("Choose CSV file for batch prediction", type="csv")
    if uploaded_file is not None:
        try:
            batch_data = pd.read_csv(uploaded_file)
            st.dataframe(batch_data.head())
            
            if st.button("Predict Batch Prices"):
                with st.spinner("Processing batch predictions..."):
                    batch_predictions = []
                    for _, row in batch_data.iterrows():
                        pred = st.session_state.predictor.predict_price(row.to_dict(), 'best')
                        batch_predictions.append(pred[0] if isinstance(pred, np.ndarray) else pred)
                    
                    batch_data['Predicted_Price'] = batch_predictions
                    st.success("Batch predictions completed!")
                    st.dataframe(batch_data)
                    
                    # Download results
                    csv = batch_data.to_csv(index=False)
                    st.download_button(
                        label="Download Results",
                        data=csv,
                        file_name="diamond_predictions.csv",
                        mime="text/csv"
                    )
        
        except Exception as e:
            st.error(f"Batch prediction failed: {str(e)}")

def show_comparison_page():
    st.markdown('<h2 class="sub-header">Model Performance Comparison</h2>', unsafe_allow_html=True)
    
    if not st.session_state.models_trained:
        st.warning("Please train models first in the Model Training page.")
        return
    
    try:
        # Get comparison results
        if hasattr(st.session_state.predictor, 'comparison_results'):
            comparison_df = st.session_state.predictor.comparison_results
        else:
            comparison_df = st.session_state.predictor.compare_all_models()
        
        # Performance metrics
        st.markdown("### 📊 Performance Metrics")
        st.dataframe(comparison_df, use_container_width=True)
        
        # Visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            # R² Score comparison
            fig_r2 = px.bar(comparison_df, x='R² Score', y='Model', orientation='h',
                           color='Type', title='R² Score Comparison')
            fig_r2.update_layout(yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig_r2, use_container_width=True)
        
        with col2:
            # RMSE comparison
            fig_rmse = px.bar(comparison_df, x='RMSE', y='Model', orientation='h',
                             color='Type', title='RMSE Comparison')
            fig_rmse.update_layout(yaxis={'categoryorder': 'total descending'})
            st.plotly_chart(fig_rmse, use_container_width=True)
        
        # Model type performance
        st.markdown("### 🏆 Performance by Model Type")
        
        type_performance = comparison_df.groupby('Type').agg({
            'R² Score': ['mean', 'std', 'max'],
            'RMSE': ['mean', 'std', 'min'],
            'MAE': ['mean', 'std', 'min']
        }).round(4)
        
        st.dataframe(type_performance, use_container_width=True)
        
        # Best model details
        best_model = comparison_df.iloc[0]
        st.markdown(f"""
        <div class="prediction-result">
            🏆 Overall Best Model: {best_model['Model']}<br>
            Type: {best_model['Type']}<br>
            R² Score: {best_model['R² Score']:.4f} | RMSE: ${best_model['RMSE']:.2f} | MAE: ${best_model['MAE']:.2f}
        </div>
        """, unsafe_allow_html=True)
        
        # Feature importance (if available)
        st.markdown("### 🎯 Feature Importance Analysis")
        
        try:
            importance_df = st.session_state.predictor.get_feature_importance_analysis()
            if importance_df is not None:
                # Plot feature importance
                fig_importance = px.bar(
                    x=importance_df['Mean'].values,
                    y=importance_df.index,
                    orientation='h',
                    title='Feature Importance'
                )
                fig_importance.update_layout(yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig_importance, use_container_width=True)
        except:
            st.info("Feature importance analysis not available for current models.")
    
    except Exception as e:
        st.error(f"Error displaying comparison: {str(e)}")

def show_about_page():
    st.markdown('<h2 class="sub-header">About Diamond Price Predictor</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    ### 🎯 Project Overview
    
    The Diamond Price Predictor is a comprehensive machine learning application that combines traditional ML algorithms 
    with deep learning models to provide accurate diamond price predictions. This project demonstrates the power of 
    ensemble methods and model comparison in real-world applications.
    
    ### 🛠️ Technologies Used
    
    **Machine Learning:**
    - Linear Regression, Ridge, Lasso
    - Random Forest, Gradient Boosting
    - XGBoost, Support Vector Regression
    
    **Deep Learning:**
    - Basic Neural Networks
    - Deep Neural Networks with Batch Normalization
    - Residual Neural Networks
    - Ensemble Neural Networks
    
    **Libraries & Frameworks:**
    - Scikit-learn for traditional ML
    - TensorFlow/Keras for deep learning
    - Streamlit for web interface
    - Plotly for interactive visualizations
    - Pandas & NumPy for data processing
    
    ### 📊 Features
    
    - **Data Analysis**: Comprehensive exploratory data analysis with interactive visualizations
    - **Model Training**: Train multiple ML and DL models with hyperparameter optimization
    - **Price Prediction**: Single and batch diamond price predictions
    - **Model Comparison**: Detailed performance comparison across all models
    - **Feature Importance**: Analysis of which features most influence diamond prices
    
    ### 🎨 Diamond Characteristics
    
    The model considers the following diamond characteristics:
    
    - **Carat**: Weight of the diamond
    - **Cut**: Quality of the cut (Fair, Good, Very Good, Premium, Ideal)
    - **Color**: Diamond color grade (D-J, with D being colorless)
    - **Clarity**: Diamond clarity grade (FL to I1)
    - **Depth**: Total depth percentage
    - **Table**: Width of the diamond's table
    - **Dimensions**: Length (x), Width (y), Height (z) in mm
    
    ### 📈 Model Performance
    
    The application trains and compares multiple models:
    
    1. **Traditional ML Models**: Fast training, interpretable results
    2. **Deep Learning Models**: Complex pattern recognition, high accuracy
    3. **Ensemble Methods**: Combines multiple models for better predictions
    
    ### 🚀 Getting Started
    
    1. Navigate to **Data Analysis** to explore the dataset
    2. Go to **Model Training** to train ML and DL models
    3. Use **Price Prediction** to predict individual diamond prices
    4. Check **Model Comparison** to see performance metrics
    
    ### 📝 Notes
    
    - The synthetic dataset is generated for demonstration purposes
    - Real-world applications should use actual diamond market data
    - Model performance may vary based on data quality and size
    - Hyperparameter tuning can significantly improve results
    
    ### 👨‍💻 Development
    
    This application showcases modern ML/DL practices including:
    - Cross-validation and hyperparameter optimization
    - Model ensembling and stacking
    - Interactive web interfaces
    - Comprehensive model evaluation
    - Production-ready code structure
    """)
    
    # Technical specifications
    st.markdown("### 🔧 Technical Specifications")
    
    specs = {
        "Python Version": "3.8+",
        "ML Framework": "Scikit-learn 1.3+",
        "DL Framework": "TensorFlow 2.15+",
        "Web Framework": "Streamlit 1.29+",
        "Visualization": "Plotly 5.17+",
        "Data Processing": "Pandas 2.1+, NumPy 1.24+"
    }
    
    for key, value in specs.items():
        st.markdown(f"- **{key}**: {value}")

if __name__ == "__main__":
    main()