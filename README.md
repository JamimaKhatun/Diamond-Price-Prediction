# 💎 Diamond Price Predictor

A comprehensive machine learning and deep learning system for predicting diamond prices using advanced algorithms and ensemble methods.

## 🌟 Features

- **Multiple ML Algorithms**: Linear Regression, Random Forest, XGBoost, SVM, and more
- **Deep Learning Models**: Neural Networks with various architectures including residual connections
- **Ensemble Methods**: Combines multiple models for superior accuracy
- **Interactive Web Interface**: Streamlit-based web application
- **Command Line Interface**: Full-featured CLI for data scientists
- **Jupyter Notebook**: Comprehensive analysis with interactive widgets
- **Model Comparison**: Detailed performance analysis across all models
- **Feature Importance**: Analysis of which diamond characteristics matter most
- **Batch Prediction**: Process multiple diamonds at once
- **Model Persistence**: Save and load trained models

## 📊 Supported Diamond Characteristics

- **Carat**: Weight of the diamond (0.2 - 5.0)
- **Cut**: Quality of the cut (Fair, Good, Very Good, Premium, Ideal)
- **Color**: Diamond color grade (D-J, with D being colorless)
- **Clarity**: Diamond clarity grade (FL, IF, VVS1, VVS2, VS1, VS2, SI1, SI2, I1)
- **Depth**: Total depth percentage (55-70%)
- **Table**: Width of the diamond's table (50-70%)
- **Dimensions**: Length (x), Width (y), Height (z) in millimeters

## 🚀 Quick Start

### Installation

1. **Clone the repository**:
```bash
git clone <repository-url>
cd diamond_price_predictor
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

### Usage Options

#### 1. 🌐 Web Interface (Recommended for beginners)

```bash
streamlit run streamlit_app.py
```

Then open your browser to `http://localhost:8501`

#### 2. 📓 Jupyter Notebook (Recommended for data scientists)

```bash
jupyter notebook Diamond_Price_Prediction_Analysis.ipynb
```

#### 3. 💻 Command Line Interface (Recommended for automation)

```bash
# Complete analysis
python cli.py --complete-analysis

# Train models only
python cli.py --train --epochs 100

# Interactive prediction
python cli.py --interactive

# Single prediction
python cli.py --predict --carat 1.5 --cut Ideal --color D --clarity VVS1

# Batch prediction
python cli.py --batch-predict diamonds.csv --output predictions.csv
```

## 📁 Project Structure

```
diamond_price_predictor/
├── src/                          # Source code modules
│   ├── data_processor.py         # Data loading and preprocessing
│   ├── ml_models.py              # Traditional ML models
│   ├── deep_learning_models.py   # Neural network models
│   └── diamond_predictor.py      # Main predictor class
├── data/                         # Dataset storage
│   └── diamonds.csv              # Sample diamond data
├── saved_models/                 # Trained model storage
├── streamlit_app.py              # Web interface
├── cli.py                        # Command line interface
├── Diamond_Price_Prediction_Analysis.ipynb  # Jupyter notebook
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## 🤖 Machine Learning Models

### Traditional ML Models
- **Linear Regression**: Simple baseline model
- **Ridge Regression**: L2 regularized linear model
- **Lasso Regression**: L1 regularized linear model
- **Random Forest**: Ensemble of decision trees
- **Gradient Boosting**: Sequential ensemble method
- **XGBoost**: Optimized gradient boosting
- **Support Vector Regression**: SVM for regression

### Deep Learning Models
- **Basic Neural Network**: Simple feedforward network
- **Deep Neural Network**: Multi-layer network with batch normalization
- **Wide Neural Network**: Network with wide hidden layers
- **Residual Neural Network**: Network with skip connections
- **Ensemble Neural Network**: Multi-branch architecture

### Ensemble Methods
- **Weighted Ensemble**: Combines top-performing models
- **Stacking**: Uses meta-learner to combine predictions
- **Voting**: Simple averaging of multiple models

## 📈 Model Performance

The system automatically evaluates all models using multiple metrics:

- **R² Score**: Coefficient of determination
- **RMSE**: Root Mean Square Error
- **MAE**: Mean Absolute Error
- **Cross-Validation**: 5-fold cross-validation scores
- **Feature Importance**: For tree-based models

## 🎯 Usage Examples

### Web Interface

1. **Data Analysis**: Explore the diamond dataset with interactive visualizations
2. **Model Training**: Train both ML and DL models with custom parameters
3. **Price Prediction**: Predict individual diamond prices with confidence intervals
4. **Model Comparison**: Compare performance across all model types

### Command Line Examples

```bash
# Complete workflow
python cli.py --complete-analysis --save-models

# Train only traditional ML models
python cli.py --train --ml-only --save-models

# Train only deep learning models with custom parameters
python cli.py --train --dl-only --epochs 200 --batch-size 64

# Predict with specific diamond characteristics
python cli.py --predict \
    --carat 2.0 \
    --cut "Ideal" \
    --color "D" \
    --clarity "VVS1" \
    --depth 61.5 \
    --table 57.0 \
    --x 8.0 \
    --y 8.0 \
    --z 5.0

# Batch prediction from CSV file
python cli.py --batch-predict input_diamonds.csv --output predicted_prices.csv

# Load saved models and predict
python cli.py --load-models --predict --carat 1.5 --cut Premium

# Show model performance summary
python cli.py --load-models --summary
```

### Python API

```python
from src.diamond_predictor import DiamondPricePredictor

# Initialize predictor
predictor = DiamondPricePredictor()

# Load and prepare data
data = predictor.load_and_prepare_data()

# Train all models
predictor.train_traditional_ml_models()
predictor.train_deep_learning_models()

# Compare models
comparison = predictor.compare_all_models()

# Make prediction
diamond = {
    'carat': 1.5,
    'cut': 'Ideal',
    'color': 'D',
    'clarity': 'VVS1',
    'depth': 61.5,
    'table': 57.0,
    'x': 7.3,
    'y': 7.3,
    'z': 4.5
}

price = predictor.predict_price(diamond)
print(f"Predicted price: ${price:,.2f}")
```

## 📊 Data Format

### Input CSV Format
```csv
carat,cut,color,clarity,depth,table,x,y,z
1.5,Ideal,D,VVS1,61.5,57.0,7.3,7.3,4.5
1.0,Premium,E,VS1,62.0,58.0,6.2,6.2,3.8
```

### Prediction Output Format
```csv
carat,cut,color,clarity,depth,table,x,y,z,predicted_price
1.5,Ideal,D,VVS1,61.5,57.0,7.3,7.3,4.5,12500.50
1.0,Premium,E,VS1,62.0,58.0,6.2,6.2,3.8,5200.25
```

## 🔧 Configuration

### Training Parameters

- **ML Models**: Automatic hyperparameter optimization using Optuna
- **DL Models**: Configurable epochs, batch size, learning rate
- **Ensemble**: Weighted combination based on validation performance

### Model Selection

- **Best Model**: Automatically selects highest-performing model
- **Specific Model**: Choose ML or DL model type
- **Ensemble**: Combines multiple models for better accuracy

## 📝 Model Interpretability

### Feature Importance
- Tree-based models provide feature importance scores
- SHAP values for model explanation (optional)
- Correlation analysis between features

### Model Comparison
- Performance metrics across all models
- Visualization of prediction accuracy
- Residual analysis for error patterns

## 🚀 Advanced Features

### Hyperparameter Optimization
- Automated tuning using Optuna
- Cross-validation for robust evaluation
- Bayesian optimization for efficiency

### Model Ensemble
- Weighted averaging based on performance
- Stacking with meta-learners
- Dynamic model selection

### Production Ready
- Model serialization and loading
- Batch processing capabilities
- Error handling and logging
- Input validation and preprocessing

## 🛠️ Development

### Adding New Models

1. **Traditional ML**: Add to `ml_models.py`
```python
def create_new_model(self):
    model = YourNewModel()
    self.models['New Model'] = model
```

2. **Deep Learning**: Add to `deep_learning_models.py`
```python
def create_new_nn(self, input_dim):
    model = keras.Sequential([...])
    return model
```

### Custom Features

1. **Data Processing**: Modify `data_processor.py`
2. **Evaluation Metrics**: Add to model classes
3. **Visualization**: Extend plotting functions

## 📋 Requirements

- Python 3.8+
- pandas 2.1+
- numpy 1.24+
- scikit-learn 1.3+
- tensorflow 2.15+
- xgboost 2.0+
- streamlit 1.29+
- plotly 5.17+
- matplotlib 3.8+
- seaborn 0.13+

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Diamond dataset inspired by the classic diamonds dataset
- Scikit-learn for traditional ML algorithms
- TensorFlow/Keras for deep learning capabilities
- Streamlit for the web interface
- Plotly for interactive visualizations

## 📞 Support

For questions, issues, or contributions:

1. **Issues**: Open a GitHub issue
2. **Discussions**: Use GitHub discussions
3. **Documentation**: Check the Jupyter notebook for detailed examples

## 🔮 Future Enhancements

- [ ] Real-time market data integration
- [ ] Advanced ensemble methods (stacking, blending)
- [ ] Model explainability with SHAP
- [ ] A/B testing framework
- [ ] REST API for web services
- [ ] Docker containerization
- [ ] Cloud deployment guides
- [ ] Time series analysis for price trends
- [ ] Image-based diamond analysis
- [ ] Mobile app interface

---

**Happy Diamond Price Predicting! 💎✨**