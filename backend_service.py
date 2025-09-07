"""
Backend Prediction Service for DiamondAI
- Tries to load/use the ML pipeline from src (Traditional ML)
- Falls back to a mock predictor if ML stack is unavailable
- Provides a simple interface for prediction and readiness checks
"""

import os
import sys
import traceback
from datetime import datetime

import numpy as np

# Attempt to import full ML pipeline
_HAS_FULL_STACK = True
try:
    from data_processor import DiamondDataProcessor
    from ml_models import TraditionalMLModels
    # Avoid importing heavy DL by default
except Exception:
    _HAS_FULL_STACK = False


class PredictionService:
    """Encapsulates model loading/training and prediction logic."""

    def __init__(self, base_dir=None, model_dir="saved_models"):
        self.base_dir = base_dir or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.model_dir = os.path.join(self.base_dir, model_dir)

        self.ready = False
        self.using_mock = False
        self.model_name = "Mock Predictor"

        # Initialize processor and models if stack available
        if _HAS_FULL_STACK:
            try:
                self.data_processor = DiamondDataProcessor()
                self.ml_models = TraditionalMLModels()

                # Try loading previously saved models first
                loaded = self._try_load_saved()
                if not loaded:
                    # Train quickly on local CSV (if present) or synthetic
                    data_path = os.path.join(self.base_dir, "data", "diamonds.csv")
                    self._train_models(data_path if os.path.exists(data_path) else None)

                self.ready = True
                self.using_mock = False
                self.model_name = self.ml_models.best_model_name or "Traditional ML"
            except Exception:
                # Fall back to mock if anything fails
                traceback.print_exc()
                self.ready = True
                self.using_mock = True
                self.data_processor = None
                self.ml_models = None
                self.model_name = "Mock Predictor"
        else:
            # Only mock available
            self.ready = True
            self.using_mock = True
            self.data_processor = None
            self.ml_models = None
            self.model_name = "Mock Predictor"

    def is_ready(self):
        return bool(self.ready)

    def _try_load_saved(self):
        """Attempt to load previously saved processor and best model metadata."""
        try:
            # We only need the data_processor and results to know best model
            proc_path = os.path.join(self.model_dir, "data_processor.joblib")
            comp_path = os.path.join(self.model_dir, "model_comparison.csv")
            if not (os.path.exists(proc_path) and os.path.exists(comp_path)):
                return False

            # Rebuild minimal state
            import joblib
            import pandas as pd
            self.data_processor = joblib.load(proc_path)
            comp = pd.read_csv(comp_path)
            # Initialize ML models just to hold a container for predict calls
            self.ml_models = TraditionalMLModels()
            self.ml_models.results = {}
            # Load the best model by name if present on disk
            best_name = comp.iloc[0]["Model"]
            if best_name.startswith("ML: "):
                model_name_clean = best_name.replace("ML: ", "")
            else:
                model_name_clean = best_name.replace("DL: ", "")
            # Try to load corresponding saved model
            prefix = os.path.join(self.model_dir, "ml_model")
            model_file = f"{prefix}_{model_name_clean.replace(' ', '_').lower()}.joblib"
            if os.path.exists(model_file):
                self.ml_models.best_model_name = model_name_clean
                self.ml_models.models = {}
                # store into results for predict convenience (consistent with ml_models API)
                self.ml_models.results[model_name_clean] = {
                    "model": joblib.load(model_file)
                }
                return True
            return False
        except Exception:
            return False

    def _train_models(self, csv_path=None):
        """Train traditional ML quickly and save artifacts."""
        # Load & preprocess
        df = self.data_processor.load_data(csv_path)
        (
            X_train_scaled,
            X_test_scaled,
            y_train,
            y_test,
            X_train,
            X_test,
        ) = self.data_processor.preprocess_data(df)

        # Train baseline models
        self.ml_models.initialize_models()
        self.ml_models.train_all_models(X_train_scaled, y_train, X_test_scaled, y_test)

        # Build a minimal comparison to pick best and save
        try:
            from diamond_predictor import DiamondPricePredictor  # for save_all_models helper
            dpp = DiamondPricePredictor()
            # Reuse fitted components
            dpp.data_processor = self.data_processor
            dpp.ml_models = self.ml_models
            import pandas as pd
            rows = []
            for name, r in self.ml_models.results.items():
                rows.append({
                    "Model": f"ML: {name}",
                    "Type": "Traditional ML",
                    "R² Score": r.get("test_r2", 0),
                    "RMSE": r.get("test_rmse", 0),
                    "MAE": r.get("test_mae", 0),
                })
            dpp.comparison_results = pd.DataFrame(rows).sort_values("R² Score", ascending=False)
            # Save artifacts
            if not os.path.exists(self.model_dir):
                os.makedirs(self.model_dir, exist_ok=True)
            dpp.save_all_models(self.model_dir)
        except Exception:
            # Saving is optional; continue
            pass

    def predict(self, features: dict) -> dict:
        """Return a normalized prediction payload.
        Output keys are camelCase for frontend compatibility.
        """
        if self.using_mock or not _HAS_FULL_STACK:
            return self._mock_predict(features)

        try:
            # Build DataFrame-like input and scale via processor
            import pandas as pd
            input_df = pd.DataFrame([{
                "carat": float(features.get("carat", 1.0)),
                "cut": features.get("cut", "Good"),
                "color": features.get("color", "G"),
                "clarity": features.get("clarity", "VS1"),
                "depth": float(features.get("depth", 61.5)),
                "table": float(features.get("table", 57.0)),
                "x": float(features.get("x", 6.0)),
                "y": float(features.get("y", 6.0)),
                "z": float(features.get("z", 3.7)),
            }])

            processed = self.data_processor.prepare_prediction_data(input_df)

            # Prefer best model if available
            model = None
            if self.ml_models and self.ml_models.best_model_name and self.ml_models.results.get(self.ml_models.best_model_name):
                model = self.ml_models.results[self.ml_models.best_model_name]["model"]
            elif self.ml_models and self.ml_models.results:
                # pick any
                model = next(iter(self.ml_models.results.values())).get("model")

            if model is None:
                return self._mock_predict(features)

            price = int(max(300, float(model.predict(processed).ravel()[0])))
            carat = max(0.01, float(features.get("carat", 1.0)))
            price_per_carat = int(price / carat)

            # Confidence heuristic (placeholder)
            confidence = int(90 + np.random.randint(0, 8))

            # Category
            if price < 2000:
                category, icon = "Budget-Friendly", "💚"
            elif price < 5000:
                category, icon = "Mid-Range", "💛"
            elif price < 10000:
                category, icon = "Premium", "🧡"
            else:
                category, icon = "Luxury", "❤️"

            return {
                "price": price,
                "confidence": confidence,
                "priceRange": {"min": int(price * 0.85), "max": int(price * 1.15)},
                "pricePerCarat": price_per_carat,
                "category": category,
                "categoryIcon": icon,
                "model": self.model_name,
                "featuresImportance": {},  # could be filled for tree-based models
            }
        except Exception:
            return self._mock_predict(features)

    def _mock_predict(self, features: dict) -> dict:
        """Pure mock fallback aligned with frontend schema (camelCase)."""
        carat = float(features.get("carat", 1.0))
        base_price = 3000 * (carat ** 2.5)

        cut_mult = {"Fair": 0.8, "Good": 0.9, "Very Good": 1.0, "Premium": 1.1, "Ideal": 1.2}
        color_mult = {"D": 1.3, "E": 1.2, "F": 1.1, "G": 1.0, "H": 0.9, "I": 0.8, "J": 0.7}
        clarity_mult = {"FL": 2.0, "IF": 1.8, "VVS1": 1.6, "VVS2": 1.4, "VS1": 1.2, "VS2": 1.0, "SI1": 0.8, "SI2": 0.6, "I1": 0.4}

        base_price *= cut_mult.get(features.get("cut", "Good"), 1.0)
        base_price *= color_mult.get(features.get("color", "G"), 1.0)
        base_price *= clarity_mult.get(features.get("clarity", "VS1"), 1.0)

        variance = np.random.normal(1.0, 0.05)
        price = int(max(300, base_price * variance))

        if price < 2000:
            category, icon = "Budget-Friendly", "💚"
        elif price < 5000:
            category, icon = "Mid-Range", "💛"
        elif price < 10000:
            category, icon = "Premium", "🧡"
        else:
            category, icon = "Luxury", "❤️"

        return {
            "price": price,
            "confidence": int(90 + np.random.randint(0, 8)),
            "priceRange": {"min": int(price * 0.85), "max": int(price * 1.15)},
            "pricePerCarat": int(price / max(carat, 0.01)),
            "category": category,
            "categoryIcon": icon,
            "model": "Mock Predictor",
            "featuresImportance": {
                "carat": 0.86,
                "clarity": 0.05,
                "color": 0.03,
                "cut": 0.02,
            },
        }