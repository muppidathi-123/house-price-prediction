"""
src/predict.py
---------------
Prediction pipeline for the House Price Prediction project.

This module loads the saved model and preprocessor, then exposes
a clean predict() interface for use by the Streamlit app.

Usage:
    from src.predict import HousePricePredictor

    predictor = HousePricePredictor()
    result = predictor.predict({
        'MedInc'    : 5.0,
        'HouseAge'  : 20.0,
        'AveRooms'  : 6.0,
        'AveBedrms' : 1.0,
        'Population': 500.0,
        'AveOccup'  : 2.5,
        'Latitude'  : 37.77,
        'Longitude' : -122.41
    })
    print(result)
"""

import numpy as np
import joblib
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class HousePricePredictor:
    """
    Loads the saved model and preprocessor, and provides
    a simple predict() interface for single house predictions.
    """

    # Class-level cache — model loads once per session, not per call
    _model        = None
    _preprocessor = None

    def __init__(
        self,
        model_path: str        = None,
        preprocessor_path: str = None
    ):
        """
        Initialize predictor and load model artifacts.

        Args:
            model_path        : path to saved model .pkl file
            preprocessor_path : path to saved preprocessor .pkl file
        """
        # Use dynamic paths if none provided — works locally AND in cloud
        self.model_path        = model_path or os.path.join(PROJECT_ROOT, 'models', 'best_model.pkl')
        self.preprocessor_path = preprocessor_path or os.path.join(PROJECT_ROOT, 'models', 'preprocessor.pkl')
        self._load_artifacts()

    # ------------------------------------------------------------------
    # PRIVATE: Loading
    # ------------------------------------------------------------------
    def _load_artifacts(self):
        """Load model and preprocessor from disk if not already cached."""

        # Only load once — reuse if already loaded
        if HousePricePredictor._model is None:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(
                    f"Model file not found: {self.model_path}\n"
                    f"Please run: python -m src.train"
                )
            if not os.path.exists(self.preprocessor_path):
                raise FileNotFoundError(
                    f"Preprocessor file not found: {self.preprocessor_path}\n"
                    f"Please run: python -m src.train"
                )

            HousePricePredictor._model        = joblib.load(self.model_path)
            HousePricePredictor._preprocessor = joblib.load(self.preprocessor_path)

        self.model        = HousePricePredictor._model
        self.preprocessor = HousePricePredictor._preprocessor

    # ------------------------------------------------------------------
    # PRIVATE: Validation
    # ------------------------------------------------------------------
    def _validate_input(self, input_dict: dict) -> None:
        """
        Validate raw user input before processing.
        Raises ValueError with clear messages on bad input.
        """
        required_fields = [
            'MedInc', 'HouseAge', 'AveRooms', 'AveBedrms',
            'Population', 'AveOccup', 'Latitude', 'Longitude'
        ]

        # Check all fields are present
        missing = [f for f in required_fields if f not in input_dict]
        if missing:
            raise ValueError(f"Missing required fields: {missing}")

        # Check numeric types
        for field in required_fields:
          val = input_dict[field]

        if not isinstance(val, (int, float)):
          raise ValueError(
           f"Field '{field}' must be a number, got {type(val).__name__}"
        )

        # These fields must be non-negative
        non_negative_fields = [
        'MedInc',
        'HouseAge',
        'AveRooms',
        'AveBedrms',
        'Population',
        'AveOccup'
        ]

        for field in non_negative_fields:
          if input_dict[field] < 0:
           raise ValueError(
            f"Field '{field}' must be non-negative, got {input_dict[field]}"
        )

        # California-specific geographic bounds
        if not (32.0 <= input_dict['Latitude'] <= 42.0):
            raise ValueError(
                f"Latitude {input_dict['Latitude']} is outside California "
                f"(expected 32.0 – 42.0)"
            )
        if not (-125.0 <= input_dict['Longitude'] <= -114.0):
            raise ValueError(
                f"Longitude {input_dict['Longitude']} is outside California "
                f"(expected -125.0 to -114.0)"
            )

        # Logical checks
        if input_dict['AveBedrms'] > input_dict['AveRooms']:
            raise ValueError(
                "AveBedrms cannot be greater than AveRooms"
            )
        if input_dict['AveOccup'] < 1.0:
            raise ValueError(
                "AveOccup (average occupants) must be at least 1.0"
            )

    # ------------------------------------------------------------------
    # PUBLIC: Predict
    # ------------------------------------------------------------------
    def predict(self, input_dict: dict) -> dict:
        """
        Generate a price prediction for a single house.

        Args:
            input_dict: dict with 8 raw house features

        Returns:
            dict with keys:
                predicted_price      : float, price in USD
                predicted_price_100k : float, price in $100k units
                confidence_range     : tuple, (low, high) estimate in USD
                model_name           : str, name of model used
        """
        # Step 1: Validate input
        self._validate_input(input_dict)

        # Step 2: Preprocess (same pipeline as training)
        X_scaled = self.preprocessor.transform_single(input_dict)

        # Step 3: Predict in log scale
        log_prediction = self.model.predict(X_scaled)[0]

        # Step 4: Convert back to original price scale
        # We used np.log1p during training, so np.expm1 reverses it
        predicted_100k = float(np.expm1(log_prediction))
        predicted_usd  = predicted_100k * 100_000

        # Step 5: Build a simple confidence range (±15% heuristic)
        # Note: A full implementation would use prediction intervals
        margin = 0.15
        low_usd  = predicted_usd * (1 - margin)
        high_usd = predicted_usd * (1 + margin)

        return {
            'predicted_price'      : round(predicted_usd, 2),
            'predicted_price_100k' : round(predicted_100k, 4),
            'confidence_range'     : (round(low_usd, 2), round(high_usd, 2)),
            'model_name'           : type(self.model).__name__
        }

    def predict_batch(self, records: list[dict]) -> list[dict]:
        """
        Predict prices for a list of houses.

        Args:
            records: list of input dicts

        Returns:
            list of result dicts
        """
        return [self.predict(record) for record in records]

    def get_model_info(self) -> dict:
        """Return basic information about the loaded model."""
        return {
            'model_type'    : type(self.model).__name__,
            'feature_count' : len(self.preprocessor.feature_names),
            'features'      : self.preprocessor.feature_names,
            'model_path'    : self.model_path
        }