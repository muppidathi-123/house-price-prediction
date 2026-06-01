"""
src/preprocessor.py
--------------------
Reusable preprocessing pipeline for the House Price Prediction project.
This module is used by both the training script and the Streamlit app.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


class HousingPreprocessor:
    """
    Handles all preprocessing steps for the California Housing dataset.

    Steps performed:
        1. Outlier capping (IQR Winsorization)
        2. Feature engineering
        3. Log transformation of target
        4. Train/test split
        5. Feature scaling
    """

    def __init__(self, test_size: float = 0.2, random_state: int = 42):
        self.test_size = test_size
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.feature_names = None
        self.iqr_bounds = {}   # stores capping bounds for reuse at inference

    # ------------------------------------------------------------------
    # STEP 1: Outlier Capping
    # ------------------------------------------------------------------
    def _cap_outliers(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Cap outliers using IQR method. If fit=True, compute bounds from data."""
        cols_to_cap = ['AveRooms', 'AveBedrms', 'AveOccup', 'Population']
        df = df.copy()

        for col in cols_to_cap:
            if col not in df.columns:
                continue
            if fit:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                self.iqr_bounds[col] = {
                    'lower': Q1 - 1.5 * IQR,
                    'upper': Q3 + 1.5 * IQR
                }
            lower = self.iqr_bounds[col]['lower']
            upper = self.iqr_bounds[col]['upper']
            df[col] = df[col].clip(lower=lower, upper=upper)

        return df

    # ------------------------------------------------------------------
    # STEP 2: Feature Engineering
    # ------------------------------------------------------------------
    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create new meaningful features from existing ones."""
        df = df.copy()

        df['RoomsPerPerson'] = df['AveRooms'] / df['AveOccup']
        df['BedroomRatio'] = df['AveBedrms'] / df['AveRooms']
        df['IncomePerPerson'] = df['MedInc'] / df['AveOccup']

        # Distance from major California cities
        df['DistanceFromSF'] = np.sqrt(
            (df['Latitude'] - 37.7749) ** 2 +
            (df['Longitude'] - (-122.4194)) ** 2
        )
        df['DistanceFromLA'] = np.sqrt(
            (df['Latitude'] - 34.0522) ** 2 +
            (df['Longitude'] - (-118.2437)) ** 2
        )

        return df

    # ------------------------------------------------------------------
    # PUBLIC METHODS
    # ------------------------------------------------------------------
    def fit_transform(self, df: pd.DataFrame):
        """
        Full preprocessing pipeline for training data.

        Returns:
            X_train_scaled, X_test_scaled, y_train, y_test
        """
        # Step 1: Cap outliers (fit bounds from this data)
        df = self._cap_outliers(df, fit=True)

        # Step 2: Feature engineering
        df = self._engineer_features(df)

        # Step 3: Log transform the target
        df['LogMedHouseVal'] = np.log1p(df['MedHouseVal'])

        # Step 4: Define X and y
        TARGET = 'LogMedHouseVal'
        self.feature_names = [
            col for col in df.columns
            if col not in ['MedHouseVal', 'LogMedHouseVal']
        ]
        X = df[self.feature_names]
        y = df[TARGET]

        # Step 5: Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state
        )

        # Step 6: Scale features (fit on train only)
        X_train_scaled = pd.DataFrame(
            self.scaler.fit_transform(X_train),
            columns=self.feature_names
        )
        X_test_scaled = pd.DataFrame(
            self.scaler.transform(X_test),
            columns=self.feature_names
        )

        return X_train_scaled, X_test_scaled, y_train, y_test

    def transform_single(self, input_dict: dict) -> pd.DataFrame:
        """
        Preprocess a single house input for prediction.
        Used by the Streamlit app at inference time.

        Args:
            input_dict: dict with raw feature values from the user

        Returns:
            Scaled DataFrame ready for model.predict()
        """
        df_input = pd.DataFrame([input_dict])

        # Apply same capping bounds learned during training
        df_input = self._cap_outliers(df_input, fit=False)

        # Apply same feature engineering
        df_input = self._engineer_features(df_input)

        # Select only the features model was trained on
        df_input = df_input[self.feature_names]

        # Apply same scaler fitted during training
        df_scaled = pd.DataFrame(
            self.scaler.transform(df_input),
            columns=self.feature_names
        )

        return df_scaled