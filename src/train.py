"""
src/train.py
-------------
Production training script.
Run this file to retrain the model from scratch:
    python src/train.py
"""

import json
import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

from src.preprocessor import HousingPreprocessor


def train():
    print("=" * 55)
    print("HOUSE PRICE PREDICTION — TRAINING PIPELINE")
    print("=" * 55)

    # Step 1: Load raw data
    print("\n[1/5] Loading dataset...")
    housing = fetch_california_housing(as_frame=True)
    df = housing.frame.copy()
    print(f"      ✅ {df.shape[0]:,} rows loaded")

    # Step 2: Preprocess
    print("\n[2/5] Preprocessing data...")
    preprocessor = HousingPreprocessor(test_size=0.2, random_state=42)
    X_train, X_test, y_train, y_test = preprocessor.fit_transform(df)
    print(f"      ✅ Train: {X_train.shape} | Test: {X_test.shape}")

    # Step 3: Train XGBoost
    print("\n[3/5] Training XGBoost model...")
    model = XGBRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbosity=0
    )
    model.fit(X_train, y_train)
    print("      ✅ Training complete")

    # Step 4: Evaluate
    print("\n[4/5] Evaluating model...")
    y_pred_log = model.predict(X_test)
    y_pred = np.expm1(y_pred_log)
    y_true = np.expm1(y_test)

    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)

    print(f"      MAE  : ${mae * 100_000:,.0f}")
    print(f"      RMSE : ${rmse * 100_000:,.0f}")
    print(f"      R²   : {r2:.4f}")

    # Step 5: Save model and preprocessor
    print("\n[5/5] Saving model and preprocessor...")
    joblib.dump(model, 'models/best_model.pkl')
    joblib.dump(preprocessor, 'models/preprocessor.pkl')

    # Save feature names
    with open('data/processed/feature_names.json', 'w') as f:
        json.dump(preprocessor.feature_names, f)

    print("      ✅ models/best_model.pkl saved")
    print("      ✅ models/preprocessor.pkl saved")
    print("\n🏁 Training pipeline complete!")


if __name__ == "__main__":
    train()