import os
import joblib
import pandas as pd
import category_encoders as ce
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline

from preprocess import preprocess_data, CATEGORICAL_COLS

DATA_PATH = "data/flights.csv"
MODEL_PATH = "models/flight_price_model.pkl"

def build_model():
    return Pipeline([
        ('encoder', ce.TargetEncoder(
            cols=CATEGORICAL_COLS,
            smoothing=10
        )),
        ('xgb', XGBRegressor(
            max_depth=4,
            n_estimators=400,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        ))
    ])

def main():
    df = pd.read_csv(DATA_PATH)
    X, y = preprocess_data(df)

    model = build_model()
    model.fit(X, y)

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    print("✅ Flight price model trained and saved")

if __name__ == "__main__":
    main()
