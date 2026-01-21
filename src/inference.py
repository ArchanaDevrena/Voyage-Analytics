import joblib
import pandas as pd
from .preprocess import preprocess_data

MODEL_PATH = "models/flight_price_model.pkl"

model = joblib.load(MODEL_PATH)


def predict_price(input_data: dict) -> float:
    df = pd.DataFrame([input_data])

    # Add dummy price to satisfy preprocess_data
    df['price'] = 0  

    X, _ = preprocess_data(df)

    prediction = model.predict(X)[0]
    return float(prediction)
