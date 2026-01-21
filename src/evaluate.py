import joblib
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

from preprocess import preprocess_data

MODEL_PATH = "models/flight_price_model.pkl"
DATA_PATH = "data/flights.csv"

def main():
    df = pd.read_csv(DATA_PATH)
    X, y = preprocess_data(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    model = joblib.load(MODEL_PATH)
    preds = model.predict(X_test)

    print("MAE:", mean_absolute_error(y_test, preds))
    print("R2 :", r2_score(y_test, preds))

if __name__ == "__main__":
    main()
