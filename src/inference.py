import joblib
import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient
from preprocess import preprocess_data

# --- MLflow setup ---
mlflow.set_tracking_uri("sqlite:///../mlflow.db")  # relative to repo root
EXPERIMENT_NAME = "Flight Price Prediction"
MODEL_PATH = "models/flight_price_model.pkl"  # fallback

def load_best_model():
    import mlflow
    import mlflow.sklearn
    import joblib
    import os

    EXPERIMENT_NAME = "Flight Price Prediction"
    LOCAL_MODEL_PATH = os.path.join("models", "flight_model_depth5_lr0.2.pkl")

    try:
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        mlflow.set_experiment(EXPERIMENT_NAME)
        client = mlflow.tracking.MlflowClient()
        exp = client.get_experiment_by_name(EXPERIMENT_NAME)
        if exp is None:
            raise ValueError(f"Experiment {EXPERIMENT_NAME} not found")

        # Get best run by MAE
        runs = client.search_runs(
            experiment_ids=[exp.experiment_id],
            order_by=["metrics.MAE ASC"],
            max_results=1
        )
        best_run = runs[0]
        model_uri = f"runs:/{best_run.info.run_id}/model"
        print(f"Loading best MLflow model: {best_run.info.run_id}")
        return mlflow.sklearn.load_model(model_uri)
    except Exception as e:
        print(f"MLflow load failed: {e}. Using local .pkl model.")
        return joblib.load(LOCAL_MODEL_PATH)


# Load model once
model = load_best_model()

def predict_price(input_data: dict) -> float:
    df = pd.DataFrame([input_data])
    df['price'] = 0  # dummy price to satisfy preprocess
    X, _ = preprocess_data(df)
    prediction = model.predict(X)[0]
    return float(prediction)
