import os
import joblib
import pandas as pd
import category_encoders as ce
import mlflow
import mlflow.sklearn

from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

from preprocess import preprocess_data, CATEGORICAL_COLS

DATA_PATH = "data/flights.csv"
MODEL_DIR = "models"

mlflow.set_experiment("Flight Price Prediction")

def build_model(max_depth, n_estimators, learning_rate):
    return Pipeline([
        ("encoder", ce.TargetEncoder(
            cols=CATEGORICAL_COLS,
            smoothing=10
        )),
        ("xgb", XGBRegressor(
            max_depth=max_depth,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        ))
    ])

def main():
    # Load & preprocess
    df = pd.read_csv(DATA_PATH)
    X, y = preprocess_data(df)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Hyperparameter grid
    max_depth_list = [3, 4, 5]
    learning_rate_list = [0.05, 0.1, 0.2]
    n_estimators = 400

    os.makedirs(MODEL_DIR, exist_ok=True)

    # Sweep over hyperparameters
    for max_depth in max_depth_list:
        for lr in learning_rate_list:
            run_name = f"xgb_depth{max_depth}_lr{lr}"
            with mlflow.start_run(run_name=run_name):
                model = build_model(max_depth, n_estimators, lr)
                
                # Train
                model.fit(X_train, y_train)
                
                # Evaluate
                preds = model.predict(X_test)
                mae = mean_absolute_error(y_test, preds)
                r2 = r2_score(y_test, preds)
                
                # Log params
                mlflow.log_params({
                    "model": "XGBoost + TargetEncoder",
                    "max_depth": max_depth,
                    "n_estimators": n_estimators,
                    "learning_rate": lr
                })
                
                # Log metrics
                mlflow.log_metric("MAE", mae)
                mlflow.log_metric("R2", r2)
                
                # Log model
                model_filename = f"flight_model_depth{max_depth}_lr{lr}.pkl"
                mlflow.sklearn.log_model(model, "model")
                joblib.dump(model, os.path.join(MODEL_DIR, model_filename))
                
                # Save native XGBoost model
                model.named_steps["xgb"].get_booster().save_model(
                    os.path.join(MODEL_DIR, f"flight_model_depth{max_depth}_lr{lr}.json")
                )

                print(f"✅ Run {run_name} done | MAE: {mae:.2f} | R2: {r2:.4f}")

if __name__ == "__main__":
    main()
