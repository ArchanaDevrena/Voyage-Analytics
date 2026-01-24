import joblib
import xgboost as xgb

# Load the pickle model from correct path
model = joblib.load("models/flight/flight_model.pkl")

# If it is a pipeline, extract the XGBoost model
if hasattr(model, "named_steps"):
    for step in model.named_steps.values():
        if isinstance(step, xgb.XGBRegressor):
            xgb_model = step
            break
else:
    xgb_model = model

# Save in XGBoost native format
xgb_model.get_booster().save_model("models/flight/flight_model.json")

print("flight_model.json created successfully!")
