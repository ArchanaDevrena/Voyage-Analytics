from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import joblib
import xgboost as xgb

app = Flask(__name__)

# ---------------- LOAD MODELS ---------------- #

# Flight Price Prediction Model (loaded using XGBoost JSON)
flight_model = xgb.Booster()
flight_model.load_model("models/flight/flight_model.json")

# Recommendation System Models (loaded using Joblib)
hotel_similarity = joblib.load("models/recommendation/hotel_similarity.pkl")
user_similarity = joblib.load("models/recommendation/user_similarity.pkl")
users_data = joblib.load("models/recommendation/users_data.pkl")

# ---------------- ROUTES ---------------- #

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/flight")
def flight():
    return render_template("flight.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        airline = float(request.form["airline"])
        source = float(request.form["source"])
        destination = float(request.form["destination"])
        stops = int(request.form["stops"])

        # Prepare input for XGBoost
        features = np.array([[airline, source, destination, stops]])
        dtest = xgb.DMatrix(features)

        # Predict price
        price = flight_model.predict(dtest)[0]

        return render_template(
            "flight.html",
            prediction=f"₹ {round(float(price), 2)}"
        )

    except Exception as e:
        print("Prediction Error:", e)
        return render_template(
            "flight.html",
            prediction="Error in prediction. Please check your inputs."
        )


@app.route("/recommend")
def recommend():
    return render_template("recommend.html")


@app.route("/get_recommendation", methods=["POST"])
def get_recommendation():
    try:
        user_id = int(request.form["user"])

        # Find similar users
        similarity_scores = list(enumerate(user_similarity[user_id]))
        similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

        # Take top 5 similar users (excluding itself)
        similarity_scores = similarity_scores[1:6]

        recommended_hotels = []

        for user, score in similarity_scores:
            hotels = users_data[users_data["user_id"] == user]["hotel_name"].values
            for h in hotels:
                if h not in recommended_hotels:
                    recommended_hotels.append(h)

        return render_template(
            "recommend.html",
            recommendations=recommended_hotels[:5]
        )

    except Exception as e:
        print("Recommendation Error:", e)
        return render_template(
            "recommend.html",
            recommendations=["Error in generating recommendations. Please try again."]
        )


# ---------------- RUN APP ---------------- #

if __name__ == "__main__":
    app.run(debug=True)
