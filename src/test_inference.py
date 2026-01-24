from inference import predict_price

# Example input
sample_input = {
    "from": "Delhi",
    "to": "Mumbai",
    "flightType": "Direct",
    "agency": "IndiGo",
    "time": 135,
    "distance": 1100,
    "month": 1
}

predicted_price = predict_price(sample_input)
print(f"Predicted Flight Price: {predicted_price:.2f}")
