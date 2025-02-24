from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Allow frontend to access API

# Load the trained model and scaler
voting_soft_classifier = joblib.load("voting_soft_classifier.pkl")
scaler = joblib.load("scaler.pkl")

# Define the prediction endpoint
@app.route("/predict", methods=["POST"])
def predict_cad():
    try:
        data = request.json
        features = np.array([[data["age"], data["sex"], data["cp"], data["trestbps"],
                             data["chol"], data["fbs"], data["restecg"], data["thalach"],
                             data["exang"], data["oldpeak"], data["slope"]]])
        scaled_features = scaler.transform(features)
        predicted_probability = voting_soft_classifier.predict_proba(scaled_features)[0][1] * 100
        prediction = voting_soft_classifier.predict(scaled_features)[0]
        predicted_class = "Positive for CAD" if prediction == 1 else "Negative for CAD"
        return jsonify({"predicted_class": predicted_class, "predicted_probability": f"{predicted_probability:.2f}%"})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
