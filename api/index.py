from flask import Flask, request, jsonify
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load your model and scaler
model = joblib.load('api\health_score_model.pkl')
scaler = joblib.load('api\scaler.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    if 'sugarPercentage' not in data or 'bloodPressure' not in data or 'averageTemprature' not in data:
        return jsonify({"error": "Missing required input data"}), 400
    input_features = np.array([[data['sugarPercentage'], data['bloodPressure'], data['averageTemprature']]])
    input_scaled = scaler.transform(input_features)
    prediction = model.predict(input_scaled)
    return jsonify({"predicted_health_condition_score": prediction[0]})

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "API is working!"})

# Vercel does not require app.run()
