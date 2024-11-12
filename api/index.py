import joblib
import numpy as np
from flask import Flask, request, jsonify
from sklearn.preprocessing import StandardScaler

# Initialize Flask app
app = Flask(__name__)

# Load the pre-trained model and scaler
model = joblib.load('health_score_model.pkl')
scaler = joblib.load('scaler.pkl')  # You need to save and load the scaler separately

# Define the prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    # Get the input data from the POST request
    data = request.get_json(force=True)

    # Check if the necessary data keys are provided
    if 'sugarPercentage' not in data or 'bloodPressure' not in data or 'averageTemprature' not in data:
        return jsonify({"error": "Missing required input data"}), 400

    # Extract features from the data
    sugar_percentage = data['sugarPercentage']
    blood_pressure = data['bloodPressure']
    average_temperature = data['averageTemprature']

    # Scale the input data
    input_features = np.array([[sugar_percentage, blood_pressure, average_temperature]])
    input_scaled = scaler.transform(input_features)

    # Make a prediction using the loaded model
    prediction = model.predict(input_scaled)

    # Return the prediction as a JSON response
    return jsonify({"predicted_health_condition_score": prediction[0]})

# Endpoint to check if the API is working
@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "API is working!"})

if __name__ == '__main__':
    # Run the Flask app on port 5000
    app.run(debug=True)
