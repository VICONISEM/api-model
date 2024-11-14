from flask import Flask, request, jsonify
import numpy as np
import os

app = Flask(__name__)

# Define paths using os.path.join to ensure compatibility across different OS
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
mean_path = os.path.join(BASE_DIR, 'native_scaler_mean.npy')
scale_path = os.path.join(BASE_DIR, 'native_scaler_scale.npy')
coefficients_path = os.path.join(BASE_DIR ,'linear_model_coefficients.npy')
intercept_path = os.path.join(BASE_DIR,'linear_model_intercept.npy')

# Load the model and scaler parameters
try:
    mean_ = np.load(mean_path)
    scale_ = np.load(scale_path)
    coefficients = np.load(coefficients_path)
    intercept = np.load(intercept_path)[0]
except Exception as e:
    print(f"Error loading model files: {e}")
    exit()

# Function to scale the input features
def scale_input(data, mean_, scale_):
    return (data - mean_) / scale_

# Function to predict the health condition score
def predict_health_score(features):
    # Scale the input features
    scaled_features = scale_input(np.array(features), mean_, scale_)
    # Make prediction using the linear regression model
    prediction = np.dot(scaled_features, coefficients) + intercept
    return float(prediction)

# Home route for API
@app.route('/')
def home():
    return "<h1>Health Condition Score Prediction API</h1><p>Use /predict endpoint with POST request to get predictions.</p>"

# Prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    # Get JSON data from request
    data = request.json
    if not data:
        return jsonify({'error': 'Invalid input data. Please provide JSON with keys: sugarPercentage, bloodPressure, averageTemprature.'}), 400

    # Extract the features from the request JSON
    try:
        sugar_percentage = float(data['sugarPercentage'])
        blood_pressure = float(data['bloodPressure'])
        average_temperature = float(data['averageTemprature'])
    except KeyError as e:
        return jsonify({'error': f'Missing key: {str(e)}. Please provide sugarPercentage, bloodPressure, and averageTemprature.'}), 400
    except ValueError:
        return jsonify({'error': 'Invalid input type. Please provide numerical values.'}), 400

    # Prepare the input for prediction
    input_features = [sugar_percentage, blood_pressure, average_temperature]

    try:
        # Make a prediction
        prediction = predict_health_score(input_features)
        return jsonify({'predictedHealthConditionScore': prediction}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Run the Flask app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
