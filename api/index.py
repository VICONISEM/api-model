from flask import Flask, jsonify, request
from flask_restful import Api, Resource
from flask_cors import CORS
import pickle
import numpy as np

app = Flask(__name__)
api = Api(app)
CORS(app)

# Load Native Model and Scaler using Pickle
with open('native_health_score_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('native_scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Define a HealthScore Resource
class HealthScore(Resource):
    def post(self):
        # Get JSON input
        data = request.get_json(force=True)

        # Check for required input data
        if 'sugarPercentage' not in data or 'bloodPressure' not in data or 'averageTemprature' not in data:
            return jsonify({"error": "Missing required input data"}), 400

        try:
            # Extract input features
            input_features = np.array([[data['sugarPercentage'], 
                                        data['bloodPressure'], 
                                        data['averageTemprature']]])

            # Scale input features using the native scaler
            input_scaled = scaler.transform(input_features)

            # Predict using the native random forest model
            prediction = model.predict(input_scaled)

            # Return the prediction
            return jsonify({"predicted_health_condition_score": float(prediction[0])})

        except Exception as e:
            # Handle any errors
            return jsonify({"error": str(e)}), 500

# Define a Health Resource for API status check
class Health(Resource):
    def get(self):
        return jsonify({"status": "API is working!"})

# Add API resources
api.add_resource(HealthScore, '/predict')
api.add_resource(Health, '/health')

# Run the application
if __name__ == '__main__':
    app.run(debug=True)
