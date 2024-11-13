from flask import Flask, jsonify, request
from flask_restful import Api, Resource
from flask_cors import CORS
import pickle
import numpy as np
import os

app = Flask(__name__)
api = Api(app)
CORS(app)

# model_path = os.path.join(os.path.dirname(__file__), 'native_health_score_model.pkl')
# scaler_path = os.path.join(os.path.dirname(__file__), 'native_scaler.pkl')

# # Load using Pickle
# with open(model_path, 'rb') as model_file:
#     model = pickle.load(model_file)

# with open(scaler_path, 'rb') as scaler_file:
#     scaler = pickle.load(scaler_file)
# # Load Native Model and Scaler using Pickle


# Define a HealthScore Resource
# class HealthScore(Resource):
    # def post(self):
    #     # Get JSON input
    #     data = request.get_json(force=True)

    #     # Check for required input data
    #     if 'sugarPercentage' not in data or 'bloodPressure' not in data or 'averageTemprature' not in data:
    #         return jsonify({"error": "Missing required input data"}), 400

    #     try:
    #         # Extract input features
    #         input_features = np.array([[data['sugarPercentage'], 
    #                                     data['bloodPressure'], 
    #                                     data['averageTemprature']]])

    #         # Scale input features using the native scaler
    #         input_scaled = scaler.transform(input_features)

    #         # Predict using the native random forest model
    #         prediction = model.predict(input_scaled)

    #         # Return the prediction
    #         return jsonify({"predicted_health_condition_score": float(prediction[0])})

    #     except Exception as e:
    #         # Handle any errors
    #         return jsonify({"error": str(e)}), 500

# Define a Health Resource for API status check
class Health(Resource):
    def get(self):
        return jsonify({"status": "API is working!"})

# Add API resources
#api.add_resource(HealthScore, '/predict')
api.add_resource(Health, '/health')

# Run the application
if __name__ == '__main__':
    app.run(debug=True)
