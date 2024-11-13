from flask import Flask, jsonify, request
from flask_restful import Api, Resource
from flask_cors import CORS
import joblib
import numpy as np
# from sklearn.preprocessing import StandardScaler

app = Flask(__name__)
api = Api(app)
CORS(app)

# Load your model and scaler
model = joblib.load('health_score_model.pkl')
scaler = joblib.load('scaler.pkl')

# # Define a HealthScore Resource
# class HealthScore(Resource):
#     def post(self):
#         data = request.get_json(force=True)
#         # Check for required input data
#         if 'sugarPercentage' not in data or 'bloodPressure' not in data or 'averageTemprature' not in data:
#             return jsonify({"error": "Missing required input data"}), 400

#         try:
#             input_features = np.array([[data['sugarPercentage'], data['bloodPressure'], data['averageTemprature']]])
#             input_scaled = scaler.transform(input_features)
#             prediction = model.predict(input_scaled)
#             return jsonify({"predicted_health_condition_score": prediction[0]})
#         except Exception as e:
#             return jsonify({"error": str(e)}), 500

# Define a Health Resource for API status check
class Health(Resource):
    def get(self):
        return jsonify({"status": "API is working!"})


# api.add_resource(HealthScore, '/predict')
api.add_resource(Health, '/health')


