from flask import Flask, jsonify, request
from flask_restful import Api, Resource
from flask_cors import CORS
import pickle
import numpy as np
import os
import joblib

# Define Native Standard Scaler
class NativeStandardScaler:
    def __init__(self):
        self.mean_ = None
        self.scale_ = None

    def fit(self, X):
        self.mean_ = np.mean(X, axis=0)
        self.scale_ = np.std(X, axis=0)

    def transform(self, X):
        return (X - self.mean_) / self.scale_

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

# Native Decision Tree for Regression
class NativeDecisionTree:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.tree = None

    def fit(self, X, y, depth=0):
        if len(set(y)) == 1 or depth == self.max_depth:
            self.tree = np.mean(y)
            return

        best_split = self.best_split(X, y)
        if best_split['feature_index'] is None:
            self.tree = np.mean(y)
            return
        
        left_mask = X[:, best_split['feature_index']] <= best_split['threshold']
        right_mask = ~left_mask

        self.tree = {
            'feature_index': best_split['feature_index'],
            'threshold': best_split['threshold'],
            'left': NativeDecisionTree(max_depth=self.max_depth),
            'right': NativeDecisionTree(max_depth=self.max_depth)
        }

        self.tree['left'].fit(X[left_mask], y[left_mask], depth + 1)
        self.tree['right'].fit(X[right_mask], y[right_mask], depth + 1)

    def best_split(self, X, y):
        best_feature, best_threshold, best_score = None, None, float('inf')
        n_samples, n_features = X.shape

        for feature_index in range(n_features):
            thresholds = np.unique(X[:, feature_index])
            for threshold in thresholds:
                left_mask = X[:, feature_index] <= threshold
                right_mask = ~left_mask
                left_y, right_y = y[left_mask], y[right_mask]

                if len(left_y) == 0 or len(right_y) == 0:
                    continue

                mse = self.mse(left_y) + self.mse(right_y)
                if mse < best_score:
                    best_feature, best_threshold, best_score = feature_index, threshold, mse

        return {'feature_index': best_feature, 'threshold': best_threshold}

    @staticmethod
    def mse(y):
        return np.var(y) * len(y)

    def predict(self, X):
        if isinstance(self.tree, (float, int)):
            return self.tree
        else:
            feature_index = self.tree['feature_index']
            threshold = self.tree['threshold']
            if X[feature_index] <= threshold:
                return self.tree['left'].predict(X)
            else:
                return self.tree['right'].predict(X)

# Native Random Forest Regressor
class NativeRandomForestRegressor:
    def __init__(self, n_estimators=100, max_depth=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.trees = []

    def fit(self, X, y):
        n_samples = X.shape[0]
        self.trees = []
        for _ in range(self.n_estimators):
            tree = NativeDecisionTree(max_depth=self.max_depth)
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            X_sample, y_sample = X[indices], y[indices]
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def predict(self, X):
        predictions = np.array([tree.predict(x) for x in X for tree in self.trees])
        predictions = predictions.reshape(len(X), -1)
        return np.mean(predictions, axis=1)

# Initialize Flask App


# Ensure classes are registered for loading
globals()['NativeRandomForestRegressor'] = NativeRandomForestRegressor
globals()['NativeDecisionTree'] = NativeDecisionTree
globals()['NativeStandardScaler'] = NativeStandardScaler

# Load Model and Scaler
app = Flask(__name__)
api = Api(app)
CORS(app)

# Define paths for model and scaler
model_path = 'api/native_health_score_model.pkl'
scaler_path = 'api/native_scaler.pkl'

# Initialize variables for model and scaler


# Load the model
if os.path.exists(model_path):
    try:
        model = joblib.load(model_path)
        print(f"Model loaded successfully from {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
else:
    print(f"Model file not found at {model_path}")

# Load the scaler
if os.path.exists(scaler_path):
    try:
        scaler = joblib.load(scaler_path)
        print(f"Scaler loaded successfully from {scaler_path}")
    except Exception as e:
        print(f"Error loading scaler: {e}")
else:
    print(f"Scaler file not found at {scaler_path}")

# Define HealthScore API Resource
class HealthScore(Resource):
    def post(self):
        try:
            # Validate that model and scaler are loaded
            if model is None:
                return {"error": "Model is not loaded"}, 500
            if scaler is None:
                return {"error": "Scaler is not loaded"}, 500

            # Get JSON input
            data = request.get_json(force=True)

            # Validate input data
            if not all(k in data for k in ('sugarPercentage', 'bloodPressure', 'averageTemprature')):
                return {"error": "Missing required input data: sugarPercentage, bloodPressure, averageTemprature"}, 400

            # Extract input features and convert to numpy array
            input_features = np.array([[data['sugarPercentage'], 
                                        data['bloodPressure'], 
                                        data['averageTemprature']]])
            
            # Scale input features using the loaded scaler
            try:
                input_scaled = scaler.transform(input_features)
            except Exception as e:
                return {"error": f"Failed to scale input features: {e}"}, 500

            # Predict using the loaded model
            try:
                prediction = model.predict(input_scaled)
            except Exception as e:
                return {"error": f"Failed to make prediction: {e}"}, 500

            # Return the prediction result
            response = {
                "predicted_health_condition_score": float(prediction[0])
            }
            return response, 200

        except Exception as e:
            # Handle any unexpected errors
            return {"error": str(e)}, 500

# Define Health Resource for API status check
class Health(Resource):
    def get(self):
        return jsonify({"status": "API is working!"})

# Register API resources
api.add_resource(HealthScore, '/predict')
api.add_resource(Health, '/health')

# Run the application
if __name__ == '__main__':
    app.run(debug=True)

