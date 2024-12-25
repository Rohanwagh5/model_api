# from flask import Flask, request, jsonify
# import joblib
# import numpy as np

# app = Flask(__name__)

# # Load the trained model
# model = joblib.load('knn_model2.joblib')

# @app.route('/', methods=['POST'])
# def predict():
#     # Parse incoming JSON data
#     data = request.json
#     features = np.array([[
#         data['age'], data['sex'], data['cp'], data['trestbps'], data['chol'],
#         data['fbs'], data['restecg'], data['thalach'], data['exang'], data['oldpeak'],
#         data['slope'], data['ca'], data['thal']
#     ]])
#     # Make a prediction using the loaded model
#     prediction = model.predict(features)
#     return jsonify({'prediction': int(prediction[0])})

# if __name__ == '__main__':
#     app.run(debug=True)

from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load('knn_model2.joblib')

# Home route for GET requests
@app.route('/', methods=['GET'])
def home():
    return "Heart Disease Prediction API is running! Use the /predict endpoint to make predictions.", 200

# Predict route for POST requests
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse incoming JSON data
        data = request.json
        features = np.array([[  
            data['age'], data['sex'], data['cp'], data['trestbps'], data['chol'],
            data['fbs'], data['restecg'], data['thalach'], data['exang'], data['oldpeak'],
            data['slope'], data['ca'], data['thal']
        ]])
        # Make a prediction using the loaded model
        prediction = model.predict(features)
        return jsonify({'prediction': int(prediction[0])}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)

