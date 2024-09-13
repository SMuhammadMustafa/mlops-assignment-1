from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np

app = Flask(__name__)
CORS(app)  # Allow CORS for all origins

# Load the trained model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def home():
    return "Welcome to the model API! Use the /predict endpoint to make predictions."

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Log the received request data
        print("Received request data:", request.json)
        
        # Get the features from the request
        data = request.json
        # Expect 8 features
        if len(data['features']) != 8:
            raise ValueError("Expected 8 features, but received {}".format(len(data['features'])))
        
        features = np.array(data['features']).reshape(1, -1)
        
        # Log the features to check if they are correctly formatted
        print("Features array:", features)
        
        # Make prediction
        prediction = model.predict(features)
        
        # Log the prediction result
        print("Prediction result:", prediction)
        
        # Return the prediction as JSON
        return jsonify({'prediction': prediction[0]})
    except Exception as e:
        # Log the exception
        print("Error occurred:", e)
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
