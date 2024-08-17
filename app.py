import pickle
import numpy as np
from flask import Flask, request, render_template, jsonify

app = Flask(__name__, template_folder="templates")

# Load the trained model
try:
    model = pickle.load(open('model.pkl', 'rb'))
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")

@app.route('/')
def home():
    return render_template('home.html')  # Ensure this file exists in the 'templates' directory

@app.route('/classify', methods=['POST', 'GET'])
def classify():
    try:
        # Retrieve the query parameters from the request
        slen = float(request.args.get('slen', 0))
        swid = float(request.args.get('swid', 0))
        plen = float(request.args.get('plen', 0))
        pwid = float(request.args.get('pwid', 0))

        # Create a new data point for prediction
        data = [[slen, swid, plen, pwid]]

        # Perform prediction using the trained classifier
        prediction = model.predict(data)
        output = str(prediction[0])

        return render_template('result.html', prediction_text=f'The Flower is {output}')
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
