from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd

app = Flask(__name__)

# Load the trained model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from POST request
    data = request.form
    age = int(data['age'])
    bmi = float(data['bmi'])
    children = int(data['children'])
    smoker = int(data['smoker'])
    region = int(data['region'])

    # Create DataFrame for the input
    input_df = pd.DataFrame([[age, bmi, children, smoker, region]],
                            columns=['age', 'bmi', 'children', 'smoker', 'region'])
    
    # Predict using the loaded model
    prediction = model.predict(input_df)[0]

    return jsonify({'prediction': prediction})

if __name__ == "__main__":
    app.run(debug=True)