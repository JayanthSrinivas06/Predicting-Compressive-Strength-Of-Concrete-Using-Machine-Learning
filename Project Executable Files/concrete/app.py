from flask import Flask, request, render_template, redirect, url_for
import joblib
import numpy as np

app = Flask(__name__)

# Load the model 
model = joblib.load('Project Executable Files/concrete/cement.pkl')


@app.route('/')
def home():
    return render_template('home.html')

@app.route('/Prediction')
def index1():
    return render_template('index1.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the values from the form
    input_features = [float(x) for x in request.form.values()]
    # Convert features to numpy array and reshape
    features = np.array(input_features).reshape(1, -1)
    
    # Predict the concrete strength
    prediction = model.predict(features)
    return render_template('result2.html', prediction_text=f'Predicted Concrete Strength: {prediction[0]:.2f} MPa')

if __name__ == '__main__':
    app.run(debug=False)
    app.run('0.0.0.0',8080)
