from flask import Flask, request, render_template
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)

# Load the trained model (only once)
try:
    model = joblib.load('credit.pkl')
except Exception as e:
    print("Error loading model:", e)  # This will print the error in your terminal
    model = None

@app.route('/')
def home():
    return render_template('index.html', prediction=None)

@app.route('/predict', methods=['POST'])
def predict():
    if not model:
        return render_template('index.html', prediction="Model failed to load.")
    try:
        features = [float(x) for x in request.form.values()]
        features = np.array(features).reshape(1, -1)
        prediction = model.predict(features)[0]
        result = "🔴 Fraudulent Transaction" if prediction == 1 else "🟢 Legitimate Transaction"
        return render_template('index.html', prediction=result)
    except Exception as e:
        return render_template('index.html', prediction="Error: " + str(e))

if __name__ == '__main__':
    app.run(debug=True)