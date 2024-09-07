from flask import Flask, render_template, request

import pandas as pd
import pickle

app = Flask(__name__, template_folder='frontend')

with open('model/model_trained.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def index():
    return render_template('index.html')
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the form data
        sepal_length = float(request.form['sepal-length'])
        sepal_width = float(request.form['sepal-width'])
        petal_length = float(request.form['petal-length'])
        petal_width = float(request.form['petal-width'])

        # Preprocess the input data
        input_data = pd.DataFrame({
            'sepal.length': [sepal_length],
            'sepal.width': [sepal_width],
            'petal.length': [petal_length],
            'petal.width': [petal_width],
        })

        # Make prediction
        prediction = model.predict(input_data)
        if prediction[0] == 0:
            prediction_label = "Setosa"
        elif prediction[0] == 1:
            prediction_label = "Versicolor"
        else:
            prediction_label = "Virginica"

        return render_template('predict.html', prediction=prediction_label)
    except ValueError:
        return render_template('index.html', error="Please enter valid numeric values.")


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0')
