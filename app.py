#............... will add flask code ...............
from flask import Flask, render_template, request
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)


#............. Load Model .............
with open('healthCareStrokeCheck.pkl', 'rb') as file:
    model = pickle.load(file)
@app.route('/')
def home():
    return render_template('healthCare_deplyment.html')

@app.route('/predict', methods = ['POST'])
def predict_stroke():
    gender = int (request.form['gender'])
    age =   float (request.form['age'])
    hypertension = int (request.form['hypertension'])
    heart_disease = int (request.form['heart_disease'])
    ever_married = int (request.form['ever_married'])
    work_type = int (request.form['work_type'])
    residence_type = int (request.form['residence_type'])
    avg_glucose_level = float (request.form['avg_glucose_level'])
    bmi = float (request.form['bmi'])
    smoking_status = int (request.form['smoking_status'])

    x_new = pd.DataFrame([{
        "gender": gender,
        "age": age,
        "hypertension": hypertension ,
        "heart_disease": heart_disease,
        "ever_married": ever_married,
        "work_type": work_type,
        "Residence_type": residence_type,
        "avg_glucose_level": avg_glucose_level,
        "bmi": bmi,
        "smoking_status": smoking_status
    }])
    #prediction = model.predict(x_new)

    # Get Binary Prediction
    prediction = model.predict(x_new)[0]

    # Get Probability Score (Likelihood of class 1 / Stroke)
    # model.predict_proba returns [ [prob_no_stroke, prob_stroke] ]
    probability = model.predict_proba(x_new)[0][1]
    prob_percent = round(probability * 100, 2)

    result_Stroke = "Stroke Risk" if prediction == 1 else "Low Risk"

    return render_template('healthCare_deplyment.html',
                           result=result_Stroke,
                           prob=prob_percent)


    '''return render_template(
        'healthCare_deplyment.html', result = "Stroke" if prediction == 1 else "Not Stroke"
    )'''


if __name__ == "__main__":
    app.run(debug=True)
