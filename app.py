from flask import Flask, request, jsonify, render_template_string
import joblib
import pandas as pd

app = Flask(__name__)

model = joblib.load('healthcare_readmission_model.pkl')
model_columns = joblib.load('model_columns.pkl')

FORM_HTML = """
<!DOCTYPE html>
<html>
<head><title>Readmission Risk Predictor</title>
<style>
body { font-family: Arial; max-width: 600px; margin: 40px auto; padding: 20px; }
label { display: block; margin-top: 12px; font-weight: bold; }
input, select { width: 100%; padding: 8px; margin-top: 4px; }
button { margin-top: 20px; padding: 10px 20px; background: #2c7be5; color: white; border: none; cursor: pointer; }
.result { margin-top: 20px; padding: 15px; background: #f0f0f0; font-size: 18px; }
.disclaimer { font-size: 12px; color: #888; margin-top: 30px; }
</style>
</head>
<body>
<h2>Hospital Readmission Risk Predictor</h2>
<form method="POST">
  <label>Age Group</label>
  <select name="age">
    <option value="[0-10)">0-10</option><option value="[10-20)">10-20</option>
    <option value="[20-30)">20-30</option><option value="[30-40)">30-40</option>
    <option value="[40-50)">40-50</option><option value="[50-60)" selected>50-60</option>
    <option value="[60-70)">60-70</option><option value="[70-80)">70-80</option>
    <option value="[80-90)">80-90</option><option value="[90-100)">90-100</option>
  </select>

  <label>Time in Hospital (days)</label>
  <input type="number" name="time_in_hospital" value="3" min="1" max="14">

  <label>Number of Lab Procedures</label>
  <input type="number" name="num_lab_procedures" value="44">

  <label>Number of Medications</label>
  <input type="number" name="num_medications" value="16">

  <label>Prior Inpatient Visits</label>
  <input type="number" name="number_inpatient" value="0">

  <label>Prior Emergency Visits</label>
  <input type="number" name="number_emergency" value="0">

  <label>Number of Diagnoses</label>
  <input type="number" name="number_diagnoses" value="7">

  <label>Primary Diagnosis Category</label>
  <select name="diag_1_cat">
    <option value="Circulatory">Circulatory</option><option value="Respiratory">Respiratory</option>
    <option value="Digestive">Digestive</option><option value="Diabetes">Diabetes</option>
    <option value="Injury">Injury</option><option value="Musculoskeletal">Musculoskeletal</option>
    <option value="Genitourinary">Genitourinary</option><option value="Neoplasms">Neoplasms</option>
    <option value="Other">Other</option>
  </select>

  <label>On Diabetes Medication?</label>
  <select name="diabetesMed"><option value="Yes">Yes</option><option value="No">No</option></select>

  <button type="submit">Predict Risk</button>
</form>
{% if result %}
<div class="result"><b>Prediction:</b> {{ result }} (Probability: {{ prob }})</div>
{% endif %}
<div class="disclaimer">⚠️ This is a portfolio/learning project only. Not a diagnostic tool. Not for real clinical use.</div>
</body>
</html>
"""

@app.route('/', methods=['GET', 'POST'])
def home():
    result = None
    prob = None
    if request.method == 'POST':
        data = {
            'age': request.form['age'],
            'time_in_hospital': int(request.form['time_in_hospital']),
            'num_lab_procedures': int(request.form['num_lab_procedures']),
            'num_medications': int(request.form['num_medications']),
            'number_inpatient': int(request.form['number_inpatient']),
            'number_emergency': int(request.form['number_emergency']),
            'number_diagnoses': int(request.form['number_diagnoses']),
            'diag_1_cat': request.form['diag_1_cat'],
            'diabetesMed': request.form['diabetesMed'],
        }
        input_df = pd.DataFrame([data])
        input_encoded = pd.get_dummies(input_df)
        input_final = input_encoded.reindex(columns=model_columns, fill_value=0)

        proba = model.predict_proba(input_final)[:, 1][0]
        result = "High Risk" if proba >= 0.5 else "Low Risk"
        prob = round(float(proba), 4)

    return render_template_string(FORM_HTML, result=result, prob=prob)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    input_df = pd.DataFrame([data])
    input_df = input_df.reindex(columns=model_columns, fill_value=0)
    proba = model.predict_proba(input_df)[:, 1][0]
    prediction = "High Risk" if proba >= 0.5 else "Low Risk"
    return jsonify({"readmission_risk": prediction, "probability": round(float(proba), 4)})

if __name__ == '__main__':
    app.run(debug=False)
