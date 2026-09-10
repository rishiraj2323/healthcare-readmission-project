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
    {% for val, label in [("[0-10)","0-10"),("[10-20)","10-20"),("[20-30)","20-30"),("[30-40)","30-40"),("[40-50)","40-50"),("[50-60)","50-60"),("[60-70)","60-70"),("[70-80)","70-80"),("[80-90)","80-90"),("[90-100)","90-100")] %}
    <option value="{{ val }}" {% if form_data.age == val %}selected{% endif %}>{{ label }}</option>
    {% endfor %}
  </select>

  <label>Time in Hospital (days)</label>
  <input type="number" name="time_in_hospital" value="{{ form_data.time_in_hospital }}" min="1" max="14">

  <label>Number of Lab Procedures</label>
  <input type="number" name="num_lab_procedures" value="{{ form_data.num_lab_procedures }}">

  <label>Number of Medications</label>
  <input type="number" name="num_medications" value="{{ form_data.num_medications }}">

  <label>Prior Inpatient Visits</label>
  <input type="number" name="number_inpatient" value="{{ form_data.number_inpatient }}">

  <label>Prior Emergency Visits</label>
  <input type="number" name="number_emergency" value="{{ form_data.number_emergency }}">

  <label>Number of Diagnoses</label>
  <input type="number" name="number_diagnoses" value="{{ form_data.number_diagnoses }}">

  <label>Primary Diagnosis Category</label>
  <select name="diag_1_cat">
    {% for val in ["Circulatory","Respiratory","Digestive","Diabetes","Injury","Musculoskeletal","Genitourinary","Neoplasms","Other"] %}
    <option value="{{ val }}" {% if form_data.diag_1_cat == val %}selected{% endif %}>{{ val }}</option>
    {% endfor %}
  </select>

  <label>On Diabetes Medication?</label>
  <select name="diabetesMed">
    <option value="Yes" {% if form_data.diabetesMed == "Yes" %}selected{% endif %}>Yes</option>
    <option value="No" {% if form_data.diabetesMed == "No" %}selected{% endif %}>No</option>
  </select>

  <button type="submit">Predict Risk</button>
</form>
{% if result %}
<div class="result"><b>Prediction:</b> {{ result }} (Probability: {{ prob }})</div>
{% endif %}
<div class="disclaimer">⚠️ This is a portfolio/learning project only. Not a diagnostic tool. Not for real clinical use.</div>
</body>
</html>
"""

DEFAULT_FORM = {
    'age': '[50-60)', 'time_in_hospital': 3, 'num_lab_procedures': 44,
    'num_medications': 16, 'number_inpatient': 0, 'number_emergency': 0,
    'number_diagnoses': 7, 'diag_1_cat': 'Circulatory', 'diabetesMed': 'Yes'
}

@app.route('/', methods=['GET', 'POST'])
def home():
    result = None
    prob = None
    form_data = DEFAULT_FORM.copy()

    if request.method == 'POST':
        form_data = {
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
        input_df = pd.DataFrame([form_data])
        input_encoded = pd.get_dummies(input_df)
        input_final = input_encoded.reindex(columns=model_columns, fill_value=0)

        proba = model.predict_proba(input_final)[:, 1][0]
        result = "High Risk" if proba >= 0.5 else "Low Risk"
        prob = round(float(proba), 4)

    return render_template_string(FORM_HTML, result=result, prob=prob, form_data=form_data)

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
