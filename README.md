# Hospital Readmission Risk Prediction

Predicting the risk of a diabetic patient being readmitted to hospital within 30 days of discharge, using real de-identified clinical records from 130 US hospitals (1999–2008).

> ⚠️ **Disclaimer:** This is a personal portfolio / learning project built to demonstrate an end-to-end machine learning workflow. It is **not** a diagnostic or clinical decision-making tool, has not been validated by any medical or regulatory body, and must not be used to inform real patient care.

---

## Problem Statement

Hospitals in the US face financial penalties under CMS (Centers for Medicare & Medicaid Services) rules when patients are readmitted within 30 days of discharge. Early identification of high-risk patients allows hospitals to plan better follow-up care and potentially reduce avoidable readmissions and costs.

This project builds a binary classification model to predict whether a diabetic patient is likely to be **readmitted within 30 days** of discharge, based on their admission, treatment, and diagnosis information.

## Dataset

- **Source:** [UCI Machine Learning Repository — Diabetes 130-US Hospitals for Years 1999–2008](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008)
- **Size:** ~101,766 patient encounters, 50 original features
- **Target:** `readmitted` — recoded to binary (`1` = readmitted within 30 days, `0` = otherwise). Original class distribution: ~11% positive class (imbalanced).
- The dataset contains only de-identified, publicly available records. No real-time or personally identifiable patient data was used at any stage.

## Ethics & Privacy Considerations

Healthcare data carries unique responsibilities, even when public and de-identified:

- **Not a diagnostic tool.** The model outputs a statistical risk score, not a medical judgment. Any real-world deployment would require clinical validation, regulatory approval (e.g., FDA/CE processes), and oversight by qualified healthcare professionals.
- **Fairness & bias.** The dataset includes sensitive attributes (age, race, gender). Models trained on historical healthcare data can inherit and amplify existing biases in care patterns. This project does not perform a full fairness audit, but this is flagged as essential future work before any real use.
- **Data minimization.** Only fields relevant to the prediction task were retained; features like `weight` and `payer_code` were dropped early due to being mostly missing and not clinically essential to the core prediction.
- **No re-identification risk.** The dataset is already de-identified at the source (UCI/Health Facts national database); no additional patient information was collected or inferred.

## Methodology

### 1. Exploratory Data Analysis
- Identified and handled missing data (`weight`, `max_glu_serum`, `payer_code` dropped due to >40–97% missingness; `race`, `medical_specialty` imputed as "Unknown")
- Found strong class imbalance in the target (~11% positive class)
- Key EDA insight: readmission rate increases steadily with the number of prior inpatient visits — the single strongest behavioral signal in the data

### 2. Feature Engineering
- Mapped raw ICD-9 diagnosis codes (700+ unique values across `diag_1/2/3`) into 9 clinically meaningful categories (Circulatory, Respiratory, Digestive, Diabetes, Injury, Musculoskeletal, Genitourinary, Neoplasms, Other)
- Engineered `A1Ctest_done` flag from the heavily-missing `A1Cresult` column
- Simplified `medical_specialty` (73 unique values) to the top 10 specialties + "Other"
- Removed zero-variance columns (`examide`, `citoglipton`) that carried no signal
- One-hot encoded all categorical variables → 116-feature model-ready dataset

### 3. Modeling
Trained and compared multiple classifiers on an 80/20 stratified train-test split, with class imbalance handled via `class_weight='balanced'` / `scale_pos_weight`:

| Model | ROC-AUC |
|---|---|
| Logistic Regression | 0.650 |
| Random Forest | 0.664 |
| XGBoost (default) | 0.677 |
| **XGBoost (tuned via RandomizedSearchCV)** | **0.684** |

The tuned XGBoost model was selected as the final model.

### 4. Feature Importance
Top predictors (tuned XGBoost):
1. `number_inpatient` — prior inpatient visit count
2. `discharge_disposition_id`
3. `number_emergency`
4. `diag_1_cat_Musculoskeletal`
5. `number_diagnoses`

This aligns with the EDA finding that prior hospital utilization is the strongest behavioral predictor of readmission risk — a result consistent with published clinical research on this dataset.

## Results

- Final Test ROC-AUC: **0.684**
- The model prioritizes recall on the minority (high-risk) class to reduce missed at-risk patients, given the higher cost of a false negative in a healthcare screening context

## Tech Stack

Python · pandas · NumPy · scikit-learn · XGBoost · Flask · Gunicorn · Render (deployment)

## Live Demo

The model is deployed as a REST API:
🔗 **https://healthcare-readmission-project.onrender.com**

Example request:
```python
import requests

response = requests.post(
    "https://healthcare-readmission-project.onrender.com/predict",
    json={ "time_in_hospital": 3, "num_lab_procedures": 44, ... }
)
print(response.json())
# {'readmission_risk': 'High Risk', 'probability': 0.52}
```

> Note: hosted on Render's free tier — the first request after inactivity may take up to ~50 seconds to respond.

## Limitations & Future Work

- ROC-AUC of 0.684 reflects the genuine difficulty of predicting human health outcomes from administrative/clinical records — behavior and biology introduce noise that's harder to model than, say, financial or purchasing data
- No formal fairness/bias audit was performed across demographic subgroups — a necessary next step before any real-world consideration
- Could explore SHAP-based interpretability, additional feature engineering on medication change patterns, and threshold tuning for different cost-sensitivity scenarios

---

*Built as part of a 3-project data science portfolio spanning fintech, e-commerce, and healthcare domains.*
