# Diabetes Risk Prediction - Deployment Guide

## Overview

This project deploys a calibrated stacking ensemble for diabetes risk prediction.

The deployment pipeline reproduces the complete machine learning workflow used during model training, including preprocessing, ensemble prediction, probability calibration, and threshold-based classification.

---

# Project Structure

```
Diabetes-Risk-Prediction/
│
├── data/
├── models/
│
│   ├── preprocessor.pkl
│   ├── xgb.pkl
│   ├── lightgbm.pkl
│   ├── mlp.pkl
│   ├── meta.pkl
│   ├── calibrated.pkl
│   ├── threshold.pkl
│   ├── feature_names.pkl
│   ├── target_labels.pkl
│   └── metadata.json
│
├── notebooks/
├── samples/
│   ├── sample_input.csv
│   └── sample_output.csv
│
├── predict.py
├── requirements.txt
├── README.md
└── DEPLOYMENT.md
```

---

# Deployment Pipeline

```
Raw Patient Data
        │
        ▼
Preprocessor
        │
        ▼
42 Engineered Features
        │
 ┌──────┴──────┐
 │             │
 ▼             ▼
XGBoost   LightGBM   MLP
        │
        ▼
Meta Logistic Regression
        │
        ▼
Probability Calibration
        │
        ▼
Decision Threshold
        │
        ▼
Final Prediction
```

---

# Required Files

The following deployment artifacts are required.

| File | Purpose |
|------|----------|
| preprocessor.pkl | Feature preprocessing |
| xgb.pkl | XGBoost model |
| lightgbm.pkl | LightGBM model |
| mlp.pkl | Neural Network |
| meta.pkl | Logistic Regression meta learner |
| calibrated.pkl | Calibrated probability model |
| threshold.pkl | Classification threshold |
| feature_names.pkl | Engineered feature names |
| target_labels.pkl | Prediction labels |
| metadata.json | Project metadata |

---

# Running Predictions

Predict from a CSV file

```bash
python predict.py --input samples/sample_input.csv
```

Save predictions

```bash
python predict.py --input samples/sample_input.csv --output samples/sample_output.csv
```

---

# Input Format

The input CSV should contain the same raw features used during model training.

Example columns include:

- age
- gender
- bmi
- systolic_bp
- diastolic_bp
- cholesterol_total
- hdl_cholesterol
- ldl_cholesterol
- triglycerides
- physical_activity_minutes_per_week
- sleep_hours_per_day
- smoking_status
- alcohol_consumption_per_week
- family_history_diabetes
- hypertension_history
- cardiovascular_history
- diet_score
- waist_to_hip_ratio
- heart_rate
- ethnicity
- education_level
- employment_status
- income_level

---

# Output Format

The generated prediction file contains:

| Column | Description |
|---------|-------------|
| Probability | Calibrated probability |
| Threshold | Decision threshold |
| Prediction | Binary prediction |
| Risk_Label | Human-readable prediction |

---

# Command Line Example

```
python predict.py --input samples/sample_input.csv
```

Example output

```
Patient #1

Probability : 33.95%

Prediction : Non-Diabetic
```

---

# Troubleshooting

## Missing model files

Ensure all deployment artifacts exist inside the models directory.

## Invalid CSV

Verify that all required raw input features are present.

## Environment

Install all required packages

```
pip install -r requirements.txt
```

---

# Author

Serah Ann Shiju

MCA Final Year

Diabetes Risk Prediction with a Calibrated Stacking Ensemble