# Diabetes Risk Prediction with a Calibrated Stacking Ensemble

A production-ready machine learning project for predicting diabetes risk using a calibrated stacking ensemble. This project combines multiple machine learning models with probability calibration to improve predictive performance and reliability.

---

## Project Overview

This project implements a complete end-to-end machine learning pipeline for diabetes risk prediction, including:

- Data preprocessing
- Feature engineering
- Hyperparameter optimization using Optuna
- Stacking ensemble learning
- Probability calibration
- Explainability using SHAP
- Model serialization for deployment

The project is organized for reproducibility and future deployment using Streamlit or FastAPI.

---

## Features

- XGBoost
- LightGBM
- Multi-layer Perceptron (MLP)
- Logistic Regression Meta Learner
- Stacking Ensemble
- Isotonic Probability Calibration
- SHAP Explainability
- Production-ready preprocessing pipeline
- Saved deployment artifacts

---

## Project Structure

```
Diabetes-Risk-Prediction/
│
├── data/
├── explainability/
├── models/
├── notebooks/
├── samples/
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Models Used

- XGBoost
- LightGBM
- Multi-layer Perceptron
- Logistic Regression (Meta Learner)

Final prediction is produced using a calibrated stacking ensemble.

---

## Explainability

Model explanations are generated using SHAP.

Generated artifacts include:

- SHAP Summary Plot
- SHAP Feature Importance Plot
- Feature Importance CSV
- SHAP Values
- SHAP Explainer
- Feature Dependence Plots

---

## Installation

Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/Diabetes-Risk-Prediction.git
```

Create a virtual environment

```bash
python -m venv venv
```

Activate the environment

### Windows

```bash
venv\Scripts\activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

## Running the Project

Open the notebooks inside the `notebooks` folder.

Production models are available inside the `models` directory.

---

## Technologies

- Python
- Scikit-learn
- XGBoost
- LightGBM
- Optuna
- SHAP
- NumPy
- Pandas
- Matplotlib
- Joblib

---

## Future Improvements

- Streamlit Web Application
- FastAPI REST API
- Docker Support
- Cloud Deployment
- Model Monitoring

---

## License

This project is licensed under the MIT License.