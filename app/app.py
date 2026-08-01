import streamlit as st
import sys
from pathlib import Path
import plotly.graph_objects as go

# --------------------------------------------------
# Add project root to Python path
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from forms import patient_form
from predict import InferencePipeline

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Diabetes Risk Prediction",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------
# Load CSS
# --------------------------------------------------

def load_css():
    css_file = Path(__file__).parent / "styles.css"

    if css_file.exists():
        with open(css_file) as f:
            st.markdown(
                f"<style>{f.read()}</style>",
                unsafe_allow_html=True,
            )

load_css()

# --------------------------------------------------
# Load ML Pipeline
# --------------------------------------------------

@st.cache_resource
def load_pipeline():
    return InferencePipeline()

pipeline = load_pipeline()

# --------------------------------------------------
# Header
# --------------------------------------------------

st.markdown("""
<div class="hero">

<h1>🩺 Diabetes Risk Prediction System</h1>

<p>
AI-powered clinical decision support system for
early diabetes risk assessment using a
<strong>Calibrated Stacking Ensemble</strong>.
</p>

</div>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:

    st.info(
        """
### 🎯 Purpose

Estimate diabetes risk from
clinical, lifestyle, and
demographic information.
"""
    )

with col2:

    st.success(
        """
### 🤖 AI Technology

Calibrated Stacking Ensemble

• XGBoost

• LightGBM

• TensorFlow MLP
"""
    )

with col3:

    st.warning(
        """
### ⚠️ Clinical Notice

This application supports
screening only.

It does not replace
professional medical advice.
"""
    )

st.markdown("<br>", unsafe_allow_html=True)

# --------------------------------------------------
# Dashboard Metrics
# --------------------------------------------------

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("🎯 Accuracy", "82.8%")

with col2:
    st.metric("⚡ Prediction", "<1 sec")

with col3:
    st.metric("🧠 AI Model", "Stacking Ensemble")

with col4:
    st.metric("🔍 Explainability", "SHAP")

st.divider()

# --------------------------------------------------
# Sidebar
# --------------------------------------------------

with st.sidebar:

    st.title("🩺 About")

    st.markdown("---")

    st.markdown("### 🤖 Model")
    st.write("Calibrated Stacking Ensemble")

    st.markdown("### 📚 Base Models")
    st.write("""
- XGBoost
- LightGBM
- TensorFlow MLP
""")

    st.markdown("### 📈 Calibration")
    st.write("Isotonic Regression")

    st.markdown("### 🔍 Explainability")
    st.write("SHAP")

    st.markdown("---")

    st.info(
        "This application is intended for educational and screening purposes only."
    )

    st.markdown("---")

    st.markdown("### 👩‍💻 Developer")
    st.write("Serah Ann Shiju")

    st.caption("Version 1.0")

# --------------------------------------------------
# Patient Form
# --------------------------------------------------

submitted, patient_data = patient_form()

# --------------------------------------------------
# Prediction
# --------------------------------------------------

if submitted:

    try:

        with st.spinner("Running AI prediction..."):

            result = pipeline.predict_single(patient_data)

    except Exception as e:

        st.error(f"Prediction failed: {e}")
        st.stop()

    probability = result["probability"] * 100
    threshold = result["threshold"] * 100
    label = result["risk_label"]

    st.markdown("---")

    st.subheader("🩺 Prediction Result")

    # --------------------------------------------------
    # Risk Status
    # --------------------------------------------------

    if result["prediction"] == 1:

        st.markdown(f"""
    <div class="risk-high">

    <div class="risk-title">
    🔴 HIGH RISK
    </div>

    <div class="risk-text">

    <b>Prediction:</b> {label}<br><br>

    <b>Estimated Probability:</b> {probability:.2f}%<br>

    <b>Decision Threshold:</b> {threshold:.2f}%

    </div>

    </div>
    """, unsafe_allow_html=True)

    else:

        st.markdown(f"""
    <div class="risk-low">

    <div class="risk-title">
    🟢 LOW RISK
    </div>

    <div class="risk-text">

    <b>Prediction:</b> {label}<br><br>

    <b>Estimated Probability:</b> {probability:.2f}%<br>

    <b>Decision Threshold:</b> {threshold:.2f}%

    </div>

    </div>
    """, unsafe_allow_html=True)

    # --------------------------------------------------
    # Metrics
    # --------------------------------------------------

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Risk Probability",
            f"{probability:.2f}%"
        )

    with col2:

        st.metric(
            "Decision Threshold",
            f"{threshold:.2f}%"
        )

    with col3:

        st.metric(
            "Prediction",
            label
        )

    st.progress(probability / 100)
    # --------------------------------------------------
    # Probability Gauge
    # --------------------------------------------------

    st.markdown("### 📊 Risk Probability Gauge")

    gauge = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probability,
            number={
                "suffix": "%",
                "font": {"size": 36}
            },
            title={
                "text": "Predicted Diabetes Risk",
                "font": {"size": 22}
            },
            gauge={
                "axis": {
                    "range": [0, 100]
                },
                "bar": {
                    "color": "#1565C0"
                },
                "steps": [
                    {
                        "range": [0, 30],
                        "color": "#C8E6C9"
                    },
                    {
                        "range": [30, 60],
                        "color": "#FFF9C4"
                    },
                    {
                        "range": [60, 100],
                        "color": "#FFCDD2"
                    }
                ],
                "threshold": {
                    "line": {
                        "color": "red",
                        "width": 4
                    },
                    "value": threshold
                }
            }
        )
    )

    gauge.update_layout(
        height=350,
        margin=dict(l=30, r=30, t=50, b=20)
    )

    st.plotly_chart(
        gauge,
        use_container_width=True
    )

    # --------------------------------------------------
    # Recommendation
    # --------------------------------------------------

    if result["prediction"] == 1:

        st.warning(
            """
### Recommendation

The patient is predicted to be at **high risk of diabetes**.

It is recommended to consult a healthcare professional for
further clinical evaluation and diagnostic testing.

This application is intended to support screening and should
not replace professional medical advice.
"""
        )

    else:

        st.info(
            """
### Recommendation

The patient is predicted to be at **low risk of diabetes**.

Continue maintaining a healthy lifestyle including regular
exercise, balanced nutrition, and periodic health check-ups.

This application is intended to support screening and should
not replace professional medical advice.
"""
        )

    # --------------------------------------------------
    # Patient Summary
    # --------------------------------------------------

    st.markdown("### 📋 Patient Summary")

    summary_col1, summary_col2 = st.columns(2)

    with summary_col1:

        st.markdown(f"""
- **Age:** {patient_data["age"]} years
- **Gender:** {patient_data["gender"]}
- **BMI:** {patient_data["bmi"]}
- **Ethnicity:** {patient_data["ethnicity"]}
- **Smoking Status:** {patient_data["smoking_status"]}
""")

    with summary_col2:

        st.markdown(f"""
- **Systolic BP:** {patient_data["systolic_bp"]} mmHg
- **Diastolic BP:** {patient_data["diastolic_bp"]} mmHg
- **Heart Rate:** {patient_data["heart_rate"]} bpm
- **Total Cholesterol:** {patient_data["cholesterol_total"]} mg/dL
- **Physical Activity:** {patient_data["physical_activity_minutes_per_week"]} min/week
""")

    # --------------------------------------------------
    # Download Prediction Report
    # --------------------------------------------------

    import pandas as pd

    report = pd.DataFrame(
        {
            "Prediction": [label],
            "Risk Probability (%)": [round(probability, 2)],
            "Decision Threshold (%)": [round(threshold, 2)],
            "Gender": [patient_data["gender"]],
            "Age": [patient_data["age"]],
            "BMI": [patient_data["bmi"]],
            "Systolic BP": [patient_data["systolic_bp"]],
            "Diastolic BP": [patient_data["diastolic_bp"]],
        }
    )

    st.download_button(
        label="📥 Download Prediction Report",
        data=report.to_csv(index=False),
        file_name="prediction_report.csv",
        mime="text/csv",
        use_container_width=True,
    )

    # --------------------------------------------------
    # About the Model
    # --------------------------------------------------

    with st.expander("ℹ️ About this AI Model"):

        st.markdown(
            """
### Model Architecture

This application uses a **Calibrated Stacking Ensemble**.

#### Base Models

- XGBoost
- LightGBM
- TensorFlow MLP

#### Meta Learner

- Logistic Regression

#### Calibration

- Isotonic Regression

#### Explainability

- SHAP

The model predicts diabetes risk using demographic,
lifestyle, and clinical information.
"""
        )

    # --------------------------------------------------
    # Confidence Indicator
    # --------------------------------------------------

    st.markdown("### 📈 Prediction Confidence")

    if probability >= 80:

        st.success(
            "The model is highly confident in this prediction."
        )

    elif probability >= 60:

        st.info(
            "The model shows moderate confidence in this prediction."
        )

    else:

        st.warning(
            "The prediction is close to the calibrated decision threshold. Interpret the result with clinical judgment."
        )
# --------------------------------------------------
# Footer
# --------------------------------------------------

st.markdown("---")

st.markdown(
"""
<div style="text-align:center; padding:20px 0;">

<h4> Diabetes Risk Prediction System</h4>

<p>
Developed by <b>Serah Ann Shiju</b> •
Version 1.0 •
© 2026
</p>

<p style="color:gray;">
Built with Streamlit • Scikit-learn • TensorFlow • XGBoost • LightGBM
</p>

</div>
""",
unsafe_allow_html=True
)