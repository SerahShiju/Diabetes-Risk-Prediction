import streamlit as st
import sys
from pathlib import Path

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
    page_title="Diabetes Risk Prediction System",
    page_icon="🌤️",
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
    <div class="eyebrow">Health Screening Tool</div>
    <h1>Diabetes Risk Prediction System</h1>
    <p>
        A clinical decision-support tool that estimates diabetes risk
        from everyday health, lifestyle, and clinical measurements
        using a calibrated stacking ensemble model.
    </p>
</div>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.info(
        "**Purpose**\n\nEstimate diabetes risk from clinical, "
        "lifestyle, and demographic information."
    )

with col2:
    st.info(
        "**Model**\n\nCalibrated stacking ensemble combining "
        "XGBoost, LightGBM, and a neural network learner."
    )

with col3:
    st.warning(
        "**Clinical Notice**\n\nThis tool supports screening only "
        "and does not replace professional medical advice."
    )

st.markdown("<br>", unsafe_allow_html=True)

# --------------------------------------------------
# Dashboard Metrics
# --------------------------------------------------

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Recall", "94.3%")

with col2:
    st.metric("Response Time", "< 1 sec")

with col3:
    st.metric("Model Type", "Stacking Ensemble")

with col4:
    st.metric("Explainability", "SHAP")

st.divider()

# --------------------------------------------------
# Sidebar
# --------------------------------------------------

with st.sidebar:

    st.markdown("### About Diabetes Risk Prediction System")

    st.markdown("---")

    st.markdown("**Model**")
    st.write("Calibrated stacking ensemble")

    st.markdown("**Base Models**")
    st.write("XGBoost, LightGBM, Neural Network")

    st.markdown("**Calibration**")
    st.write("Isotonic regression")

    st.markdown("**Explainability**")
    st.write("SHAP")

    st.markdown("---")

    st.info(
        "Intended for educational and screening purposes only."
    )

    st.markdown("---")

    st.markdown("**Developer**")
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

        with st.spinner("Running prediction..."):

            result = pipeline.predict_single(patient_data)

    except Exception as e:

        st.error(f"Prediction failed: {e}")
        st.stop()

    probability = result["probability"] * 100
    threshold = result["threshold"] * 100
    label = result["risk_label"]

    st.markdown("---")

    st.subheader("Prediction Result")

    # --------------------------------------------------
    # Risk Status
    # --------------------------------------------------

    if result["prediction"] == 1:

        st.markdown(f"""
    <div class="risk-high">
        <div class="risk-title">Elevated Risk</div>
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
        <div class="risk-title">Low Risk</div>
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
        st.metric("Risk Probability", f"{probability:.2f}%")

    with col2:
        st.metric("Decision Threshold", f"{threshold:.2f}%")

    with col3:
        st.metric("Prediction", label)

    st.progress(probability / 100)

    # --------------------------------------------------
    # Recommendation
    # --------------------------------------------------

    if result["prediction"] == 1:

        st.warning(
            "**Recommendation**\n\n"
            "The patient is predicted to be at elevated risk of diabetes. "
            "Consulting a healthcare professional for further clinical "
            "evaluation and diagnostic testing is recommended.\n\n"
            "This tool is intended to support screening and should "
            "not replace professional medical advice."
        )

    else:

        st.info(
            "**Recommendation**\n\n"
            "The patient is predicted to be at low risk of diabetes. "
            "Continue maintaining a healthy lifestyle including regular "
            "exercise, balanced nutrition, and periodic check-ups.\n\n"
            "This tool is intended to support screening and should "
            "not replace professional medical advice."
        )

    # --------------------------------------------------
    # Patient Summary
    # --------------------------------------------------

    st.markdown("### Patient Summary")

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
        label="Download Prediction Report",
        data=report.to_csv(index=False),
        file_name="prediction_report.csv",
        mime="text/csv",
        use_container_width=True,
    )

    # --------------------------------------------------
    # About the Model
    # --------------------------------------------------

    with st.expander("About this model"):

        st.markdown(
            "**Model Architecture**\n\n"
            "This application uses a calibrated stacking ensemble.\n\n"
            "**Base Models:** XGBoost, LightGBM, Neural Network\n\n"
            "**Meta Learner:** Logistic Regression\n\n"
            "**Calibration:** Isotonic Regression\n\n"
            "**Explainability:** SHAP\n\n"
            "The model predicts diabetes risk using demographic, "
            "lifestyle, and clinical information."
        )

    # --------------------------------------------------
    # Confidence Indicator
    # --------------------------------------------------

    st.markdown("### Prediction Confidence")

    if probability >= 80:
        st.success("The model is highly confident in this prediction.")

    elif probability >= 60:
        st.info("The model shows moderate confidence in this prediction.")

    else:
        st.warning(
            "The prediction is close to the calibrated decision threshold. "
            "Interpret the result with clinical judgment."
        )

# --------------------------------------------------
# Footer
# --------------------------------------------------

st.markdown("---")

st.markdown(
"""
<div class="app-footer">
    <strong>Diabetes Prediction System</strong> — Diabetes Risk Prediction<br>
    Developed by Serah Ann Shiju · Version 1.0 · © 2026<br>
    Built with Streamlit, Scikit-learn, XGBoost, LightGBM
</div>
""",
unsafe_allow_html=True
)