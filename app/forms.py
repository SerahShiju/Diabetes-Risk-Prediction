import streamlit as st


def patient_form():
    """
    Display the patient information form.

    Returns
    -------
    tuple
        (submitted, patient_data)
    """

    st.subheader(" Patient Information")

    with st.form("patient_form"):

        # =====================================================
        # Demographics
        # =====================================================

        st.markdown("### 👤 Demographics")

        col1, col2 = st.columns(2)

        with col1:

            age = st.slider(
                "Age",
                18,
                100,
                50
            )

            gender = st.selectbox(
                "Gender",
                ["Female", "Male"]
            )

            ethnicity = st.selectbox(
                "Ethnicity",
                [
                    "White",
                    "Black",
                    "Asian",
                    "Hispanic",
                    "Other"
                ]
            )

        with col2:

            education_level = st.selectbox(
                "Education Level",
                [
                    "Highschool",
                    "Bachelor",
                    "Master",
                    "PhD"
                ]
            )

            income_level = st.selectbox(
                "Income Level",
                [
                    "Low",
                    "Lower-Middle",
                    "Upper-Middle",
                    "High"
                ]
            )

            employment_status = st.selectbox(
                "Employment Status",
                [
                    "Employed",
                    "Unemployed",
                    "Retired"
                ]
            )

        st.divider()

        # =====================================================
        # Lifestyle
        # =====================================================

        st.markdown("### 🏃 Lifestyle")

        col1, col2 = st.columns(2)

        with col1:

            alcohol = st.slider(
                "Alcohol Consumption (per week)",
                0,
                20,
                2
            )

            physical_activity = st.slider(
                "Physical Activity (minutes/week)",
                0,
                600,
                150
            )

            diet_score = st.slider(
                "Diet Score",
                0.0,
                10.0,
                5.0,
                0.1
            )

        with col2:

            sleep = st.slider(
                "Sleep Hours",
                3.0,
                12.0,
                7.0,
                0.1
            )

            screen_time = st.slider(
                "Screen Time (hours/day)",
                0.0,
                15.0,
                5.0,
                0.1
            )

            smoking = st.selectbox(
                "Smoking Status",
                [
                    "Never",
                    "Former",
                    "Current"
                ]
            )

        st.divider()

        # =====================================================
        # Vital Signs
        # =====================================================

        st.markdown("### Vital Signs")

        col1, col2 = st.columns(2)

        with col1:

            bmi = st.slider(
                "BMI",
                10.0,
                50.0,
                25.0,
                0.1
            )

            waist_ratio = st.slider(
                "Waist-to-Hip Ratio",
                0.50,
                1.50,
                0.90,
                0.01
            )

            systolic = st.slider(
                "Systolic BP",
                80,
                220,
                120
            )

        with col2:

            diastolic = st.slider(
                "Diastolic BP",
                50,
                140,
                80
            )

            heart_rate = st.slider(
                "Heart Rate",
                40,
                150,
                72
            )

        st.divider()

        # =====================================================
        # Laboratory Tests
        # =====================================================

        st.markdown("###  Laboratory Measurements")

        col1, col2 = st.columns(2)

        with col1:

            cholesterol = st.slider(
                "Total Cholesterol",
                100,
                400,
                180
            )

            hdl = st.slider(
                "HDL Cholesterol",
                20,
                100,
                50
            )

        with col2:

            ldl = st.slider(
                "LDL Cholesterol",
                40,
                250,
                100
            )

            triglycerides = st.slider(
                "Triglycerides",
                40,
                400,
                150
            )

        st.divider()

        # =====================================================
        # Medical History
        # =====================================================

        st.markdown("###  Medical History")

        col1, col2, col3 = st.columns(3)

        with col1:

            family_history = st.selectbox(
                "Family History",
                [0, 1],
                format_func=lambda x: "Yes" if x else "No"
            )

        with col2:

            hypertension = st.selectbox(
                "Hypertension",
                [0, 1],
                format_func=lambda x: "Yes" if x else "No"
            )

        with col3:

            cardiovascular = st.selectbox(
                "Cardiovascular Disease",
                [0, 1],
                format_func=lambda x: "Yes" if x else "No"
            )

        submitted = st.form_submit_button(
            "🔍 Predict Diabetes Risk",
            use_container_width=True
        )

    patient_data = {
        "age": age,
        "alcohol_consumption_per_week": alcohol,
        "physical_activity_minutes_per_week": physical_activity,
        "diet_score": diet_score,
        "sleep_hours_per_day": sleep,
        "screen_time_hours_per_day": screen_time,
        "bmi": bmi,
        "waist_to_hip_ratio": waist_ratio,
        "systolic_bp": systolic,
        "diastolic_bp": diastolic,
        "heart_rate": heart_rate,
        "cholesterol_total": cholesterol,
        "hdl_cholesterol": hdl,
        "ldl_cholesterol": ldl,
        "triglycerides": triglycerides,
        "gender": gender,
        "ethnicity": ethnicity,
        "education_level": education_level,
        "income_level": income_level,
        "smoking_status": smoking,
        "employment_status": employment_status,
        "family_history_diabetes": family_history,
        "hypertension_history": hypertension,
        "cardiovascular_history": cardiovascular,
    }

    return submitted, patient_data 