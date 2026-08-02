import streamlit as st

st.title("Deployment Test")

try:
    import joblib
    st.success(f"✅ Joblib: {joblib.__version__}")
except Exception as e:
    st.error(f"❌ Joblib import failed: {e}")

try:
    import plotly
    st.success(f"✅ Plotly: {plotly.__version__}")
except Exception as e:
    st.error(f"❌ Plotly import failed: {e}")

try:
    import pandas
    st.success(f"✅ Pandas: {pandas.__version__}")
except Exception as e:
    st.error(f"❌ Pandas import failed: {e}")

try:
    import sklearn
    st.success(f"✅ Scikit-learn: {sklearn.__version__}")
except Exception as e:
    st.error(f"❌ Scikit-learn import failed: {e}")