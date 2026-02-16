import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model and scaler
model = joblib.load("final_svm_model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Honey Adulteration Detector", layout="centered")

st.title("🍯 Honey Adulteration Detection System")
st.markdown("Upload spectral data (128 wavelength features) to detect adulteration level.")

uploaded_file = st.file_uploader("Upload Spectral CSV File", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    if df.shape[1] != 128:
        st.error("⚠ File must contain exactly 128 spectral columns.")
    else:
        X_scaled = scaler.transform(df.values)
        predictions = model.predict(X_scaled)

        predicted_value = predictions[0]

        st.success("Prediction Completed ✅")

        # 🔥 BIG RESULT DISPLAY
        st.markdown("---")
        st.subheader("🔎 Predicted Adulteration Level")

        if predicted_value == 0:
            st.success(f"🟢 Pure Honey (0%)")
        elif predicted_value <= 10:
            st.info(f"🟡 Low Adulteration ({predicted_value}%)")
        elif predicted_value <= 50:
            st.warning(f"🟠 Moderate Adulteration ({predicted_value}%)")
        else:
            st.error(f"🔴 High Adulteration ({predicted_value}%)")

        st.markdown("---")

        # Optional: show raw preview
        with st.expander("View Uploaded Spectral Data"):
            st.dataframe(df.head())
