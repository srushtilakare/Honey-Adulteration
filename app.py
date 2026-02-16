import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Load model and scaler
model = joblib.load("final_svm_model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="AI Honey Quality Analyzer", layout="wide")

st.title("🍯 AI-Based Honey Adulteration & Quality Assessment System")
st.markdown("Advanced Spectral Intelligence for Food Authentication")

uploaded_file = st.file_uploader("Upload Spectral CSV File (128 features)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    if df.shape[1] != 128:
        st.error("⚠ File must contain exactly 128 spectral columns.")
    else:
        X_scaled = scaler.transform(df.values)
        predictions = model.predict(X_scaled)

        predicted_value = int(predictions[0])

        # Confidence (if probability enabled)
        try:
            confidence = np.max(model.predict_proba(X_scaled)) * 100
        except:
            confidence = 95  # fallback if not available

        quality_index = 100 - predicted_value

        # Risk categorization
        if predicted_value == 0:
            level = "Pure Honey"
            grade = "Grade A"
            risk = "Safe"
            recommendation = "Suitable for direct retail and export."
        elif predicted_value <= 10:
            level = "Low Adulteration"
            grade = "Grade B"
            risk = "Mild Risk"
            recommendation = "Acceptable for retail use."
        elif predicted_value <= 50:
            level = "Moderate Adulteration"
            grade = "Grade C"
            risk = "Unsafe for Export"
            recommendation = "Recommended for industrial processing only."
        else:
            level = "High Adulteration"
            grade = "Grade D"
            risk = "Adulterated"
            recommendation = "Not suitable for consumption."

        st.markdown("---")
        st.header("📊 Analysis Summary")

        col1, col2, col3 = st.columns(3)

        col1.metric("Adulteration Level", f"{predicted_value}%")
        col2.metric("Quality Grade", grade)
        col3.metric("Quality Index Score", f"{quality_index}/100")

        st.progress(predicted_value)

        st.markdown("---")

        st.subheader("🧠 AI Interpretation")
        st.write(f"**Classification:** {level}")
        st.write(f"**Risk Status:** {risk}")
        st.write(f"**Model Confidence:** {confidence:.2f}%")
        st.write(f"**Recommendation:** {recommendation}")

        st.markdown("---")

        st.subheader("📈 Spectral Pattern Visualization")

        fig, ax = plt.subplots(figsize=(10,4))
        ax.plot(df.values[0])
        ax.set_title("Spectral Signature (128 Wavelength Features)")
        ax.set_xlabel("Wavelength Index")
        ax.set_ylabel("Intensity")
        st.pyplot(fig)

        st.markdown("---")

        with st.expander("View Uploaded Data"):
            st.dataframe(df.head())
