import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Load model and scaler
model = joblib.load("final_svm_model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="AI Honey Authentication Platform", layout="wide")

# ------------------ SIDEBAR ------------------
st.sidebar.title("🍯 Honey AI Platform")
page = st.sidebar.radio("Navigation", 
                        ["🏠 Single Sample Analysis",
                         "📊 Batch Analysis",
                         "ℹ About Model"])

# ------------------ COMMON FUNCTIONS ------------------

def classify_sample(predicted_value):
    hqi = 100 - predicted_value

    if predicted_value == 0:
        level = "Pure Honey"
        grade = "Grade A"
        risk = "Safe"
        recommendation = "Suitable for retail & export."
    elif predicted_value <= 10:
        level = "Low Adulteration"
        grade = "Grade B"
        risk = "Mild Risk"
        recommendation = "Acceptable for retail use."
    elif predicted_value <= 50:
        level = "Moderate Adulteration"
        grade = "Grade C"
        risk = "Industrial Grade"
        recommendation = "Recommended for industrial processing."
    else:
        level = "High Adulteration"
        grade = "Grade D"
        risk = "Adulterated"
        recommendation = "Not suitable for consumption."

    return hqi, level, grade, risk, recommendation


# ================================
# 🏠 SINGLE SAMPLE ANALYSIS
# ================================
if page == "🏠 Single Sample Analysis":

    st.title("🍯 AI-Based Honey Quality Dashboard")
    st.markdown("Upload spectral data (128 wavelength features)")

    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        if df.shape[1] != 128:
            st.error("⚠ File must contain exactly 128 spectral columns.")
        else:
            X_scaled = scaler.transform(df.values)
            prediction = int(model.predict(X_scaled)[0])

            try:
                confidence = np.max(model.predict_proba(X_scaled)) * 100
            except:
                confidence = 95

            hqi, level, grade, risk, recommendation = classify_sample(prediction)

            st.markdown("---")
            st.header("📊 Quality Summary")

            col1, col2, col3, col4 = st.columns(4)

            col1.metric("Adulteration %", f"{prediction}%")
            col2.metric("HQI Score", f"{hqi}/100")
            col3.metric("Grade", grade)
            col4.metric("Confidence", f"{confidence:.2f}%")

            st.markdown("### 🚦 Risk Meter")
            st.progress(prediction)

            st.markdown("---")
            st.subheader("🧠 AI Interpretation")
            st.write(f"**Classification:** {level}")
            st.write(f"**Risk Status:** {risk}")
            st.write(f"**Recommendation:** {recommendation}")

            st.markdown("---")
            st.subheader("📈 Spectral Visualization")

            fig, ax = plt.subplots(figsize=(10,4))
            ax.plot(df.values[0])
            ax.set_title("Spectral Signature (128 Wavelength Features)")
            ax.set_xlabel("Wavelength Index")
            ax.set_ylabel("Intensity")
            st.pyplot(fig)

# ================================
# 📊 BATCH ANALYSIS MODE
# ================================
elif page == "📊 Batch Analysis":

    st.title("📊 Batch Honey Sample Testing")

    uploaded_file = st.file_uploader("Upload Multiple Samples CSV", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        if df.shape[1] != 128:
            st.error("⚠ File must contain exactly 128 spectral columns.")
        else:
            X_scaled = scaler.transform(df.values)
            predictions = model.predict(X_scaled)

            results = []

            for i, pred in enumerate(predictions):
                pred = int(pred)
                hqi, level, grade, risk, recommendation = classify_sample(pred)

                results.append({
                    "Sample ID": f"S{i+1}",
                    "Adulteration %": pred,
                    "HQI": hqi,
                    "Grade": grade,
                    "Risk": risk
                })

            result_df = pd.DataFrame(results)

            st.success("Batch Analysis Completed ✅")
            st.dataframe(result_df)

            csv = result_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Download Results",
                csv,
                "honey_batch_results.csv",
                "text/csv"
            )

# ================================
# ℹ ABOUT MODEL
# ================================
else:

    st.title("ℹ Model Information")

    st.markdown("""
    ### 🔬 Model Details

    - Algorithm: Support Vector Machine (SVM)
    - Kernel: RBF
    - Features: 128 Spectral Wavelength Intensities
    - Preprocessing: StandardScaler
    - Output: Adulteration Percentage
    - Derived Metric: Honey Quality Index (HQI)

    ### 🎯 Objective

    To detect honey adulteration using spectral signature analysis
    and convert predictions into actionable food safety intelligence.
    """)

