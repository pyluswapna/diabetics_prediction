
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Diabetes Prediction AI",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>

    /* Main background */
    .stApp {
        background-color: #f7f9fc;
    }

    /* Main title */
    .main-title {
        font-size: 42px;
        font-weight: 700;
        color: #17324d;
        margin-bottom: 5px;
    }

    /* Subtitle */
    .subtitle {
        font-size: 17px;
        color: #667085;
        margin-bottom: 25px;
    }

    /* Information cards */
    .info-card {
        background-color: white;
        padding: 20px;
        border-radius: 15px;
        border: 1px solid #e5e7eb;
        box-shadow: 0px 4px 15px rgba(0, 0, 0, 0.05);
        margin-bottom: 15px;
    }

    .card-title {
        font-size: 19px;
        font-weight: 650;
        color: #17324d;
        margin-bottom: 7px;
    }

    .card-text {
        color: #667085;
        font-size: 15px;
    }

    /* Section heading */
    .section-heading {
        font-size: 25px;
        font-weight: 700;
        color: #17324d;
        margin-top: 20px;
        margin-bottom: 12px;
    }

    /* Prediction card */
    .prediction-card {
        background-color: white;
        padding: 30px;
        border-radius: 18px;
        border: 1px solid #e5e7eb;
        box-shadow: 0px 6px 20px rgba(0, 0, 0, 0.07);
        text-align: center;
        margin-top: 20px;
        margin-bottom: 20px;
    }

    /* Button */
    div.stFormSubmitButton > button {
        border-radius: 10px;
        height: 52px;
        font-size: 17px;
        font-weight: 600;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# MODEL PATH
# =========================================================

MODEL_PATH = Path(__file__).parent / "diabetic_prediction_pipeline_1.pkl"


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():

    return joblib.load(MODEL_PATH)


try:

    model = load_model()

except FileNotFoundError:

    st.error(
        "❌ Model file not found. "
        "Please keep `diabetic_prediction_pipeline_1.pkl` "
        "in the same folder as `app_5.py`."
    )

    st.stop()

except Exception as e:

    st.error(f"❌ Unable to load the model: {e}")

    st.stop()


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown("## 🩺 Diabetes AI")

    st.markdown("---")

    st.markdown("### 🤖 Model")

    st.success("XGBoost")

    st.markdown("### 📊 Problem")

    st.write("Binary Classification")

    st.markdown("### 🎯 Prediction")

    st.write("Diabetes / No Diabetes")

    st.markdown("### ⚙️ Techniques")

    st.write("✓ Data Preprocessing")
    st.write("✓ One-Hot Encoding")
    st.write("✓ Feature Scaling")
    st.write("✓ SMOTE")
    st.write("✓ XGBoost")

    st.markdown("---")

    st.markdown("### 📈 Test Performance")

    st.write("Accuracy: **96.85%**")
    st.write("Precision: **88.27%**")
    st.write("Recall: **74.12%**")
    st.write("F1-Score: **80.58%**")
    st.write("ROC-AUC: **0.9791**")
    st.write("PR-AUC: **0.8887**")


# =========================================================
# MAIN HEADER
# =========================================================

st.markdown(
    '<div class="main-title">🩺 Diabetes Prediction AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Machine Learning powered diabetes prediction dashboard'
    '</div>',
    unsafe_allow_html=True
)




# =========================================================
# INPUT SECTION
# =========================================================

st.markdown(
    '<div class="section-heading">👤 Enter Health Information</div>',
    unsafe_allow_html=True
)

st.write(
    "Enter the details below and click **Predict Diabetes**."
)


with st.form("diabetes_prediction_form"):

    # =====================================================
    # PERSONAL INFORMATION
    # =====================================================

    st.markdown("### 👤 Personal Information")

    col1, col2, col3 = st.columns(3)

    with col1:

        gender = st.selectbox(
            "Gender",
            ["Female", "Male", "Other"]
        )

    with col2:

        age = st.number_input(
            "Age",
            min_value=0.0,
            max_value=120.0,
            value=54.0,
            step=1.0
        )

    with col3:

        smoking_history = st.selectbox(
            "Smoking History",
            [
                "never",
                "No Info",
                "current",
                "former",
                "ever",
                "not current"
            ]
        )


    # =====================================================
    # HEALTH INFORMATION
    # =====================================================

    st.markdown("### ❤️ Health Information")

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        hypertension = st.selectbox(
            "Hypertension",
            [0, 1],
            format_func=lambda x:
                "No" if x == 0 else "Yes"
        )

    with col2:

        heart_disease = st.selectbox(
            "Heart Disease",
            [0, 1],
            format_func=lambda x:
                "No" if x == 0 else "Yes"
        )

    with col3:

        bmi = st.number_input(
            "BMI",
            min_value=0.0,
            max_value=100.0,
            value=27.32,
            step=0.01
        )

    with col4:

        hba1c_level = st.number_input(
            "HbA1c Level",
            min_value=0.0,
            max_value=20.0,
            value=6.6,
            step=0.1
        )


    # =====================================================
    # BLOOD GLUCOSE
    # =====================================================

    st.markdown("### 🩸 Blood Glucose")

    blood_glucose_level = st.number_input(
        "Blood Glucose Level",
        min_value=0.0,
        max_value=1000.0,
        value=140.0,
        step=1.0
    )


    st.markdown("")


    # =====================================================
    # PREDICT BUTTON
    # =====================================================

    submitted = st.form_submit_button(
        "🔍 Predict Diabetes",
        use_container_width=True
    )


# =========================================================
# PREDICTION
# =========================================================

if submitted:

    # -----------------------------------------------------
    # CREATE INPUT DATAFRAME
    # -----------------------------------------------------

    new_data = pd.DataFrame(
        [{
            "gender": gender,
            "age": age,
            "hypertension": hypertension,
            "heart_disease": heart_disease,
            "smoking_history": smoking_history,
            "bmi": bmi,
            "HbA1c_level": hba1c_level,
            "blood_glucose_level": blood_glucose_level
        }]
    )


    try:

        # -------------------------------------------------
        # MODEL PREDICTION
        # -------------------------------------------------

        prediction = model.predict(new_data)[0]


        # -------------------------------------------------
        # PROBABILITY
        # -------------------------------------------------

        if hasattr(model, "predict_proba"):

            probability = model.predict_proba(
                new_data
            )[0, 1]

        else:

            probability = None


        # =================================================
        # RESULT
        # =================================================

        st.markdown(
            '<div class="section-heading">📊 Prediction Result</div>',
            unsafe_allow_html=True
        )


        # -------------------------------------------------
        # DIABETES RESULT
        # -------------------------------------------------

        if prediction == 1:

            st.markdown(
                """
                <div style="
                    background-color:#ffe5e5;
                    padding:28px;
                    border-radius:18px;
                    text-align:center;
                    border:2px solid #ff4b4b;
                ">

                    <h1 style="
                        color:#d00000;
                        margin:0;
                        font-size:34px;
                    ">

                        🔴 DIABETES

                    </h1>

                </div>
                """,
                unsafe_allow_html=True
            )


        # -------------------------------------------------
        # NO DIABETES RESULT
        # -------------------------------------------------

        else:

            st.markdown(
                """
                <div style="
                    background-color:#e8f7ee;
                    padding:28px;
                    border-radius:18px;
                    text-align:center;
                    border:2px solid #28a745;
                ">

                    <h1 style="
                        color:#16833a;
                        margin:0;
                        font-size:34px;
                    ">

                        🟢 NO DIABETES

                    </h1>

                </div>
                """,
                unsafe_allow_html=True
            )


        # =================================================
        # PROBABILITY
        # =================================================

        if probability is not None:

            st.markdown("### 🎯 Diabetes Probability")

            col1, col2, col3 = st.columns([1, 2, 1])

            with col2:

                st.metric(
                    "Probability",
                    f"{probability:.2%}"
                )

                st.progress(
                    float(probability)
                )


        # =================================================
        # INPUT SUMMARY
        # =================================================

        st.markdown("### 📋 Entered Information")

        summary_col1, summary_col2 = st.columns(2)


        with summary_col1:

            st.write(
                f"👤 **Gender:** {gender}"
            )

            st.write(
                f"🎂 **Age:** {age:.0f}"
            )

            st.write(
                f"🚬 **Smoking History:** "
                f"{smoking_history}"
            )

            st.write(
                f"⚖️ **BMI:** {bmi:.2f}"
            )


        with summary_col2:

            st.write(
                f"🩸 **HbA1c Level:** "
                f"{hba1c_level:.1f}"
            )

            st.write(
                f"🧪 **Blood Glucose:** "
                f"{blood_glucose_level:.0f}"
            )

            st.write(
                f"❤️ **Hypertension:** "
                f"{'Yes' if hypertension == 1 else 'No'}"
            )

            st.write(
                f"❤️ **Heart Disease:** "
                f"{'Yes' if heart_disease == 1 else 'No'}"
            )


        # =================================================
        # MODEL INPUT DATA
        # =================================================

        with st.expander("🔎 View Model Input Data"):

            st.dataframe(
                new_data,
                use_container_width=True
            )


    # =====================================================
    # ERROR HANDLING
    # =====================================================

    except Exception as e:

        st.error(
            f"❌ Prediction failed: {e}"
        )
