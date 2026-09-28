
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
        background: #f7f9fc;
    }

    /* Main title */
    .main-title {
        font-size: 42px;
        font-weight: 700;
        color: #17324d;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 17px;
        color: #667085;
        margin-bottom: 25px;
    }

    /* Cards */
    .info-card {
        background: white;
        padding: 22px;
        border-radius: 15px;
        border: 1px solid #e5e7eb;
        box-shadow: 0px 4px 15px rgba(0,0,0,0.05);
        margin-bottom: 15px;
    }

    .card-title {
        font-size: 20px;
        font-weight: 650;
        color: #17324d;
        margin-bottom: 8px;
    }

    .card-text {
        color: #667085;
        font-size: 15px;
    }

    /* Prediction card */
    .prediction-card {
        background: white;
        padding: 30px;
        border-radius: 18px;
        border: 1px solid #e5e7eb;
        box-shadow: 0px 6px 20px rgba(0,0,0,0.07);
        text-align: center;
        margin-top: 20px;
        margin-bottom: 20px;
    }

    .prediction-title {
        font-size: 25px;
        font-weight: 700;
        color: #17324d;
    }

    .prediction-value {
        font-size: 32px;
        font-weight: 750;
        margin-top: 10px;
    }

    /* Section heading */
    .section-heading {
        font-size: 25px;
        font-weight: 700;
        color: #17324d;
        margin-top: 20px;
        margin-bottom: 12px;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #667085;
        font-size: 14px;
        padding: 25px;
    }

    /* Button */
    div.stButton > button,
    div[data-testid="stFormSubmitButton"] button {
        border-radius: 10px;
        height: 50px;
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
        "Please keep diabetic_prediction_pipeline_1.pkl "
        "in the same folder as app_5.py."
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

    st.markdown(
        """
        ### 📌 Project

        **Diabetes Prediction using Machine Learning**

        This application uses a trained **XGBoost** model
        to estimate the probability of diabetes from the
        entered information.
        """
    )

    st.markdown("---")

    st.markdown("### 🤖 Model")

    st.success("XGBoost")

    st.markdown("### 🔧 Pipeline")

    st.write("✓ Data preprocessing")
    st.write("✓ Feature encoding")
    st.write("✓ Feature scaling")
    st.write("✓ SMOTE")
    st.write("✓ XGBoost prediction")

    st.markdown("---")

    
# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="main-title">🩺 Diabetes Prediction AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Machine Learning powered diabetes risk prediction dashboard'
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# TOP INFORMATION CARDS
# =========================================================

col1, col2, col3, col4 = st.columns(4)


with col1:

    st.markdown(
        """
        <div class="info-card">
            <div class="card-title">🤖 Model</div>
            <div class="card-text">XGBoost Classifier</div>
        </div>
        """,
        unsafe_allow_html=True
    )


with col2:

    st.markdown(
        """
        <div class="info-card">
            <div class="card-title">📊 Problem</div>
            <div class="card-text">Binary Classification</div>
        </div>
        """,
        unsafe_allow_html=True
    )


with col3:

    st.markdown(
        """
        <div class="info-card">
            <div class="card-title">🎯 Target</div>
            <div class="card-text">Diabetes Prediction</div>
        </div>
        """,
        unsafe_allow_html=True
    )


with col4:

    st.markdown(
        """
        <div class="info-card">
            <div class="card-title">⚙️ Technique</div>
            <div class="card-text">SMOTE + ML Pipeline</div>
        </div>
        """,
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
    "Please enter the information below and click "
    "**Predict Diabetes**."
)


with st.form("diabetes_form"):

    # -----------------------------------------------------
    # PERSONAL INFORMATION
    # -----------------------------------------------------

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


    # -----------------------------------------------------
    # HEALTH INFORMATION
    # -----------------------------------------------------

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


    # -----------------------------------------------------
    # BLOOD GLUCOSE
    # -----------------------------------------------------

    st.markdown("### 🩸 Blood Glucose")

    blood_glucose_level = st.number_input(
        "Blood Glucose Level",
        min_value=0.0,
        max_value=1000.0,
        value=140.0,
        step=1.0
    )


    st.markdown("")


    # -----------------------------------------------------
    # PREDICT BUTTON
    # -----------------------------------------------------

    submitted = st.form_submit_button(
        "🔍 Predict Diabetes",
        use_container_width=True
    )


# =========================================================
# PREDICTION
# =========================================================

if submitted:

    # Create dataframe
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

            try:

        # -------------------------------------------------
        # MODEL PREDICTION
        # -------------------------------------------------

        prediction = model.predict(new_data)[0]

        # -------------------------------------------------
        # PROBABILITY
        # -------------------------------------------------

        if hasattr(model, "predict_proba"):

            probability = model.predict_proba(new_data)[0, 1]

        else:

            probability = None

        # -------------------------------------------------
        # PREDICTION RESULT
        # -------------------------------------------------

        st.markdown("### 📊 Prediction Result")

        if prediction == 1:

            st.markdown(
                """
                <div style="
                    background-color:#ffe5e5;
                    padding:25px;
                    border-radius:15px;
                    text-align:center;
                    border:2px solid #ff4b4b;
                ">
                    <h1 style="color:#d00000; margin:0;">
                        🔴 DIABETES
                    </h1>
                </div>
                """,
                unsafe_allow_html=True
            )

        else:

            st.markdown(
                """
                <div style="
                    background-color:#e8f7ee;
                    padding:25px;
                    border-radius:15px;
                    text-align:center;
                    border:2px solid #28a745;
                ">
                    <h1 style="color:#16833a; margin:0;">
                        🟢 NO DIABETES
                    </h1>
                </div>
                """,
                unsafe_allow_html=True
            )

        # -------------------------------------------------
        # PROBABILITY
        # -------------------------------------------------

        if probability is not None:

            st.markdown("### 🎯 Diabetes Probability")

            col1, col2, col3 = st.columns([1, 2, 1])

            with col2:

                st.metric(
                    "Probability",
                    f"{probability:.2%}"
                )

                st.progress(float(probability))

        # -------------------------------------------------
        # INPUT SUMMARY
        # -------------------------------------------------

        st.markdown("### 📋 Entered Information")

        summary_col1, summary_col2 = st.columns(2)

        with summary_col1:

            st.write(f"👤 **Gender:** {gender}")

            st.write(f"🎂 **Age:** {age:.0f}")

            st.write(
                f"🚬 **Smoking History:** {smoking_history}"
            )

            st.write(f"⚖️ **BMI:** {bmi:.2f}")

        with summary_col2:

            st.write(
                f"🩸 **HbA1c Level:** {hba1c_level:.1f}"
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

        # -------------------------------------------------
        # MODEL INPUT DATA
        # -------------------------------------------------

        with st.expander("🔎 View Model Input Data"):

            st.dataframe(
                new_data,
                use_container_width=True
            )

    except Exception as e:

        st.error(
            f"❌ Prediction failed: {e}"
        )



        
