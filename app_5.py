import streamlit as st
import pandas as pd
import joblib
import time
import warnings

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Diabetes Prediction",
    page_icon="🩺",
    layout="centered"
)

# ─────────────────────────────────────────────────────────────
# Load Saved Files
# ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    artifacts = joblib.load("diabetic_prediction_pipeline_1.pkl")

    return (
        artifacts["model"],
        artifacts["cat_encod"],
        artifacts["num_encod"]
    )

model, cat_encod, num_encod = load_artifacts()

# Numerical columns
NUM_COLS = ["bmi", "HbA1c_level", "blood_glucose_level"]

# Categorical columns
CAT_COLS = ["gender", "smoking_history"]

# ─────────────────────────────────────────────────────────────
# Styling
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
.main {
    background-color: #f4f6f9;
}

.stButton>button {
    background-color: #0d6efd;
    color: white;
    border-radius: 10px;
    height: 3em;
    width: 100%;
    font-size: 18px;
}

.stButton>button:hover {
    background-color: #0b5ed7;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────
st.title("🩺 Diabetes Prediction System")

st.write(
    "Enter patient health details to predict diabetes."
)

st.divider()

# ─────────────────────────────────────────────────────────────
# Form
# ─────────────────────────────────────────────────────────────
with st.form("diabetes_form"):

    st.subheader("👤 Patient Information")

    col1, col2 = st.columns(2)

    with col1:

        gender = st.selectbox(
            "Gender",
            ["Male", "Female", "Other"]
        )

        smoking_history = st.selectbox(
            "Smoking History",
            ["never", "former", "current", "not current", "ever", "No Info"]
        )

        hypertension = st.selectbox(
            "Hypertension",
            ["No", "Yes"]
        )

    with col2:

        heart_disease = st.selectbox(
            "Heart Disease",
            ["No", "Yes"]
        )

        bmi = st.number_input(
            "BMI",
            min_value=10.0,
            max_value=70.0,
            value=22.5
        )

        hba1c = st.number_input(
            "HbA1c Level",
            min_value=3.0,
            max_value=15.0,
            value=5.5
        )

    blood_glucose = st.number_input(
        "Blood Glucose Level",
        min_value=50,
        max_value=400,
        value=120
    )

    submitted = st.form_submit_button(
        "Predict Diabetes"
    )

# ─────────────────────────────────────────────────────────────
# Prediction
# ─────────────────────────────────────────────────────────────
if submitted:

    with st.spinner("Analyzing patient data..."):
        time.sleep(2)

    # ---------------------------------------------------------
    # Numerical Data Scaling
    # ---------------------------------------------------------
    num_df = pd.DataFrame(
        [[bmi, hba1c, blood_glucose]],
        columns=NUM_COLS
    )

    num_scaled = pd.DataFrame(
        num_encod.transform(num_df),
        columns=NUM_COLS
    )

    # ---------------------------------------------------------
    # Categorical Encoding
    # ---------------------------------------------------------
    known = {
        col: list(cats)
        for col, cats in zip(CAT_COLS, cat_encod.categories_)
    }

    def safe(col, val):
        return val if val in known[col] else known[col][0]

    cat_df_raw = pd.DataFrame(
        [[
            safe("gender", gender),
            safe("smoking_history", smoking_history)
        ]],
        columns=CAT_COLS
    )

    cat_encoded = cat_encod.transform(cat_df_raw)

    if hasattr(cat_encoded, "toarray"):
        cat_encoded = cat_encoded.toarray()

    cat_df = pd.DataFrame(
        cat_encoded,
        columns=cat_encod.get_feature_names_out(CAT_COLS)
    )

    # ---------------------------------------------------------
    # No Transformation Columns
    # ---------------------------------------------------------
    extra_df = pd.DataFrame(
        [[
            1 if hypertension == "Yes" else 0,
            1 if heart_disease == "Yes" else 0
        ]],
        columns=["hypertension", "heart_disease"]
    )

    # ---------------------------------------------------------
    # Final DataFrame
    # ---------------------------------------------------------
    full_df = pd.concat(
        [num_scaled, cat_df, extra_df],
        axis=1
    )

    # Match exact model column order
    expected = list(model.feature_names_in_)

    for col in expected:
        if col not in full_df.columns:
            full_df[col] = 0

    full_df = full_df[expected]

    # ---------------------------------------------------------
    # Prediction
    # ---------------------------------------------------------
    prediction = model.predict(full_df)[0]

    probability = model.predict_proba(full_df)[0]

    diabetic = int(prediction) == 1

    # ---------------------------------------------------------
    # Output
    # ---------------------------------------------------------
    st.divider()

    if diabetic:

        st.error(
            "## ⚠️ Diabetic Prediction: Positive"
        )

        st.write(
            "The patient is likely to have diabetes."
        )

        st.write(
            f"Prediction Confidence: {max(probability)*100:.2f}%"
        )

    else:

        st.success(
            "## ✅ Diabetic Prediction: Negative"
        )

        st.write(
            "The patient is unlikely to have diabetes."
        )

        st.write(
            f"Prediction Confidence: {max(probability)*100:.2f}%"
        )
