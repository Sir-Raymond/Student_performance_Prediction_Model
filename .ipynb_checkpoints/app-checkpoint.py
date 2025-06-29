import streamlit as st
import pandas as pd
import joblib

# Load model and scaler
model = joblib.load('best_model_logreg.pkl')
scaler = joblib.load('scaler.pkl')

st.set_page_config(page_title="Student Pass Predictor", layout="centered")

# --- UI ---
st.title("🎓 Student Pass/Fail Prediction App")
st.markdown("This app predicts whether a student is likely to pass or fail based on exam scores and background.")

# Input form
with st.form("student_form"):
    st.subheader("📋 Student Info")

    gender = st.selectbox("Gender", ["female", "male"])
    race = st.selectbox("Race/Ethnicity", ["group A", "group B", "group C", "group D", "group E"])
    parental_edu = st.selectbox("Parental Level of Education", [
        "some high school", "high school", "some college", "associate's degree", "bachelor's degree", "master's degree"])
    lunch = st.selectbox("Lunch Type", ["standard", "free/reduced"])
    test_prep = st.selectbox("Test Preparation Course", ["none", "completed"])

    st.subheader("✏️ Exam Scores")
    math_score = st.slider("Math Score", 0, 100, 70)
    reading_score = st.slider("Reading Score", 0, 100, 70)
    writing_score = st.slider("Writing Score", 0, 100, 70)

    submitted = st.form_submit_button("Predict")

# Dictionaries for encoding
map_gender = {"female": 0, "male": 1}
map_race = {"group A": 0, "group B": 1, "group C": 2, "group D": 3, "group E": 4}
map_parental = {
    "some high school": 0,
    "high school": 1,
    "some college": 2,
    "associate's degree": 3,
    "bachelor's degree": 4,
    "master's degree": 5
}
map_lunch = {"standard": 0, "free/reduced": 1}
map_test_prep = {"none": 0, "completed": 1}

# Predict
if submitted:
    input_data = pd.DataFrame([{
        "gender": map_gender[gender],
        "race/ethnicity": map_race[race],
        "parental level of education": map_parental[parental_edu],
        "lunch": map_lunch[lunch],
        "test preparation course": map_test_prep[test_prep],
        "math score": math_score,
        "reading score": reading_score,
        "writing score": writing_score
    }])

    # Feature engineering (match training columns exactly)
    input_data['score_range'] = max(math_score, reading_score, writing_score) - min(math_score, reading_score, writing_score)
    input_data['parental_edu_x_writing'] = input_data['parental level of education'] * writing_score
    input_data['race_x_math'] = input_data['race/ethnicity'] * math_score

    # Reorder/Select features to match training
    input_data = input_data[[
        'gender', 'race/ethnicity', 'parental level of education', 'lunch',
        'test preparation course', 'math score', 'reading score', 'writing score',
        'score_range', 'parental_edu_x_writing', 'race_x_math']]

    scaled = scaler.transform(input_data)
    prediction = model.predict(scaled)[0]
    proba = model.predict_proba(scaled)[0][1]

    st.subheader("🔍 Prediction Result")
    if prediction == 1:
        st.success(f"✅ The student is likely to PASS. Confidence: {proba:.2f}")
    else:
        st.error(f"❌ The student is likely to FAIL. Confidence: {1 - proba:.2f}")

    # Reasoning hint
    if proba >= 0.8:
        st.info("Confidence is high — scores are strong across all exams.")
    elif proba >= 0.6:
        st.warning("Scores are decent, but a bit borderline. Improving one subject could help.")
    else:
        st.error("Low confidence — student may need additional support or preparation.")
