# app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# --- Page Configuration ---
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="centered"
)


# --- Caching the Models ---
# Use st.cache_resource to load models only once and speed up the app
@st.cache_resource
def load_models():
    """Load all the trained models and encoders."""
    knn_model = joblib.load('knn_model.joblib')
    svm_model = joblib.load('svm_model.joblib')
    ann_model = load_model('ann_model.h5')
    preprocessor = joblib.load('preprocessor.joblib')
    label_encoder = joblib.load('label_encoder.joblib')
    return knn_model, svm_model, ann_model, preprocessor, label_encoder

# Load the models
knn_model, svm_model, ann_model, preprocessor, label_encoder = load_models()


# --- UI Layout ---
st.title("🎓 Student Academic Performance Predictor")
st.write(
    "This app predicts a student's final academic performance based on their personal, "
    "social, and school-related attributes. Please provide the student's information below."
)

st.header("Student Information")

# Create columns for a cleaner layout
col1, col2 = st.columns(2)

# --- Input Widgets ---
# We will select a subset of the most important features for the GUI to keep it simple.
with col1:
    sex = st.selectbox("Gender (sex)", options=['F', 'M'])
    age = st.slider("Age", 15, 22, 16)
    studytime = st.selectbox("Weekly Study Time (studytime)", options=[1, 2, 3, 4], format_func=lambda x: f"{x} ({['<2 hrs', '2-5 hrs', '5-10 hrs', '>10 hrs'][x-1]})")
    failures = st.selectbox("Number of Past Class Failures (failures)", options=[0, 1, 2, 3])

with col2:
    schoolsup = st.selectbox("Extra Educational Support (schoolsup)", options=['yes', 'no'])
    paid = st.selectbox("Extra Paid Classes (paid)", options=['yes', 'no'])
    absences = st.slider("Number of School Absences (absences)", 0, 93, 5)
    goout = st.selectbox("Going Out with Friends (goout)", options=[1, 2, 3, 4, 5], format_func=lambda x: f"{x} ({['Very Low', 'Low', 'Medium', 'High', 'Very High'][x-1]})")

# --- Prediction Logic ---
if st.button("Predict Performance", type="primary"):

    # 1. Create a DataFrame from user inputs
    # The order and names of columns MUST match the original training data.
    # We will fill the rest of the columns with default/mode values.
    # This is a simplified approach. For a full app, you'd add widgets for all features.
    input_data = {
        'school': 'GP', 'sex': sex, 'age': age, 'address': 'U', 'famsize': 'GT3',
        'Pstatus': 'A', 'Medu': 4, 'Fedu': 4, 'Mjob': 'at_home', 'Fjob': 'teacher',
        'reason': 'course', 'guardian': 'mother', 'traveltime': 2, 'studytime': studytime,
        'failures': failures, 'schoolsup': schoolsup, 'famsup': 'no', 'paid': paid,
        'activities': 'no', 'nursery': 'yes', 'higher': 'yes', 'internet': 'yes',
        'romantic': 'no', 'famrel': 4, 'freetime': 3, 'goout': goout, 'Dalc': 1,
        'Walc': 1, 'health': 3, 'absences': absences
    }
    input_df = pd.DataFrame([input_data])

    st.write("---")
    st.header("Prediction Results")

    # 2. Pre-process the user's input using the saved preprocessor
    # This converts strings like 'at_home' into the numerical format the models expect.
    input_processed = preprocessor.transform(input_df)

    # 3. Make predictions with all three models using the PROCESSED data
    knn_prediction = knn_model.predict(input_processed)[0]
    svm_prediction = svm_model.predict(input_processed)[0]
    
    ann_pred_prob = ann_model.predict(input_processed)
    ann_pred_index = np.argmax(ann_pred_prob, axis=1)[0]
    ann_prediction = label_encoder.inverse_transform([ann_pred_index])[0]

    # Display results
    res_col1, res_col2, res_col3 = st.columns(3)
    with res_col1:
        st.subheader("KNN Model")
        st.success(f"Predicted: **{knn_prediction}**")
    
    with res_col2:
        st.subheader("SVM Model")
        st.success(f"Predicted: **{svm_prediction}**")

    with res_col3:
        st.subheader("ANN Model")
        st.success(f"Predicted: **{ann_prediction}**")

st.markdown("---")

st.write("Developed by Low Jia Yuan, Abigail Chong Yung Ping, and Nathaniel Woo Shih Yan.")
