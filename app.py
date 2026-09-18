import streamlit as st
import pandas as pd
import joblib
model = joblib.load('knn_model.pkl')
scaler = joblib.load('scaler.pkl')
expected_columns = joblib.load('columns.pkl')

st.title("Heart Disease Prediction")
st.markdown("This app predicts the likelihood of heart disease based on user input.")
st.markdown("Please enter the following information:")
age= st.slider("Age", 18, 100, 40)
sex= st.selectbox("Sex", ["Male", "Female"])
chest_pain= st.selectbox("Chest Pain Type", ["ATA", "NAP", "ASY", "TA"])
resting_bp= st.number_input("Resting Blood Pressure", min_value=0, max_value=200, value=120)
cholesterol= st.number_input("Cholesterol", min_value=0, max_value=600, value=200)
fasting_bs= st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0,1])
resting_ecg= st.selectbox("Resting ECG", ["Normal", "ST-T Abnormality", "Left Ventricular Hypertrophy"])
max_hr= st.slider("Max Heart Rate Achieved", 60, 220, 150)
exercise_angina= st.selectbox("Exercise Induced Angina", [0,1])
oldpeak= st.number_input("Oldpeak (ST depression induced by exercise)", 0.0, 6.0, 1.0)
st_slope= st.selectbox("Slope of the peak exercise ST segment", ["Upsloping", "Flat", "Downsloping"])

if st.button("Predict"):
     raw_input = {
        'Age': age,
        'RestingBP': resting_bp,
        'Cholesterol': cholesterol,
        'FastingBS': fasting_bs,
        'MaxHR': max_hr,
        'Oldpeak': oldpeak,
        'Sex_' + sex: 1,
        'ChestPainType_' + chest_pain: 1,
        'RestingECG_' + resting_ecg: 1,
        'ExerciseAngina_' + exercise_angina: 1,
        'ST_Slope_' + st_slope: 1
    }
    input_df = pd.DataFrame([raw_input])
    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[expected_columns]
    scaled_input = scaler.transform(input_df)
    prediction = model.predict(scaled_input)[0]
    if prediction == 1:
        st.error("⚠️ High Risk of Heart Disease")
    else:
        st.success("✅ Low Risk of Heart Disease")