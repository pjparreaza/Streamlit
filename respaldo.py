import streamlit as st
import joblib  # Assuming joblib is installed

# Load the trained decision tree model
model = joblib.load("mymodel.joblib")

# Title and description for your app
st.title("Diabetes Prediction App")
st.write("Enter your information below to get a prediction on diabetes.")

# User input fields for each feature
age = st.number_input("Age (years)")
glucose = st.number_input("Blood Glucose Level (mg/dL)")
blood_pressure = st.number_input("Blood Pressure (mmHg)")
skin_thickness = st.number_input("Skin Thickness (mm)")
insulin = st.number_input("Insulin Level (μU/mL)")
bmi = st.number_input("Body Mass Index (kg/m^2)")
diabetes_pedigree = st.number_input("Diabetes Pedigree Function")

# Button for user to submit their input
submit_button = st.button("Predict")

# Prediction logic (executed when the user clicks the button)
if submit_button:
    user_data = [[age, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree]]
    prediction = model.predict(user_data)[0]

    # Display the prediction
    if prediction == 1:
        st.write("**Prediction:** You are likely Diabetic.")
    else:
        st.write("**Prediction:** You are likely Non-Diabetic.")
        st.write("**Note:** This is just a prediction based on the model. Please consult a medical professional for any diagnosis.")