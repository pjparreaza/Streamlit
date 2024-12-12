import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
import joblib  # Assuming you have joblib installed for model loading

# Load the preprocessed data (assuming it's saved as "clean_test.csv")
try:
    df_test = pd.read_csv("clean_test.csv")
except FileNotFoundError:
    st.error("Error: clean_test.csv file not found. Please ensure it's in the same directory.")
    exit()

# Model loading
try:
    model = joblib.load("mymodel.joblib")
except FileNotFoundError:
    st.error("Error: mymodel.joblib file not found. Please ensure it's trained and saved.")
    exit()

# Streamlit App Structure
st.title("Diabetes Prediction")  # Replace "Coffe" if not relevant to your data

# Display the test dataset (preprocessed data for prediction)
st.write("**Test Dataset** (for reference)")
st.dataframe(df_test.style.set_properties(**{'background-color': 'lightblue'}))

# **Check for missing features in user input**
missing_features = set(model.feature_names_in_) - set(df_test.columns)

if missing_features:
    st.error(f"Error: The following features are missing from the input data but were used during training: {', '.join(missing_features)}")
    st.stop()  # Halt execution if essential features are missing

# User Input Section (replace with relevant features from your model)
st.header("Enter Patient Information:")

# **Ensure user input covers all required features**
user_data = {}
for feature in model.feature_names_in_:
    if feature in df_test.columns:
        user_data[feature] = st.number_input(feature, min_value=df_test[feature].min(), max_value=df_test[feature].max())
    else:
        st.warning(f"Feature '{feature}' not found in input data. Prediction may be inaccurate.")

# Prepare user input for prediction
user_data = pd.DataFrame(user_data).transpose()  # Create DataFrame from user input

# Make prediction using the loaded model
prediction = model.predict(user_data)

# Display prediction result
if prediction[0] == 0:
    st.success("**Prediction:** Low probability of diabetes.")
elif prediction[0] == 1:
    st.warning("**Prediction:** Medium probability of diabetes. Please consult a doctor.")
else:
    st.error("**Prediction:** High probability of diabetes. Please consult a doctor immediately.")

# Note: Consider adding a disclaimer that this is not a substitute for medical advice.