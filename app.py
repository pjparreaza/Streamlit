import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

# Función para cargar datos y preprocesarlos
def load_data():
    url = 'https://raw.githubusercontent.com/4GeeksAcademy/decision-tree-project-tutorial/main/diabetes.csv'
    df = pd.read_csv(url)
    df = df.drop_duplicates().reset_index(drop=True)
    return df

df_ch = load_data()

# Mostrar los datos en un DataFrame
st.write("Datos cargados:")
st.dataframe(df_ch)

# Graficos EDA

# Función para crear histogramas
def plot_histograms(data, cols, rows, figsize=(15, 10)):
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    numeric_cols = data.select_dtypes(include=['number']).columns 
    for i, col in enumerate(numeric_cols):
        plt.subplot(rows, cols, i+1)
        plt.hist(data[col])
        plt.title(col)
    plt.tight_layout()
    st.pyplot(fig)

# Función para crear boxplots
def plot_boxplots(data, cols, rows, figsize=(15, 10)):
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    numeric_cols = data.select_dtypes(include=['number']).columns 
    for i, col in enumerate(numeric_cols):
        plt.subplot(rows, cols, i+1)
        plt.boxplot(data[col])
        plt.title(col)
    plt.tight_layout()
    st.pyplot(fig)

# Mostrar histogramas
st.write("Histogramas:")
plot_histograms(df_ch, cols=3, rows=3)

# Mostrar boxplots
st.write("Boxplots:")
plot_boxplots(df_ch, cols=3, rows=3)

# Pairplot
st.write("Pairplot:")
fig_pairplot = sns.pairplot(data=df_ch)
st.pyplot(fig_pairplot)

# Heatmap de correlación
st.write("Heatmap de correlación:")
fig_corr, ax_corr = plt.subplots(figsize=(15, 15))
sns.heatmap(df_ch.corr(), annot=True, fmt=".2f", ax=ax_corr)
st.pyplot(fig_corr)

# Graficos del parallel_coordinates
st.write("Parallel Coordinates Plot:")
fig_pc = plt.figure(figsize=(12, 6))
pd.plotting.parallel_coordinates(df_ch, "Outcome", color=["#E58139", "#39E581", "#8139E5"])
st.pyplot(fig_pc)

# Cargar y mostrar el modelo de árbol de decisión
model = joblib.load("mymodel.joblib")

# Gráfico del árbol de decisión
st.write("Árbol de decisión:")
fig_tree = plt.figure(figsize=(15, 15))
tree.plot_tree(model, feature_names=X_train.columns, class_names=["0", "1", "2"], filled=True)
st.pyplot(fig_tree)

# Título y descripción de la app
st.title("Diabetes Prediction App")

# Campos de entrada del usuario para cada característica
age = st.number_input("Age (years)")
glucose = st.number_input("Blood Glucose Level (mg/dL)")
blood_pressure = st.number_input("Blood Pressure (mmHg)")
skin_thickness = st.number_input("Skin Thickness (mm)")
insulin = st.number_input("Insulin Level (μU/mL)")
bmi = st.number_input("Body Mass Index (kg/m^2)")
diabetes_pedigree = st.number_input("Diabetes Pedigree Function")

# Botón para que el usuario envíe su entrada
submit_button = st.button("Predict")

# Lógica de predicción (se ejecuta cuando el usuario hace clic en el botón)
if submit_button:
    user_data = [[age, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree]]
    prediction = model.predict(user_data)[0]

    # Mostrar la predicción
    if prediction == 1:
        st.write("**Prediction:** You are likely Diabetic.")
    else:
        st.write("**Prediction:** You are likely Non-Diabetic.")
    st.write("**Note:** This is just a prediction based on the model. Please consult a medical professional for any diagnosis.")
