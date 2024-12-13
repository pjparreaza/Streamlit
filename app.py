import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import train_test_split

# Title and description for your app
st.title("Aplicación para predecir la diabetes")

# Función para cargar datos y preprocesarlos
def load_data():
    url = 'https://raw.githubusercontent.com/4GeeksAcademy/decision-tree-project-tutorial/main/diabetes.csv'
    df = pd.read_csv(url)
    df = df.drop_duplicates().reset_index(drop=True)
    return df

df_ch = load_data()

st.write("Este conjunto de datos proviene originalmente del Instituto Nacional de Diabetes y Enfermedades Digestivas y Renales. El objetivo es predecir en base a medidas diagnósticas si un paciente tiene o no diabetes, para lo cual procedemos a realizar un arbol de decisión")

# Mostrar los datos en un DataFrame
st.write("Procederemos a mostrar los datos cargados:")
st.dataframe(df_ch)

st.write("En este conjunto de datos contienen las siguientes variables:")

st.write(" * Pregnancies. Número de embarazos del paciente (numérico)")
st.write(" * Glucose. Concentración de glucosa en plasma a las 2 horas de un test de tolerancia oral a la glucosa (numérico)")
st.write(" * BloodPressure. Presión arterial diastólica (medida en mm Hg) (numérico) ")
st.write(" * SkinThickness. Grosor del pliegue cutáneo del tríceps (medida en mm) (numérico)")
st.write(" * Insulin. Insulina sérica de 2 horas (medida en μU/ml) (numérico)")
st.write(" * BMI. Índice de masa corporal (numérico)")
st.write(" *DiabetesPedigreeFunction. Función de pedigrí de diabetes (numérico)")
st.write(" * Age. Edad del paciente (numérico)")
st.write(" * Outcome. Variable de clase (0 o 1), siendo 0 negativo en diabetes y 1, positivo (numérico) ")

# Graficos EDA
st.header("EDA completo")

st.markdown("### Estadisticos")
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

st.write("de los graficos anteriores podemos indicar que:")

st.write(" + Variable `Pregnancies`: la distribucion del numero de embarazos de los pacientes posee asimetria a la derecha y presencia de valores extremos.")

st.write(" + Variable `Glucose`: la distribucion de la concentración de glucosa en plasma a las 2 horas de un test de tolerancia oral a la glucosa posee un comportamiento simetrico y no posee valores extremos.")

st.write(" + Variable `BloodPressure`: la diostribucion de la presión arterial diastólica (medida en mm Hg) posee un comportamiento simetrico y posee valores extremos.")

st.write(" + Variable `SkinThickness`: la distribucion del grosor del pliegue cutáneo del tríceps posee un comportamiento asimetrico a la derecha y posee valores extremos.")

st.write(" + Variable `Insulin`: la distribucion de la insulina sérica de 2 horas posee un comportamiento asimetrico a la derecha y posee valores extremos.")

st.write(" + Variable `BMI`: la distribucion del Indice de masa corporal posee un comportamiento simetrico y posee valores extremos.")

st.write(" + Variable `DiabetesPedigreeFunction`: la distribucion de la función de pedigrí de diabetes posee asimetria a la derecha y presencia de valores extremos.")

st.write(" + Variable `Age`: la distribucion de las edades de los pacientes posee asimetria a la derecha y presencia de valores extremos.")

st.write(" + Variable `Outcome`: la variable de clase (0 o 1), siendo 0 negativo en diabetes y 1, positivo por ser de conteo revela que la poblacion de estudio posee una poblacion sana de diabetes superior a la poblacion que sufre de diabeles.")


# st.subheader("Estadisticos Multivariantes")

# Pairplot
#st.write("Pairplot:")
# fig_pairplot = sns.pairplot(data=df_ch)
# st.pyplot(fig_pairplot)

# Heatmap de correlación
st.write("Matriz de correlación:")
fig_corr, ax_corr = plt.subplots(figsize=(15, 15))
sns.heatmap(df_ch.corr(), annot=True, fmt=".2f", ax=ax_corr)
st.pyplot(fig_corr)

st.write(" Al observar la correlacion, se puede decir que, las variables con mayor relacion moderada (<60%) en el mismo sentido son, en primera instancia es las variable edad con numero de embarazos, la cual posee un 54% (0,54) seguida y la glucosa con la variable de estado de la diabetes, la cual tiene un 47% (0,47).")


st.subheader("Construcción un modelo de árbol de decisión")

# st.write("Procedamos a vizualizar la relacion las variables que son consideradas independientes respecto al objetivo, para ello realizaremos una grafica llamama `parallel_coordinates`")

# Graficos del parallel_coordinates
# st.write("Parallel Coordinates Plot:")
# fig_pc = plt.figure(figsize=(12, 6))
# pd.plotting.parallel_coordinates(df_ch, "Outcome", color=["#E58139", "#39E581", "#8139E5"])
# st.pyplot(fig_pc)


X = df_ch.drop("Outcome", axis = 1)
y = df_ch["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

selection_model = SelectKBest(k = 7)
selection_model.fit(X_train, y_train)

selected_columns = X_train.columns[selection_model.get_support()]
X_train_sel = pd.DataFrame(selection_model.transform(X_train), columns = selected_columns)
X_test_sel = pd.DataFrame(selection_model.transform(X_test), columns = selected_columns)

train_data = X_train_sel
test_data = X_test_sel

X_train_sel["Outcome"] = y_train.values
X_test_sel["Outcome"] = y_test.values

X_train = train_data.drop(["Outcome"], axis = 1)
y_train = train_data["Outcome"]
X_test = test_data.drop(["Outcome"], axis = 1)
y_test = test_data["Outcome"]


# Load and display decision tree model (assuming 'mymodel.joblib' exists)
try:
    model = joblib.load("mymodel.joblib")
except FileNotFoundError:
    st.error("Model 'mymodel.joblib' not found. Please train the model first.")
    model = None  # Set model to None to prevent errors in prediction section

# Decision tree plot (if model loaded successfully)
if model is not None:

    st.write("Procesamos a observar nuestro arbol de decisión:")
    fig_tree = plt.figure(figsize=(15, 15))
    try:  # Handle potential errors during tree plotting
        tree.plot_tree(model, feature_names=X_train.columns, class_names=["0", "1", "2"], filled=True)
    except Exception as e:
        st.error(f"Error plotting decision tree: {e}")
    st.pyplot(fig_tree)

st.write("En base a los resultados anteriores podemos indicar que:")

st.write(" + El arbol posee 14 niveles.")
st.write(" + Tenemos 105 nodos terminales.")
st.write(" + En cada nodo se aprecia el valor del indice de Gini, criterio utilizado para medir el grado de pudeza de cada nodo. Mientras mas pequeño el indice de Gini implica un nodo mas puro.")
st.write("+ Todos los nodos terminales poseen un indice de Gini cuyo valor es tan despreciable que es considerado nulo.")

st.write("Interpretacion del nodo inicial (raiz o padre):")

st.write("La variable que tiene mayor influencia en la presencia de la diabetes es la concentracion de glucosa, la cantidad de datos observados en el nodo raiz es de 614, de aqui se desprende el primer nivel para el cual se tiene dos alternativas:")

st.write(" + 1. La presencia de la diabetes si esta determinada por la glucosa y esto depende de la edad.")
st.write(" + 2. La presencia de la diabetes no esta determinada por la glucosa y esta depende del Indice de Masa Corporal.")



st.subheader("Predicción de Modelo")

st.write("Ingrese su información para obtener una predicción sobre la diabetes.")

# User input fields for prediction
age = st.number_input("Edad (años)")
glucose = st.number_input("Nivel de glucosa en sangre (mg/dL)")
blood_pressure = st.number_input("Presión arterial (mmHg)")
skin_thickness = st.number_input("Grosor de la piel (mm)")
insulin = st.number_input("Nivel de insulina (μU/mL)")
bmi = st.number_input("Índice de masa corporal (kg/m^2)")
diabetes_pedigree = st.number_input("Función del pedigrí de la diabetes")

# Button for prediction and logic
submit_button = st.button("Predecir")
if submit_button:
    user_data = [[age, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree]]

    # Train the model if it's not loaded (replace with your actual training data)
    if model is None:
        X_train, X_test, y_train, y_test = train_test_split(df_ch.drop("Outcome", axis=1), df_ch["Outcome"], test_size=0.2, random_state=42)
        X_train, y_train = select_features(X_train.copy())  # Select features for training
        model = train_model(X_train, y_train)

    prediction = model.predict(user_data)[0]

    # Show prediction
    if prediction == 1:
        st.write("** Predicción: ** Es probable que sea diabético(a).")
    else:
        st.write("**Predicción:** Es probable que no sea diabético.")
    st.write("**Nota:** Esta es solo una predicción basada en el modelo. Consulte a un profesional médico para cualquier diagnóstico.")