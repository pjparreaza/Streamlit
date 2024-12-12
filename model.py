import pandas as pd
import joblib


total_data = pd.read_csv("https://raw.githubusercontent.com/4GeeksAcademy/decision-tree-project-tutorial/main/diabetes.csv")

total_data = total_data.drop_duplicates().reset_index(drop = True)

total_data.head()

# Graficos del EDA

# estadisticos univariados

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns   

# Obtener las columnas numericas **antes** de llamar a la funcion
numeric_cols = total_data.select_dtypes(include=['number']).columns 

def plot_histograms(data, cols, rows, figsize=(15, 10)):
    """
    Crea una cuadricula de histogramas para las columnas numericas de un DataFrame.

    Args:
        data: DataFrame con los datos.
        cols: Numero de columnas en la cuadricula.
        rows: Numero de filas en la cuadricula.
        figsize: Tamaño de la figura.
    """

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    for i, col in enumerate(numeric_cols):
        plt.subplot(rows, cols, i+1)
        plt.hist(data[col])
        plt.title(col)
    plt.tight_layout()
    plt.show()

# Crear una cuadrícula de 3x3
plot_histograms(total_data, cols=3, rows=3)


# numeric_cols = total_data.select_dtypes(include=['number']).columns 

def plot_boxplots(data, cols, rows, figsize=(15, 10)):
    """
    Crea una cuadricula de diagramas de caja para las columnas numericas de un DataFrame.

    Args:
        data: DataFrame con los datos.
        cols: Numero de columnas en la cuadricula.
        rows: Numero de filas en la cuadricula.
        figsize: Tamaño de la figura.
    """

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    for i, col in enumerate(numeric_cols):
        plt.subplot(rows, cols, i+1)
        plt.boxplot(data[col])
        plt.title(col)
    plt.tight_layout()
    plt.show()

# Crear una cuadrícula de 3x3
plot_boxplots(total_data, cols=3, rows=3)


# Estadisticos multivariantes

sns.pairplot(data = total_data)

# correlacion

fig, axes = plt.subplots(figsize=(15, 15))

sns.heatmap(total_data.corr(), annot = True, fmt = ".2f")

plt.tight_layout()

# Draw Plot
plt.show()




# Selección de características

from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import train_test_split

X = total_data.drop("Outcome", axis = 1)
y = total_data["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

selection_model = SelectKBest(k = 7)
selection_model.fit(X_train, y_train)

selected_columns = X_train.columns[selection_model.get_support()]
X_train_sel = pd.DataFrame(selection_model.transform(X_train), columns = selected_columns)
X_test_sel = pd.DataFrame(selection_model.transform(X_test), columns = selected_columns)

X_train_sel["Outcome"] = y_train.values
X_test_sel["Outcome"] = y_test.values
X_train_sel.to_csv("clean_train.csv", index = False)
X_test_sel.to_csv("clean_test.csv", index = False)

train_data = pd.read_csv("clean_train.csv")
test_data = pd.read_csv("clean_test.csv")

X_train = train_data.drop(["Outcome"], axis = 1)
y_train = train_data["Outcome"]
X_test = test_data.drop(["Outcome"], axis = 1)
y_test = test_data["Outcome"]


# Graficos del parallel_coordinates

plt.figure(figsize=(12, 6))

pd.plotting.parallel_coordinates(total_data, "Outcome", color = ("#E58139", "#39E581", "#8139E5"))

plt.show()



from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(random_state = 42)
model.fit(X_train, y_train)

# grafico del arbol
from sklearn import tree

fig = plt.figure(figsize=(15,15))

tree.plot_tree(model, feature_names = list(X_train.columns), class_names = ["0", "1", "2"], filled = True)

plt.show()


# Numero de nodos terminales
num_nodos_terminales = model.tree_.n_leaves
print("Numero de nodos terminales:", num_nodos_terminales)


# Funcion niveles de Arbol

def get_tree_depth(tree_, feature_names=None):
    """
    Calcula la profundidad (número de niveles) de un árbol de decisión.

    Args:
        tree_: El árbol de decisión entrenado.
        feature_names: (Opcional) Nombres de las características.

    Returns:
        int: La profundidad del árbol.
    """

    children_left = tree_.children_left
    children_right = tree_.children_right

    def recurse(node):
        if children_left[node] < 0:
            return 0
        left_height = 1 + recurse(children_left[node])
        right_height = 1 + recurse(children_right[node])
        return max(left_height, right_height)

    return recurse(0)

# Salida de la funcion
depth = get_tree_depth(model.tree_)
print("La profundidad del árbol es:", depth)



y_pred = model.predict(X_test)

from sklearn.metrics import accuracy_score

accuracy_score(y_test, y_pred)

# modelo optimizado
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier

hyperparams = {
    "criterion": ["gini", "entropy"],
    "max_depth": [None, 5, 10, 20],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4]
}

grid = GridSearchCV(model, hyperparams, scoring = "accuracy", cv = 10)
grid

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

grid.fit(X_train, y_train)

# print(f"Lo mejores hiperparametros son: {grid.best_params_}")

model = DecisionTreeClassifier(criterion = "entropy", max_depth = 5, min_samples_leaf = 4, min_samples_split = 2, random_state = 42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

from sklearn.metrics import accuracy_score

accuracy_score(y_test, y_pred)

try:
    joblib.dump(model,"mymodel.joblib")
except Exception as e:
    print(f"Error: {e}")