# --------------------------------------------------------------------------------- #
"""
    Autores:
    1) Nombre: John Rojas Chinchilla
       ID: 118870938
       Correo: john.rojas.chinchilla@est.una.ac.cr
       Horario: 1pm

    2) Nombre: Abigail Salas Ramirez
       ID: 402570890
       Correo: abigail.salas.ramirez@est.una.ac.cr
       Horario: 1pm

    3) Nombre: Axel Monge Ramirez
       ID: 118640655
       Correo: axel.monge.ramirez@est.una.ac.cr
       Horario: 1pm

    4) Nombre: Andrel Ramirez Solis
       ID: 118460426
       Correo: andrel.ramirez.solis@est.una.ac.cr
       Horario: 1pm
"""
# --------------------------------------------------------------------------------- #
"""
-----------------------SCRIPT playTennis Decision Stump----------------------------

    Este script demuestra el uso de varias funcionalidades en el módulo Scikitty:
    - Cargar un dataset.
    - Preparar y dividir los datos en conjuntos de entrenamiento y prueba.
    - Entrenar un modelo de stump de decisión.
    - Evaluar el modelo utilizando varias métricas.
    - Guardar y cargar el modelo de stump de decisión en/desde un archivo JSON.
    - Verificar que el stump cargado es funcionalmente equivalente al stump original.
"""
# --------------------------------------------------------------------------------- #

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Obtener el directorio actual del script.
directorio_actual = os.path.dirname(os.path.abspath(__file__))
# Obtener el directorio superior.
directorio_superior = os.path.abspath(os.path.join(directorio_actual, os.pardir))
# Se agrega el directorio superior a los paths que reconoce este archivo python.
sys.path.append(directorio_superior)

# --------------------------------------------------------------------------------- #

from Scikitty.models.DecisionStump import DecisionStump
from Scikitty.view.TreeVisualizer import TreeVisualizer
from Scikitty.persist.TreePersistence import TreePersistence

# Se almacena el nombre del archivo donde se guarda el dataset.
file_name = 'playTennis'

# Cargar los datos.
data = pd.read_csv(f'../datasets/{file_name}.csv')

# Preparar los datos.
features = data.drop('Play Tennis', axis=1)  # Asume que 'Play Tennis' es la columna objetivo
labels = data['Play Tennis']

# Convert categorical features to numerical values
features = pd.get_dummies(features)

# Dividir los datos.
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Crear e instanciar el stump de decisión.
criterio_impureza = 'entropy'
min_muestras_div = 2
max_profundidad = 1
ds = DecisionStump(X_train, y_train, criterio=criterio_impureza, 
                   min_muestras_div=min_muestras_div, max_profundidad=max_profundidad)
ds.fit()

# Predecir etiquetas y probabilidades.
y_pred, y_prob = ds.predict(X_test)
class_probabilities = ds.predict_class_probabilities()

# Calcular las métricas.
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')
conf_matrix = confusion_matrix(y_test, y_pred)

# Imprimir los resultados por consola.
print("\n------------------------------ DECISION STUMP ------------------------------\n")
print("Exactitud:", accuracy)
print("Precisión:", precision)
print("Recall:", recall)
print("F1-score:", f1)
print("Matriz de confusión:")
print(conf_matrix)
print("Etiquetas predichas:", y_pred)
print("Probabilidades predichas:", y_prob)
print("Etiquetas reales:", y_test.tolist())
print("\nProbabilidades por clase:")
for class_label, prob in class_probabilities.items():
    print(f"Clase: {class_label}, Probabilidad: {prob}")

# Visualizar matriz de confusión.
plt.figure(figsize=(8, 6))
plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Matriz de Confusión')
plt.colorbar()
tick_marks = np.arange(len(ds.etiquetas_originales))
plt.xticks(tick_marks, ds.etiquetas_originales, rotation=45)
plt.yticks(tick_marks, ds.etiquetas_originales)
fmt = 'd'
thresh = conf_matrix.max() / 2.
for i, j in np.ndindex(conf_matrix.shape):
    plt.text(j, i, format(conf_matrix[i, j], fmt),
             horizontalalignment="center",
             color="white" if conf_matrix[i, j] > thresh else "black")
plt.ylabel('Etiqueta verdadera')
plt.xlabel('Etiqueta predicha')
plt.tight_layout()
plt.show()

# Visualizar el árbol.
tree_structure = ds.get_tree_structure()
visualizer = TreeVisualizer()
visualizer.graph_tree(tree_structure)
visualizer.get_graph(f'STUMPTENNIS_tree', ver=True)
