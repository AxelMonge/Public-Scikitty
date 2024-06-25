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

import sys
import os

# Obtener el directorio actual del script.
directorio_actual = os.path.dirname(os.path.abspath(__file__))
# Obtener el directorio superior.
directorio_superior = os.path.abspath(os.path.join(directorio_actual, os.pardir))
# Se agrega el directorio superior a los paths que reconoce este archivo python.
sys.path.append(directorio_superior)

# --------------------------------------------------------------------------------- #
"""
    -----------------------SCRIPT breast_cancer SCIKITTY----------------------------
    En este ejemplo, demostraremos cómo utilizar nuestra implementación de la regresión logística.
    Usaremos el dataset público de "Breast Cancer" de la biblioteca sklearn, que es útil para probar algoritmos de clasificación.
    El flujo del código incluye cargar y normalizar los datos, dividir el dataset en conjuntos de entrenamiento y prueba,
    entrenar el modelo de regresión logística, realizar predicciones, evaluar el modelo y visualizar los resultados con una matriz
    de confusión.
    Nos parece muy interesante el rendimiento de la regresión logística y, se denota que, ya que dicha técnica de ML es la base
    para redes neuronales (perceptrón) y con este dataset tiene un rendimiento excelente, el deep learning nos resulta una
    técnica de ML asombrosa.
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from Scikitty.models.LogisticRegression import LogisticRegression

# Cargar el dataset de Breast Cancer
data = load_breast_cancer()
X, y = data.data, data.target

# Normalizar los datos para mejorar la convergencia del gradiente descendente
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Dividir el dataset en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Instanciar y entrenar el modelo
model = LogisticRegression(learning_rate=0.01, n_iterations=1000)
model.fit(X_train, y_train)

# Realizar predicciones sobre el conjunto de prueba
predictions = model.predict(X_test)

# Evaluar el modelo utilizando diferentes métricas de rendimiento
accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions)
recall = recall_score(y_test, predictions)
f1 = f1_score(y_test, predictions)

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")

# Generar la matriz de confusión
cm = confusion_matrix(y_test, predictions)
cm_display = sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')

print(f"VISUALIZANDO MATRIZ DE CONFUSIÓN 1 = positivo, 0 = negativo")

# Configurar etiquetas de los ejes
cm_display.set_xlabel('Predicted Labels')
cm_display.set_ylabel('True Labels')
cm_display.set_title('Confusion Matrix')

# Mostrar la gráfica
plt.show()
