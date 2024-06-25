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
   -----------------------SCRIPT california_housing SCIKITTY----------------------------
    En este ejemplo, demostraremos cómo utilizar nuestra implementación desde cero de la regresión lineal.
    Usaremos el dataset público de "California Housing" de la biblioteca sklearn, que es útil para probar algoritmos de regresión.
    El flujo del código incluye cargar y normalizar los datos, dividir el dataset en conjuntos de entrenamiento y prueba,
    entrenar el modelo de regresión lineal, realizar predicciones, evaluar el modelo y visualizar los resultados.
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from Scikitty.models.LinearRegression import LinearRegression

# Cargar el dataset de California Housing
california = fetch_california_housing()
X, y = california.data, california.target

# Normalizar los datos para mejorar la convergencia del gradiente descendente
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Dividir el dataset en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Instanciar y entrenar el modelo
model = LinearRegression(learning_rate=0.01, n_iterations=1000)
model.fit(X_train, y_train)

# Realizar predicciones sobre el conjunto de prueba
predictions = model.predict(X_test)

# Evaluar el modelo utilizando el Error Cuadrático Medio (MSE), entre más cercano a 0, mejor desempeño del modelo
mse = np.mean((predictions - y_test) ** 2)
print(f"Mean Squared Error: {mse}")

# Visualizar las predicciones vs los valores reales
plt.scatter(y_test, predictions)
plt.xlabel("True Values")
plt.ylabel("Predictions")
plt.title("True Values vs Predictions")
plt.show()
