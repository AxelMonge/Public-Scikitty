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
-----------------------SCRIPT Fake Weights BOOSTING SCIKITTY----------------------------

    Este script demuestra el uso de varias funcionalidades en el módulo Scikitty:
    - Cargar un dataset.
    - Preparar y dividir los datos en conjuntos de entrenamiento y prueba.
    - Entrenar un modelo de tree gradient boosting con target contínuo con stumps.
    - Evaluar el modelo utilizando varias métricas.
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

from Scikitty.models.TreeBoosting import TreeBoosting
from Scikitty.metrics.accuracy_score import puntuacion_de_exactitud
from Scikitty.metrics.precision_score import puntuacion_de_precision
from Scikitty.metrics.recall_score import puntuacion_de_recall
from Scikitty.metrics.f1_score import puntuacion_de_f1
from Scikitty.metrics.confusion_matrix import matriz_de_confusion
from Scikitty.model_selection.train_test_split import train_test_split
import pandas as pd

# Se almacena el nombre del archivo donde se guarda el dataset.
file_name = 'fake_weights'

# Cargar los datos.
data = pd.read_csv(f'../datasets/{file_name}.csv', delimiter=';')

# Preparar los datos.
features = data.drop(['id', 'Weight(y)'], axis=1)  # Asume que 'Weight(y)' es la columna objetivo y elimina 'id'
labels = data['Weight(y)']

# Dividir los datos.
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Crear e instanciar el modelo de boosting.
n_estimators = 30
learning_rate = 1
criterio_impureza = 'entropy'
criterio_continuo = 'MSE'
tb = TreeBoosting(n_estimators=n_estimators, learning_rate=learning_rate, criterio=criterio_impureza, criterio_continuo=criterio_continuo)
tb.fit(X_train, y_train)

# ---------------------------------------------------------

def test_boosting(tb, file_name, X_test, y_test):
    print(f"{y_test=:}")
    # Imprimir resultados.
    y_pred = tb.predict(X_test)

    # Se calculan las metricas.
    accuracy = puntuacion_de_exactitud(y_test, y_pred)
    precision = puntuacion_de_precision(y_test, y_pred, average='weighted')
    recall = puntuacion_de_recall(y_test, y_pred, average='weighted')
    f1 = puntuacion_de_f1(y_test, y_pred, average='weighted')
    conf_matrix = matriz_de_confusion(y_test, y_pred)

    # Se imprimen los resultados por consola.
    print("\n------------------------------ BOOSTING SCIKITTY ------------------------------\n")
    print("Exactitud:", accuracy)
    print("Precisión:", precision)
    print("Recall:", recall)
    print("F1-score:", f1)
    print("Matriz de confusión:")
    print(conf_matrix)
    print("Etiquetas predichas:", y_pred)
    print("Etiquetas reales:", y_test.tolist())

# ---------------------------------------------------------

test_boosting(tb, file_name, X_test, y_test)
