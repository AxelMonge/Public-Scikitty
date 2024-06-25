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
-----------------------SCRIPT fictional_reading_place SCI-KIT LEARN----------------------------

    Este script demuestra el uso de varias funcionalidades en el módulo scikit-learn:
    - Cargar un dataset.
    - Codificar características categóricas.
    - Preparar y dividir los datos en conjuntos de entrenamiento y prueba.
    - Entrenar un modelo de árbol de decisión.
    - Visualizar el árbol de decisión.
    - Evaluar el modelo utilizando varias métricas.
"""
# --------------------------------------------------------------------------------- #

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import matplotlib.pyplot as plt

# Se almacena el nombre del archivo donde se guarda el dataset
file_name = 'fictional_reading_place'

# Cargar los datos
data = pd.read_csv(f'../datasets/{file_name}.csv')

# Preparar los datos.
features = data.drop('example', axis=1) # Eliminar la columna 'example' que es solo un indicador de instancias
features = features.drop('user_action', axis=1)  # Características del dataset
labels = data['user_action']

# Codificar características categóricas
encoder = OneHotEncoder()
features_encoded = encoder.fit_transform(features).toarray()

# Dividir los datos
X_train, X_test, y_train, y_test = train_test_split(features_encoded, labels, test_size=0.2, random_state=42)

# Crear e instanciar el árbol de decisión
dt = DecisionTreeClassifier(criterion='gini', min_samples_split=2, max_depth=5, random_state=42)
dt.fit(X_train, y_train)

# Visualizar el árbol
plt.figure(figsize=(12, 8))
plot_tree(dt, feature_names=encoder.get_feature_names_out().tolist(), class_names=dt.classes_.tolist(), filled=True)
plt.savefig(f'{file_name}_tree-scikitlearn.png')

# Imprimir resultados
y_pred = dt.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')
conf_matrix = confusion_matrix(y_test, y_pred)

print("\n------------------------------ ARBOL SCI-KIT ------------------------------\n")
print("Exactitud:", accuracy)
print("Precisión:", precision)
print("Recall:", recall)
print("F1-score:", f1)
print("Matriz de confusión:") 
print(conf_matrix)
print("Etiquetas predichas:", y_pred)
print("Etiquetas reales:", y_test.tolist())
print("\nVisualizando el árbol de Sci-Kit Learn...\n")
plt.show()