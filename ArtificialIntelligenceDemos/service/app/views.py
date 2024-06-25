import csv
import numpy as np
import pandas as pd
import io
import base64
from base64 import b64encode
import matplotlib.pyplot as plt
import seaborn as sns
from django.shortcuts import render
from rest_framework.decorators import api_view
from rest_framework.response import Response
from scikitty.models.DecisionTree import DecisionTree
from scikitty.view.TreeVisualizer import TreeVisualizer
from scikitty.models.LinearRegression import LinearRegression
from scikitty.models.LogisticRegression import LogisticRegression
from scikitty.models.TreeBoosting import TreeBoosting
from scikitty.models.DecisionStump import DecisionStump
from scikitty.persist.TreePersistence import TreePersistence
from scikitty.metrics.accuracy_score import puntuacion_de_exactitud
from scikitty.metrics.precision_score import puntuacion_de_precision
from scikitty.metrics.recall_score import puntuacion_de_recall
from scikitty.metrics.f1_score import puntuacion_de_f1
from scikitty.metrics.confusion_matrix import matriz_de_confusion
from scikitty.model_selection.train_test_split import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.datasets import fetch_california_housing
from sklearn.datasets import load_breast_cancer
import os

# Add Graphviz path to the environment PATH
os.environ["PATH"] += os.pathsep + r"C:\Program Files\Graphviz\bin"

# Create your views here.

# Cambiar el backend de matplotlib a 'Agg' para evitar problemas de interfaz gráfica
plt.switch_backend('Agg')


@api_view(['POST'])
def transform_csv(request):
    data = request.data
    csv_data = data.get('csv')
    feature_target = data.get('featureTarget')
    file_name = data.get('fileName')
    criterion = data.get('criterion')
    create_csv(csv_data, file_name)
    metrics = create_tree(file_name, feature_target, criterion)
    return Response(metrics)


@api_view(['POST'])
def sklearn_dt(request):
    data = request.data
    file_name = data.get('fileName')
    csv_data = data.get('csv')
    feature_target = data.get('featureTarget')
    criterion = data.get('criterion')
    create_csv(csv_data, file_name)
    metrics = create_sklearn_dt(file_name, feature_target, criterion)
    return Response(metrics)


@api_view(['POST'])
def scikitty_linear_regression(request):
    data = request.data
    file_name = data.get('fileName')
    csv_data = data.get('csv')
    feature_target = data.get('featureTarget')
    create_csv(csv_data, file_name)
    mse, plot_base64 = _create_classification_model(file_name, feature_target)
    return Response({"mse": mse, "plot": plot_base64})


@api_view(['POST'])
def scikitty_logistic_regression(request):
    data = request.data
    file_name = data.get('fileName')
    csv_data = data.get('csv')
    feature_target = data.get('featureTarget')
    create_csv(csv_data, file_name)
    accuracy, precision, recall, f1, plot_base64 = _create_logistic_classification_model(
        file_name, feature_target)
    return Response({"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1, "plot": plot_base64})


@api_view(['POST'])
def scikitty_tree_boosting(request):
    data = request.data
    file_name = data.get('fileName')
    csv_data = data.get('csv')
    feature_target = data.get('featureTarget')
    criterion = data.get('criterion')
    create_csv(csv_data, file_name)
    metrics = _create_tree_boosting(file_name, feature_target, criterion)
    return Response(metrics)


@api_view(['GET'])
def scikitty_breast_cancer(request):
    accuracy, precision, recall, f1, plot_base64 = _create_logistic_regression_breast_cancer()
    return Response({"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1, "plot": plot_base64})


@api_view(['GET'])
def scikitty_california_housing(request):
    mse, plot_base64 = _create_linear_regression_california_housing()
    return Response({"mse": mse, "plot": plot_base64})


def _create_linear_regression_california_housing():
    # Cargar el dataset de California Housing
    california = fetch_california_housing()
    X, y = california.data, california.target

    # Normalizar los datos para mejorar la convergencia del gradiente descendente
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Dividir el dataset en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Instanciar y entrenar el modelo
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Realizar predicciones sobre el conjunto de prueba
    predictions = model.predict(X_test)

    # Evaluar el modelo utilizando el Error Cuadrático Medio (MSE), entre más cercano a 0, mejor desempeño del modelo
    mse = np.mean((predictions - y_test) ** 2)
    print(f"Mean Squared Error: {mse}")

    # Visualizar las predicciones vs los valores reales y guardar la imagen en un buffer
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, predictions)
    plt.xlabel("True Values")
    plt.ylabel("Predictions")
    plt.title("True Values vs Predictions")

    # Guardar la imagen en un buffer de bytes
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plot_base64 = base64.b64encode(buffer.read()).decode('utf-8')
    buffer.close()

    # Cerrar la figura para liberar la memoria
    plt.close()

    return mse, plot_base64

# def _create_linear_regression_california_housing():

#     # Cargar el dataset de California Housing
#     california = fetch_california_housing()
#     X, y = california.data, california.target

#     # Normalizar los datos para mejorar la convergencia del gradiente descendente
#     scaler = StandardScaler()
#     X = scaler.fit_transform(X)

#     # Dividir el dataset en conjuntos de entrenamiento y prueba
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.2, random_state=42)

#     # Instanciar y entrenar el modelo
#     model = LinearRegression()
#     model.fit(X_train, y_train)

#     # Realizar predicciones sobre el conjunto de prueba
#     predictions = model.predict(X_test)

#     # Convertir los valores reales y predicciones en categorías discretas
#     bins = np.linspace(min(y_test), max(y_test), 5)
#     y_test_binned = np.digitize(y_test, bins)
#     predictions_binned = np.digitize(predictions, bins)

#     # Generar la matriz de confusión
#     cm = confusion_matrix(y_test_binned, predictions_binned)
#     plt.figure(figsize=(10, 6))
#     cm_display = sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')

#     # Evaluar el modelo utilizando el Error Cuadrático Medio (MSE), entre más cercano a 0, mejor desempeño del modelo
#     mse = np.mean((predictions - y_test) ** 2)
#     print(f"Mean Squared Error: {mse}")

#     print("VISUALIZANDO MATRIZ DE CONFUSIÓN")

#     # Configurar etiquetas de los ejes
#     cm_display.set_xlabel('Predicted Labels')
#     cm_display.set_ylabel('True Labels')
#     cm_display.set_title('Confusion Matrix')

#     # Guardar la imagen en un buffer de bytes
#     buffer = io.BytesIO()
#     plt.savefig(buffer, format='png')
#     buffer.seek(0)
#     plot_base64 = base64.b64encode(buffer.read()).decode('utf-8')
#     buffer.close()

#     # Cerrar la figura para liberar la memoria
#     plt.close()

#     return mse, plot_base64


def _create_logistic_regression_breast_cancer():
    # Cargar el dataset de Breast Cancer
    data = load_breast_cancer()
    X, y = data.data, data.target

    # Normalizar los datos para mejorar la convergencia del gradiente descendente
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Dividir el dataset en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

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
    plt.figure(figsize=(10, 6))
    cm_display = sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')

    print("VISUALIZANDO MATRIZ DE CONFUSIÓN 1 = positivo, 0 = negativo")

    # Configurar etiquetas de los ejes
    cm_display.set_xlabel('Predicted Labels')
    cm_display.set_ylabel('True Labels')
    cm_display.set_title('Confusion Matrix')

    # Guardar la imagen en un buffer de bytes
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plot_base64 = base64.b64encode(buffer.read()).decode('utf-8')
    buffer.close()

    # Cerrar la figura para liberar la memoria
    plt.close()

    return accuracy, precision, recall, f1, plot_base64


def _create_tree_boosting(file_name, feature_target, criterion):

    # Cargar los datos.
    data = pd.read_csv(f'{file_name}.csv')

    # Preparar los datos.
    # Asume que 'Disease' es la columna objetivo
    features = data.drop(feature_target, axis=1)
    labels = data[feature_target]

    # Dividir los datos.
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42)

    # Crear e instanciar el modelo de boosting.
    tb = TreeBoosting(n_estimators=4, learning_rate=0.05,
                      criterio=criterion, criterio_continuo='MSE')
    tb.fit(X_train, y_train)

    # Imprimir resultados.
    y_pred = tb.predict(X_test)

    # Se calculan las métricas.
    accuracy = puntuacion_de_exactitud(y_test, y_pred)
    precision = puntuacion_de_precision(y_test, y_pred, average='weighted')
    recall = puntuacion_de_recall(y_test, y_pred, average='weighted')
    f1 = puntuacion_de_f1(y_test, y_pred, average='weighted')
    # Calcular la matriz de confusión
    conf_mat, classes = _scikitty_confusion_matrix(y_test, y_pred)

    # Crear la matriz con los títulos
    conf_mat_with_titles = _create_confusion_matrix_with_titles(
        conf_mat, classes)

    # Se imprimen los resultados por consola.
    print("\n------------------------------ BOOSTING SCIKITTY ------------------------------\n")
    print("Exactitud:", accuracy)
    print("Precisión:", precision)
    print("Recall:", recall)
    print("F1-score:", f1)
    print("Matriz de confusión:")
    print(conf_mat_with_titles)
    print("Etiquetas predichas:", y_pred)
    print("Etiquetas reales:", y_test.tolist())

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "conf_matrix": conf_mat_with_titles,
        "features": y_pred,
        "real_features": y_test.tolist()
    }
    return metrics


def _create_logistic_classification_model(file_name, target_column):
    # Cargar los datos desde el archivo CSV
    df = pd.read_csv(f'{file_name}.csv')

    # Separar características y objetivo
    X = df.drop(target_column, axis=1)
    y = df[target_column]

    # Codificar variables categóricas en características
    encoder = OneHotEncoder()
    X = encoder.fit_transform(X).toarray()

    # Codificar la columna objetivo
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    # Normalizar las características
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Dividir el dataset en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # Instanciar y entrenar el modelo de clasificación
    model = LogisticRegression()
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
    plt.figure(figsize=(10, 6))
    cm_display = sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')

    print("VISUALIZANDO MATRIZ DE CONFUSIÓN 1 = positivo, 0 = negativo")

    # Configurar etiquetas de los ejes
    cm_display.set_xlabel('Predicted Labels')
    cm_display.set_ylabel('True Labels')
    cm_display.set_title('Confusion Matrix')

    # Guardar la imagen en un buffer de bytes
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plot_base64 = base64.b64encode(buffer.read()).decode('utf-8')
    buffer.close()

    # Cerrar la figura para liberar la memoria
    plt.close()

    return accuracy, precision, recall, f1, plot_base64


def _create_classification_model(file_name, target_column):
    # Cargar los datos desde el archivo CSV
    df = pd.read_csv(f'{file_name}.csv')

    # Separar características y objetivo
    X = df.drop(target_column, axis=1)
    y = df[target_column]

    # Codificar variables categóricas en características
    encoder = OneHotEncoder()
    X = encoder.fit_transform(X).toarray()

    # Codificar la columna objetivo
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    # Normalizar las características
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Dividir el dataset en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # Instanciar y entrenar el modelo de clasificación
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Realizar predicciones sobre el conjunto de prueba
    predictions = model.predict(X_test)

    # Evaluar el modelo utilizando la precisión (accuracy)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Accuracy: {accuracy}")

    # Visualizar las predicciones vs los valores reales y guardar la imagen en un buffer
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, predictions)
    plt.xlabel("True Values")
    plt.ylabel("Predictions")
    plt.title("True Values vs Predictions")

    # Guardar la imagen en un buffer de bytes
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plot_base64 = base64.b64encode(buffer.read()).decode('utf-8')
    buffer.close()

    # Cerrar la figura para liberar la memoria
    plt.close()

    return accuracy, plot_base64


def create_csv(data, file):
    csv_filename = file + ".csv"
    with open(csv_filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data)
    return csv_filename


def create_tree(file_name, featureTarget, criterion):
    # Cargar los datos.
    data = pd.read_csv(f'{file_name}.csv')

    # Preparar los datos.
    features = data.drop(featureTarget, axis=1)
    labels = data[featureTarget]

    # Dividir los datos.
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42)

    # Crear e instanciar el árbol de decisión.
    criterio_impureza = criterion
    min_muestras_div = 2
    max_profundidad = 5
    dt = DecisionTree(X_train, y_train, criterio=criterio_impureza,
                      min_muestras_div=min_muestras_div, max_profundidad=max_profundidad)
    dt.fit()

    # ---------------------------------------------------------

    def test_tree(dt, file_name, X_test, y_test):

        # Visualizar el árbol.
        tree_structure = dt.get_tree_structure()
        visualizer = TreeVisualizer()
        visualizer.graph_tree(tree_structure)
        visualizer.get_graph(f'{file_name}_tree', ver=False)

        # Convertir la imagen a base64
        with open(f'{file_name}_tree.png', 'rb') as img_file:
            img_base64 = b64encode(img_file.read()).decode('utf-8')

        # Imprimir resultados.
        y_pred = dt.predict(X_test)

        # Se calculan las metricas.
        accuracy = puntuacion_de_exactitud(y_test, y_pred)
        precision = puntuacion_de_precision(y_test, y_pred, average='weighted')
        recall = puntuacion_de_recall(y_test, y_pred, average='weighted')
        f1 = puntuacion_de_f1(y_test, y_pred, average='weighted')

        # Calcular la matriz de confusión
        conf_mat, classes = _scikitty_confusion_matrix(y_test, y_pred)

        # Crear la matriz con los títulos
        conf_mat_with_titles = _create_confusion_matrix_with_titles(
            conf_mat, classes)

        # Imprimir la matriz con títulos
        print(conf_mat_with_titles)

        # Se imprimen los resultados por consola.
        print("\n------------------------------ ARBOL ORIGINAL SCIKITTY ------------------------------\n")
        print("¿El árbol es balanceado?", dt.is_balanced())
        print("Exactitud:", accuracy)
        print("Precisión:", precision)
        print("Recall:", recall)
        print("F1-score:", f1)
        print("Matriz de confusión:")
        print(conf_mat_with_titles)
        print("Etiquetas predichas:", y_pred)
        print("Etiquetas reales:", y_test.tolist())
        print("\nVisualizando el árbol original...\n")

        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "conf_matrix": conf_mat_with_titles,
            "features": y_pred,
            "real_features": y_test.tolist(),
            "is_balanced": dt.is_balanced(),
            "plot_base64": img_base64
        }
        return metrics

    # ---------------------------------------------------------

    treeMetrics = test_tree(dt, file_name, X_test, y_test)

    # Guardar el árbol en un archivo JSON
    TreePersistence.save_tree(dt, f'{file_name}.json')

    # Cargar el árbol desde el archivo JSON
    nueva_raiz = TreePersistence.load_tree(f'{file_name}.json')
    nuevo_dt = DecisionTree(X_train, y_train, criterio=criterio_impureza,
                            min_muestras_div=min_muestras_div, max_profundidad=max_profundidad)
    nuevo_dt.set_tree(nueva_raiz)

    test_tree(nuevo_dt, file_name, X_test, y_test)

    return treeMetrics


def create_sklearn_dt(file_name, featureTarget, criterion):
    data = pd.read_csv(f'{file_name}.csv')

    # Preparar los datos.
    features = data.drop(featureTarget, axis=1)
    labels = data[featureTarget]

    # Codificar las variables categóricas.
    encoder = OneHotEncoder()
    features_encoded = encoder.fit_transform(features)

    # Dividir los datos.
    X_train, X_test, y_train, y_test = train_test_split(
        features_encoded, labels, test_size=0.2, random_state=42)

    # Crear e instanciar el árbol de decisión.
    dt = DecisionTreeClassifier(
        criterion=criterion, min_samples_split=2, max_depth=5, random_state=42)
    dt.fit(X_train, y_train)

    # Visualizar el árbol.
    plt.figure(figsize=(10, 6))
    plot_tree(dt, feature_names=encoder.get_feature_names_out(
    ).tolist(), class_names=dt.classes_.tolist(), filled=True)
    plt.savefig(f'{file_name}_tree-scikitlearn.png')

    # Convertir la imagen a base64
    with open(f'{file_name}_tree-scikitlearn.png', 'rb') as img_file:
        img_base64 = b64encode(img_file.read()).decode('utf-8')

    # Imprimir resultados.
    y_pred = dt.predict(X_test)

    # Se calculan las metricas.
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Crear la matriz con los títulos
    conf_mat_with_titles = _create_confusion_matrix_with_titles(
        conf_matrix, dt.classes_)

    # Imprimir la matriz con títulos
    print(conf_mat_with_titles)

    # Se imprimen los resultados por consola.
    print("\n------------------------------ ARBOL SCI-KIT ------------------------------\n")
    print("Exactitud:", accuracy)
    print("Precisión:", precision)
    print("Recall:", recall)
    print("F1-score:", f1)
    print("Matriz de confusión:")
    print(conf_mat_with_titles)
    print("Etiquetas predichas:", y_pred)
    print("Etiquetas reales:", y_test.tolist())
    print("\nVisualizando el árbol de Sci-Kit Learn...\n")

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "conf_matrix": conf_mat_with_titles,
        "features": y_pred,
        "real_features": y_test.tolist(),
        "plot_base64": img_base64
    }
    return metrics


def _scikitty_confusion_matrix(y_true, y_pred):
    # Obtener los valores únicos que pueden tomar las clases
    classes = np.unique(np.concatenate((y_true, y_pred)))
    num_classes = len(classes)

    # Mapear las etiquetas a índices numéricos
    class_to_index = {label: i for i, label in enumerate(classes)}
    y_true_mapped = np.array([class_to_index[label] for label in y_true])
    y_pred_mapped = np.array([class_to_index[label] for label in y_pred])

    # Inicializar la matriz de confusión con ceros
    confusion_mat = np.zeros((num_classes, num_classes), dtype=int)

    # Rellenar la matriz de confusión
    for i in range(len(y_true_mapped)):
        true_class = y_true_mapped[i]
        pred_class = y_pred_mapped[i]
        confusion_mat[true_class, pred_class] += 1

    return confusion_mat, classes


def _create_confusion_matrix_with_titles(confusion_mat, classes):
    # Crear una matriz con una fila y una columna adicionales para los títulos
    num_classes = len(classes)
    mat_with_titles = np.empty(
        (num_classes + 1, num_classes + 1), dtype=object)

    # Agregar los títulos de las columnas
    mat_with_titles[0, 1:] = classes
    # Agregar los títulos de las filas
    mat_with_titles[1:, 0] = classes

    # Agregar los valores de la matriz de confusión
    mat_with_titles[1:, 1:] = confusion_mat

    # El título de la esquina superior izquierda puede dejarse vacío
    mat_with_titles[0, 0] = "Down: Real / Right: Predicted"

    return mat_with_titles
