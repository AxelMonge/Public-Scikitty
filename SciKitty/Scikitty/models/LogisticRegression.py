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

import numpy as np
from Scikitty.models.LinearRegression import LinearRegression

class LogisticRegression:
    """
    Clase que implementa el algoritmo de regresión logística usando descenso por gradiente y la función sigmoide.
    La implementación es la base de las redes neuronales y el perceptrón.

    Atributos
    ---------
    learning_rate : float
        Tasa de aprendizaje para el algoritmo de optimización.
    n_iterations : int
        Número de iteraciones para el algoritmo de optimización.
    weights : np.ndarray
        Pesos del modelo (coeficientes).
    bias : float
        Sesgo del modelo (intercepto).

    Ejemplos
    --------
    >>> from logistic_regression import LogisticRegression
    >>> import numpy as np
    >>> X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
    >>> y = np.array([0, 0, 1, 1])
    >>> model = LogisticRegression(learning_rate=0.01, n_iterations=1000)
    >>> model.fit(X, y)
    >>> predictions = model.predict(X)
    """

    def __init__(self, learning_rate=0.01, n_iterations=1000):
        """
        Inicializa la clase LogisticRegression con la tasa de aprendizaje y el número de iteraciones.

        Parametros
        ----------
        learning_rate : float, opcional
            Tasa de aprendizaje para el algoritmo de optimización (por defecto es 0.01).
        n_iterations : int, opcional
            Número de iteraciones para el algoritmo de optimización (por defecto es 1000).
        """
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        """
        Ajusta el modelo de regresión logística a los datos de entrenamiento.

        Parametros
        ----------
        X : np.ndarray
            Matriz de características del conjunto de entrenamiento.
        y : np.ndarray
            Vector de etiquetas del conjunto de entrenamiento.

        Ejemplos
        --------
        >>> X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
        >>> y = np.array([0, 0, 1, 1])
        >>> model = LogisticRegression(learning_rate=0.01, n_iterations=1000)
        >>> model.fit(X, y)
        """
        # Inicializa los parámetros
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Algoritmo de descenso de gradiente
        for _ in range(self.n_iterations):
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self._sigmoid(linear_model)

            # Calcula los gradientes
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            # Actualiza los parámetros
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        """
        Realiza predicciones utilizando el modelo ajustado.

        Parametros
        ----------
        X : np.ndarray
            Matriz de características del conjunto de prueba.

        Retorna
        -------
        np.ndarray
            Vector de predicciones.

        Ejemplos
        --------
        >>> X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
        >>> model = LogisticRegression(learning_rate=0.01, n_iterations=1000)
        >>> model.fit(X, y)
        >>> predictions = model.predict(X)
        """
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self._sigmoid(linear_model)
        y_predicted_class = [1 if i > 0.5 else 0 for i in y_predicted]
        return np.array(y_predicted_class)

    def _sigmoid(self, x):
        """
        Calcula la función sigmoide.

        Parametros
        ----------
        x : np.ndarray
            Entrada para calcular la función sigmoide.

        Retorna
        -------
        np.ndarray
            Resultado de la función sigmoide.

        Ejemplos
        --------
        >>> model = LogisticRegression()
        >>> model._sigmoid(np.array([0, 2]))
        """
        return 1 / (1 + np.exp(-x))
