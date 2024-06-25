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

class LinearRegression:
    """
    Clase que implementa el algoritmo de regresión usando descenso por gradiente, bias y la fórmula simplificada de los gradientes para regresión lineal.

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
    >>> from linear_regression import LinearRegression
    >>> import numpy as np
    >>> X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    >>> y = np.dot(X, np.array([1, 2])) + 3
    >>> model = LinearRegression(learning_rate=0.01, n_iterations=1000)
    >>> model.fit(X, y)
    >>> predictions = model.predict(X)
    """

    def __init__(self, learning_rate=0.01, n_iterations=1000):
        """
        Inicializa la clase LinearRegression con la tasa de aprendizaje y el número de iteraciones.

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
        Ajusta el modelo de regresión lineal a los datos de entrenamiento.

        Parametros
        ----------
        X : np.ndarray
            Matriz de características del conjunto de entrenamiento.
        y : np.ndarray
            Vector de etiquetas del conjunto de entrenamiento.

        Ejemplos
        --------
        >>> X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
        >>> y = np.dot(X, np.array([1, 2])) + 3
        >>> model = LinearRegression(learning_rate=0.01, n_iterations=1000)
        >>> model.fit(X, y)
        """
        # Inicializa los parámetros
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Algoritmo de descenso de gradiente
        for _ in range(self.n_iterations):
            y_predicted = self._predict(X)

            # Calcula los gradientes
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            # Actualiza los parámetros
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def _predict(self, X):
        """
        Calcula las predicciones del modelo para una matriz de características dada.

        Parametros
        ----------
        X : np.ndarray
            Matriz de características.

        Retorna
        -------
        np.ndarray
            Vector de predicciones.

        Ejemplos
        --------
        >>> X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
        >>> model = LinearRegression(learning_rate=0.01, n_iterations=1000)
        >>> model.fit(X, y)
        >>> predictions = model._predict(X)
        """
        return np.dot(X, self.weights) + self.bias

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
        >>> X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
        >>> model = LinearRegression(learning_rate=0.01, n_iterations=1000)
        >>> model.fit(X, y)
        >>> predictions = model.predict(X)
        """
        return self._predict(X)
