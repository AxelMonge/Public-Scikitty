# --------------------------------------------------------------------------------- #
"""
    Autores:
    1) Nombre: John Rojas Chinchilla
       ID: 118870938
       Correo: john.rojas.chinchilla@est.una.ac.cr
       Horario: 1pm

    2) Nombre: Abigail Salas Ramírez
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
from Scikitty.models.DecisionStump import DecisionStump

class TreeBoosting:
    """
        Implementación del algoritmo de Tree Boosting para clasificación y regresión.

        Atributos
        ---------
        n_estimators : int
            Número de árboles o stumps a entrenar.
        learning_rate : float
            Tasa de aprendizaje para ajustar las predicciones.
        criterio : str
            Criterio utilizado para medir la calidad de las divisiones en clasificación.
        criterio_continuo : str
            Criterio utilizado para medir la calidad de las divisiones en regresión.
        stumps : list
            Lista de stumps (árboles de decisión) entrenados.
        gammas : list
            Lista de valores gamma utilizados para ajustar las predicciones en regresión.
        first_stump : DecisionStump
            Primer stump utilizado para inicializar las probabilidades de clase en clasificación.
        has_decimals : bool
            Indica si las etiquetas de entrenamiento tienen decimales (solo para regresión).
        is_continuous : bool
            Indica si el problema es de regresión (True) o clasificación (False).
        classes : array
            Array de clases únicas en el conjunto de datos de entrenamiento (solo para clasificación).
        """
    
    """
        Resumen del Flujo del Código

        Este código implementa el algoritmo de Tree Boosting para clasificación y regresión. A continuación, se detalla el flujo del código y las diferencias clave entre los casos continuos y categóricos:

        1. Inicialización:
            - El modelo se inicializa con los parámetros especificados, como el número de estimadores (n_estimators), la tasa de aprendizaje (learning_rate), y los criterios para medir la calidad de las divisiones (criterio y criterio_continuo).

        2. Entrenamiento (fit):
            - Se determinan las características del problema (regresión o clasificación) en función de las etiquetas (y).
            - Para problemas continuos (regresión):
                - Se inicializa F0 como la media de las etiquetas.
                - En cada iteración, se calculan los residuos entre las etiquetas y las predicciones actuales.
                - Se entrena un stump usando los residuos como etiquetas y se calculan las predicciones del stump.
                - Se encuentra el valor óptimo de gamma para ajustar las predicciones y se actualizan las predicciones usando gamma.
                - Los stumps y los valores de gamma se almacenan.
            - Para problemas categóricos (clasificación):
                - Se entrena un primer stump para calcular las probabilidades iniciales de clase.
                - En cada iteración, se calculan los residuos entre las etiquetas y las probabilidades predichas actuales.
                - Se entrena un stump para cada clase usando los residuos como etiquetas y se calculan las predicciones del stump.
                - Las predicciones se actualizan directamente usando los residuos y la tasa de aprendizaje (sin usar gamma para evitar underfitting).
                - Los stumps se almacenan.

        3. Predicción (predict):
            - Para problemas continuos (regresión):
                - Se inicializa F0 como la media de las etiquetas y se ajustan las predicciones iterativamente usando los stumps y los valores de gamma almacenados.
            - Para problemas categóricos (clasificación):
                - Se utiliza el primer stump para inicializar las probabilidades de clase.
                - Las predicciones se ajustan iterativamente usando los stumps almacenados para predecir el residuo del árbol anterior, corrigiendo sus errores. Esto afecta las probabilidades de cada clase en cada ejemplo.
                - Las clases finales se determinan tomando la clase con la probabilidad más alta.

        4. Residuos y Actualización de Predicciones:
            - En regresión, los residuos se calculan como la diferencia directa entre las etiquetas y las predicciones.
            - En clasificación, los residuos se calculan como la diferencia entre las etiquetas verdaderas (convertidas a 0 y 1) y las probabilidades predichas.
            - En regresión, se utilizó gamma para regularizar las actualizaciones, mientras que en clasificación se actualizan las predicciones directamente usando los residuos y la tasa de aprendizaje.
    """

    def __init__(self, n_estimators=100, learning_rate=0.1, criterio='entropy', criterio_continuo='MSE'):
        """
        Inicializa el modelo de Tree Boosting con los parámetros especificados.

        Parametros
        ----------
        n_estimators : int, opcional (default=100)
            Número de árboles o stumps a entrenar.
        learning_rate : float, opcional (default=0.1)
            Tasa de aprendizaje para ajustar las predicciones.
        criterio : str, opcional (default='entropy')
            Criterio utilizado para medir la calidad de las divisiones en clasificación.
        criterio_continuo : str, opcional (default='MSE')
            Criterio utilizado para medir la calidad de las divisiones en regresión.
        """
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.criterio = criterio
        self.criterio_continuo = criterio_continuo
        self.stumps = []
        self.gammas = []
        self.first_stump = None  # Variable para almacenar el primer stump usado en objetivos categóricos
        self.has_decimals = None
        self.is_continuous = None
        self.classes = None

    def fit(self, X, y):
        """
        Entrena el modelo de Tree Boosting utilizando los datos de entrenamiento proporcionados.
        
        Para problemas continuos (regresión), se calcula un valor gamma óptimo para ajustar las predicciones. gamma se utiliza para regularizar las actualizaciones.
        Para problemas categóricos (clasificación), no se utiliza gamma para evitar el underfitting y porque mejora las métricas. En su lugar, las predicciones se ajustan directamente usando los residuos.

        Parametros
        ----------
        X : array-like, shape (n_samples, n_features)
            Conjunto de datos de entrenamiento.
        y : array-like, shape (n_samples,)
            Etiquetas de entrenamiento.
        """
        y = np.array(y)
        
        # Determinar si el problema es de regresión o clasificación
        self.is_continuous = len(np.unique(y)) > 2 and not isinstance(y[0], str)
        print(f"Target continuo: {self.is_continuous}")
        print(f"TARGET A PREDECIR EN FIT= {y}")

        if self.is_continuous:
            # Verificar si las etiquetas tienen decimales
            self.has_decimals = not np.all(y == y.astype(int))
            print(f"Etiquetas originales tienen decimales: {self.has_decimals}")

            # Inicializar F0 como la media de las etiquetas para regresión
            self.F0 = np.mean(y)
            F_m = np.full(y.shape, self.F0)
        else:
            # Inicializar F0 usando proporciones de clases para clasificación
            self.classes = np.unique(y)
            self.F0 = self._initialize_F0_classification(y)
            self.first_stump = DecisionStump(X, y, criterio=self.criterio, criterio_continuo=self.criterio_continuo)
            self.first_stump.fit()
            F_m = self.first_stump.predict_class_probabilities_X(X)
        print(f"Inicialización de probabilidades de clases: {F_m}")

        for m in range(self.n_estimators):
            # Calcular los residuos entre las etiquetas y las predicciones actuales
            residuals = self._compute_residuals(y, F_m)
            print(f"Iteración {m + 1}")
            print(f"Residuals: {residuals}")

            if self.is_continuous:
                # Entrenar un stump usando los residuos como etiquetas para regresión
                stump = DecisionStump(X, residuals, criterio=self.criterio, criterio_continuo=self.criterio_continuo)
                stump.fit()
                predictions, _ = stump.predict(X)
                predictions = np.array(predictions)
                print(f"Predicciones del stump: {predictions}")

                # Calcular el valor óptimo de gamma
                gamma_m = self._find_optimal_gamma(y, F_m, predictions, self.is_continuous)
                print(f"Gamma {m + 1}: {gamma_m}")

                # Actualizar las predicciones usando gamma
                F_m += self.learning_rate * gamma_m * predictions
                print(f"Nuevas predicciones F_m: {F_m}")

                # Almacenar el stump y gamma
                self.stumps.append(stump)
                self.gammas.append(gamma_m)
            else:
                # Ajustar las predicciones directamente usando los residuos para clasificación
                stumps_m = []
                for k, class_k in enumerate(self.classes):
                    stump = DecisionStump(X, residuals[:, k], criterio=self.criterio, criterio_continuo=self.criterio_continuo)
                    stump.fit()
                    predictions, _ = stump.predict(X)
                    predictions = np.array(predictions)
                    print(f"Predicciones del stump para la clase {class_k}: {predictions}")

                    # Actualizar las predicciones usando los residuos directamente
                    F_m[:, k] += self.learning_rate * residuals[:, k]
                    print(f"Nuevas predicciones F_m para la clase {class_k}: {F_m[:, k]}")

                    stumps_m.append(stump)

                # Almacenar los stumps
                self.stumps.append(stumps_m)
            print("------------------------------------------------")

    def _initialize_F0_classification(self, y):
        """
        Inicializa F0 para clasificación utilizando la proporción de clases.

        Parametros
        ----------
        y : array-like, shape (n_samples,)
            Etiquetas de entrenamiento.

        Returns
        -------
        F0 : array, shape (n_classes,)
            Inicialización de F0 para cada clase.
        """
        unique_classes, counts = np.unique(y, return_counts=True)
        p = counts / counts.sum()
        return np.log(p / (1 - p))

    def _compute_residuals(self, y, F_m):
        """
        Calcula los residuos entre las etiquetas verdaderas y las predicciones actuales.

        Parametros
        ----------
        y : array-like, shape (n_samples,)
            Etiquetas de entrenamiento.
        F_m : array-like, shape (n_samples,) or (n_samples, n_classes)
            Predicciones actuales del modelo.

        Returns
        -------
        residuals : array-like, shape (n_samples,) or (n_samples, n_classes)
            Residuos entre las etiquetas verdaderas y las predicciones actuales.
        """
        if self.is_continuous:
            # Residuos para regresión
            return y - F_m
        else:
            # Residuos para clasificación
            residuals = np.zeros_like(F_m)
            for k, class_k in enumerate(self.classes):
                p_k = 1 / (1 + np.exp(-F_m[:, k]))
                residuals[:, k] = (y == class_k).astype(int) - p_k
            return residuals

    def _find_optimal_gamma(self, y, F_m_class, predictions, is_continuous):
        """
        Calcula el valor óptimo de gamma para ajustar las predicciones en regresión.

        Parametros
        ----------
        y : array-like, shape (n_samples,)
            Etiquetas de entrenamiento.
        F_m_class : array-like, shape (n_samples,)
            Predicciones actuales del modelo para una clase.
        predictions : array-like, shape (n_samples,)
            Predicciones del stump actual.
        is_continuous : bool
            Indica si el problema es de regresión (True) o clasificación (False).

        Returns
        -------
        gamma : float
            Valor óptimo de gamma para ajustar las predicciones.
        """
        if is_continuous:
            return np.sum((y - F_m_class) * predictions) / np.sum(predictions ** 2)

    def predict(self, X):
        """
        Realiza predicciones utilizando el modelo entrenado de Tree Boosting.

        Parametros
        ----------
        X : array-like, shape (n_samples, n_features)
            Conjunto de datos de prueba.

        Returns
        -------
        final_predictions : array-like, shape (n_samples,)
            Predicciones del modelo.
        """
        if self.is_continuous:
            # Inicializar F0 para predicciones en regresión
            F_m = np.full(X.shape[0], self.F0)
        else:
            # Usar el primer stump para inicializar F0 en clasificación
            F_m = self.first_stump.predict_class_probabilities_X(X)
        print(f"Inicialización F0 para predicción: {F_m}")

        for m, stumps_m in enumerate(self.stumps):
            if self.is_continuous:
                # Realizar predicciones para regresión
                predictions, _ = stumps_m.predict(X)
                predictions = np.array(predictions)
                F_m += self.learning_rate * self.gammas[m] * predictions
                print(f"Iteración {m + 1}: Predicciones del stump: {predictions}")
                print(f"Gamma {m + 1}: {self.gammas[m]}")
                print(f"Nuevas predicciones F_m: {F_m}")
            else:
                # Realizar predicciones para clasificación
                for k, stump in enumerate(stumps_m):
                    predictions, _ = stump.predict(X)
                    predictions = np.array(predictions)
                    F_m[:, k] += self.learning_rate * predictions
                    print(f"Iteración {m + 1}, Clase {self.classes[k]}: Predicciones del stump: {predictions}")
                    print(f"Nuevas predicciones F_m para la clase {self.classes[k]}: {F_m[:, k]}")

        if self.is_continuous:
            final_predictions = F_m
            print(f"Predicción final (antes de redondear): {final_predictions}")
            if not self.has_decimals:
                final_predictions = np.round(final_predictions).astype(int)
        else:
            final_predictions = self.classes[np.argmax(F_m, axis=1)]

        print(f"Predicción final: {final_predictions}")
        return np.array(final_predictions)
