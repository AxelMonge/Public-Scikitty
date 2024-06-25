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
import sklearn.model_selection

def train_test_split(X, y, test_size=0.2, random_state=None):
    """
        Divide los datos en conjuntos de entrenamiento y prueba para un modelo de árbol de decisión.
        Usamos la implementación de SKLearn para este propósito.

        Parámetros:
        X: Conjunto de datos con las características (features) que se usarán para entrenar el modelo.
        y: Conjunto de datos con las etiquetas (targets) correspondientes a cada muestra en X.
        (Ambos X, y, usando los datasets especificados para sci-kitty, pueden tener sus column names). 
        test_size: float, int or None, optional (default=0.2)
            Proporción del conjunto de datos que se incluirá en el conjunto de prueba:
            - Si float, debería estar entre 0.0 y 1.0 y representa la fracción del conjunto de datos a utilizar como prueba.
            - Si int, representa el número absoluto de muestras de prueba.
            - Si None, el valor predeterminado es 0.25.
            Nosotros usamos siempre 0.2 (lo recomendado) en los scripts.
        random_state: int or None, optional (default=None)
            Controla la aleatoriedad del muestreo. Usar un int específico para reproducir los mismos resultados en futuras ejecuciones (666 para los
            increíbles, 42 para gente normal como nosotros).

        Retorna:
        X_train: Conjunto de características (filas que representan instancias/ejemplos) para el entrenamiento.
        X_test: Conjunto de características (filas que representan instancias/ejemplos) para la prueba.
        y_train: Conjunto de etiquetas (columna donde cada fila es la etiqueta correcta de una instancia) para el entrenamiento.
        y_test: Conjunto de etiquetas (columna donde cada fila es la etiqueta correcta de una instancia) para la prueba.
    """
    return sklearn.model_selection.train_test_split(X, y, test_size=test_size, random_state=random_state)
