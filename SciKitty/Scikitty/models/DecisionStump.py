# ---------------------------------------------------------------------------------
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
# ---------------------------------------------------------------------------------

import numpy as np
import pandas as pd

class Nodo:
    """
    Clase que conforma un nodo de un árbol de decisión, almacenando información como
    la regla utilizada para comparar y separar los datos de un split, la etiqueta o valor
    que más se repite en el split, la impureza de los datos, y el listado de etiquetas o
    el "y_train" con el split del nodo actual.
    """
    
    def __init__(self, es_hoja=False, regla=None, etiquetas=np.array([]), impureza=0):
        """
        Inicializa un nodo del árbol de decisión, cada nodo tiene su regla de división además
        de un atributo que indica si es un nodo final (hoja).
        """
        # Define si el nodo es una hoja o no
        self.es_hoja = es_hoja
        # Regla utilizada para dividir el conjunto de datos en este nodo
        self.regla = regla
        # Impureza del conjunto de datos en este nodo
        self.impureza = impureza
        # Etiquetas de los datos en este nodo
        self.etiquetas = etiquetas
        # Etiqueta más común en el conjunto de datos de este nodo
        self.etiqueta = self._etiqueta_mas_comun(etiquetas)
        # Número de muestras en este nodo
        self.muestras = etiquetas.size
        # Subárbol izquierdo
        self.izquierda = None
        # Subárbol derecho
        self.derecha = None

    def __str__(self):
        # Representación del nodo para impresión
        return f"Hoja: {self.etiqueta}" if self.es_hoja else f"Regla: {self.regla}"

    def _etiqueta_mas_comun(self, etiquetas):
        # Encuentra la etiqueta más común en el conjunto de datos
        valores, conteos = np.unique(etiquetas, return_counts=True)
        return valores[np.argmax(conteos)]

class DecisionStump:
    """
    Nuestra implementación de DecisionTree modificada para ser de altura 1 y poder
    predecir probabilidades de clases del target en caso de ser categórico multiclase o binario.
    """

    def __init__(self, caracteristicas, etiquetas, criterio='entropy', min_muestras_div=2, max_profundidad=1, criterio_continuo='MSE'):
        """
        El árbol recibirá una lista de características que dividirá en los nombres de dichas características y sus
        valores, además de las etiquetas correctas a predecir en estructuras de numpy.

        El árbol recibe como parámetro escrito si el criterio de impureza a utilizar es "gini" o "entropy" además de 
        hiperparámetros de regularización que el usuario definirá para controlar el modelo, como el máximo de niveles
        de profundidad del árbol y el mínimo de muestras que debe haber para hacer una nueva división de características.
        """
        # Inicializa las características y etiquetas del conjunto de datos
        self.caracteristicas = np.array(caracteristicas)
        self.etiquetas = np.array(etiquetas)
        # Almacena los nombres de las características si están disponibles
        if len(caracteristicas) > 0:
            self.nombres_caracteristicas = caracteristicas.columns.tolist() if isinstance(caracteristicas, pd.DataFrame) else [f'Característica[{i}]' for i in range(np.array(caracteristicas).shape[1])]
        else:
            self.nombres_caracteristicas = []
        # Criterio para calcular la impureza de los nodos
        self.criterio = criterio
        # Número mínimo de muestras necesarias para dividir un nodo
        self.min_muestras_div = min_muestras_div
        # Profundidad máxima del árbol
        self.max_profundidad = max_profundidad
        # Nodo raíz del árbol
        self.raiz = None
        # Etiquetas únicas en el conjunto de datos
        self.etiquetas_originales = np.unique(etiquetas)
        # Criterio para calcular la impureza de nodos con objetivos continuos
        self.criterio_continuo = criterio_continuo
        # Inicializar estructuras para atributos categóricos/binarios y continuos
        self.atributos_binarios_categoricos = []
        self.atributos_continuos = []
        # Clasificar atributos en categóricos/binarios y continuos
        self._clasificar_atributos()

    def get_tree(self, nodo=None):
        """
        Función que retorna un diccionario con los datos completos del árbol actual. Se van
        obteniendo los datos de cada nodo del árbol de forma recursiva.
        """
        # Si no se proporciona un nodo, se usa la raíz
        if nodo is None:
            nodo = self.raiz
        # Crea un diccionario que representa la estructura del árbol
        nodo_dict = {
            'es_hoja': nodo.es_hoja,
            'regla': nodo.regla,
            'etiqueta': nodo.etiqueta,
            'impureza': nodo.impureza,
            'etiquetas': nodo.etiquetas.tolist(),
            'muestras': nodo.muestras
        }
        # Si el nodo no es una hoja, agrega las ramas izquierda y derecha
        if not nodo.es_hoja:
            nodo_dict['izquierda'] = self.get_tree(nodo.izquierda)
            nodo_dict['derecha'] = self.get_tree(nodo.derecha)
        return nodo_dict

    def set_tree(self, root_node):
        """
        Función que setea la raiz del árbol en base a un nodo pasado por parametro.
        """
        # Establece la raíz del árbol a partir de un nodo dado
        self.raiz = root_node

    def is_balanced(self, umbral=0.5):
        """
        Evalúa si el dataset está balanceado basándose en un umbral de balance.
        Un dataset se considera balanceado si la proporción de la clase minoritaria respecto
        a la clase mayoritaria, del target o etiquetas a predecir, es mayor o igual al umbral
        establecido por el usuario. Devuelve true o false según el dataset esté balanceado o
        no.
        """
        # Verifica si el conjunto de datos está balanceado según un umbral dado
        _, conteos = np.unique(self.etiquetas, return_counts=True)
        if len(conteos) == 1:
            return False
        proporción_min_max = conteos.min() / conteos.max()
        return proporción_min_max >= umbral

    def fit(self):
        """
        Entrena el árbol de decisión utilizando los datos proporcionados. Llama al proceso de construir un árbol
        con las características y etiquetas.
        """
        # Construye el árbol a partir de las características y etiquetas proporcionadas
        self.raiz = self._construir_arbol(self.caracteristicas, self.etiquetas, 0)
        if self.raiz.regla:
            print(f"Regla del Stump: {self.raiz.regla}")

    def _clasificar_atributos(self):
        """
        Clasifica los atributos en continuos y categóricos/binarios al inicio del entrenamiento del árbol.
        """
        # Clasifica los atributos en categóricos/binarios y continuos
        for i, nombre in enumerate(self.nombres_caracteristicas):
            valores_unicos = np.unique(self.caracteristicas[:, i])
            if len(valores_unicos) <= 2 or isinstance(valores_unicos[0], str):
                self.atributos_binarios_categoricos.append(i)
            else:
                self.atributos_continuos.append(i)

    def _construir_arbol(self, caracteristicas, etiquetas, profundidad_actual):
        """
        Valida si se debe seguir dividiendo el conjunto de datos, en caso afirmativo, busca la mejor regla de
        división y divide el conjunto de datos en izquierda y derecha según la regla de división y llama 
        recursivamente a si mismo para construir el árbol de los nodos izquierda y derecha, teniendo cada uno
        de ellos un nuevo subconjunto de datos. En caso negativo, define el nodo como hoja y representará a una 
        etiqueta (la etiqueta más común que posea).
        """
        # Calcula la impureza del nodo actual
        nodo_impureza = self._calcular_impureza(etiquetas)
        # Verifica si se debe detener la división
        if self._detener_division(etiquetas, caracteristicas.shape[0], profundidad_actual):
            return Nodo(es_hoja=True, impureza=nodo_impureza, etiquetas=etiquetas)
        # Encuentra la mejor regla de división
        mejor_regla, _ = self._elegir_mejor_regla(caracteristicas, etiquetas)
        if not mejor_regla:
            return Nodo(es_hoja=True, impureza=nodo_impureza, etiquetas=etiquetas)
        # Divide el conjunto de datos según la mejor regla encontrada
        indices_izquierda, indices_derecha = self._dividir(caracteristicas, mejor_regla)
        # Construye el subárbol izquierdo
        subarbol_izquierdo = self._construir_arbol(
            caracteristicas[indices_izquierda], etiquetas[indices_izquierda], profundidad_actual + 1)
        # Construye el subárbol derecho
        subarbol_derecho = self._construir_arbol(
            caracteristicas[indices_derecha], etiquetas[indices_derecha], profundidad_actual + 1)
        # Crea un nodo con la mejor regla y subárboles izquierdo y derecho
        nodo = Nodo(regla=mejor_regla, impureza=nodo_impureza, etiquetas=etiquetas)
        nodo.izquierda = subarbol_izquierdo
        nodo.derecha = subarbol_derecho
        return nodo

    def _detener_division(self, etiquetas, num_muestras, profundidad_actual):
        """
        Indica si hay alguna razón para detener el split, ya sea debido a hiperparámetros o debido a que el
        conjunto ya es totalmente puro.
        """
        # Verifica si se debe detener la división del nodo
        if len(np.unique(etiquetas)) == 1 or num_muestras < self.min_muestras_div:
            return True
        if self.max_profundidad is not None and profundidad_actual >= self.max_profundidad:
            return True
        return False

    def _calcular_impureza_y_probabilidad(self, etiquetas, mascara):
        """
        Calcula la impureza y la probabilidad de una división.
        """
        # Calcula la impureza y la probabilidad de una división
        etiquetas_divididas = etiquetas[mascara]
        probabilidad = len(etiquetas_divididas) / len(etiquetas)
        impureza = self._calcular_impureza(etiquetas_divididas)
        return impureza, probabilidad

    def _calcular_impureza_division(self, etiquetas, mascara_division):
        """
        Calcula la impureza de una división.
        """
        # Calcula la impureza de una división
        impureza_valor, probabilidad_valor = self._calcular_impureza_y_probabilidad(etiquetas, mascara_division)
        impureza_no_valor, probabilidad_no_valor = self._calcular_impureza_y_probabilidad(etiquetas, ~mascara_division)
        impureza = probabilidad_valor * impureza_valor + probabilidad_no_valor * impureza_no_valor
        return impureza

    def _elegir_mejor_regla(self, caracteristicas, etiquetas):
        """
        Encuentra la regla que genera la menor impureza respecto a las etiquetas a predict.
        """
        # Encuentra la regla que genera la menor impureza
        mejor_impureza = float('inf')
        mejor_regla = None
        # Iterar sobre atributos categóricos/binarios
        for indice in self.atributos_binarios_categoricos:
            caracteristica = caracteristicas[:, indice]
            valores_unicos = np.unique(caracteristica)
            for valor in valores_unicos:
                mascara_division = caracteristica == valor
                impureza = self._calcular_impureza_division(etiquetas, mascara_division)
                if impureza < mejor_impureza:
                    mejor_impureza = impureza
                    mejor_regla = (indice, '==', valor)
        # Iterar sobre atributos continuos
        for indice in self.atributos_continuos:
            caracteristica = caracteristicas[:, indice]
            valores_unicos = np.unique(caracteristica)
            valores_ordenados = np.sort(valores_unicos)
            puntos_medios = (valores_ordenados[:-1] + valores_ordenados[1:]) / 2
            for punto in puntos_medios:
                mascara_division = caracteristica <= punto
                impureza = self._calcular_impureza_division(etiquetas, mascara_division)
                if impureza < mejor_impureza:
                    mejor_impureza = impureza
                    mejor_regla = (indice, '<=', punto)
        return mejor_regla, mejor_impureza

    def _dividir(self, caracteristicas, regla):
        """
        Divide el conjunto de datos dependiendo si cumplen la regla o no.
        """
        # Divide el conjunto de datos según la regla dada
        indice_columna, condicion, valor = regla
        if condicion == '<=':
            indices_izquierda = np.where(caracteristicas[:, indice_columna] <= valor)[0]
            indices_derecha = np.where(caracteristicas[:, indice_columna] > valor)[0]
        elif condicion == '==':
            indices_izquierda = np.where(caracteristicas[:, indice_columna] == valor)[0]
            indices_derecha = np.where(caracteristicas[:, indice_columna] != valor)[0]
        return indices_izquierda, indices_derecha

    def _calcular_impureza(self, etiquetas):
        """
        Escoge que criterio usar y devuelve la impureza calculada respecto a las etiquetas
        dependiendo del criterio escogido por el usuario en la definición del árbol de decisión
        para etiquetas multiclase o binarias, o MSE para etiquetas contínuas (target contínuo).
        """
        # Calcula la impureza del conjunto de datos en este nodo
        valores_unicos = np.unique(etiquetas)
        if etiquetas.size == 0:
            return 0
        es_binaria = len(valores_unicos) <= 2
        es_categorica = isinstance(etiquetas[0], str) and len(valores_unicos) > 2
        if not (es_binaria or es_categorica):
            if isinstance(self.criterio_continuo, str):
                if (self.criterio_continuo == 'MSE') or (self.criterio_continuo == 'MSE'):
                    return self._calcular_mse(etiquetas)
            else:
                return self.criterio_continuo(etiquetas)
        elif isinstance(self.criterio, str):
            if self.criterio == 'entropy':
                return self._calcular_entropia(etiquetas)
            elif self.criterio == 'gini':
                return self._calcular_gini(etiquetas)
        else:
            return self.criterio(etiquetas)

    def _calcular_entropia(self, etiquetas):
        """
        Devuelve la impureza utilizando las probabilidades de cada etiqueta usando el criterio entropía.
        """
        # Calcula la entropía del conjunto de datos en este nodo
        _, conteos = np.unique(etiquetas, return_counts=True)
        probabilidades = conteos / conteos.sum()
        return -np.sum(probabilidades * np.log2(probabilidades))

    def _calcular_gini(self, etiquetas):
        """
        Devuelve la impureza utilizando las probabilidades de cada etiqueta usando el criterio gini.
        """
        # Calcula el índice Gini del conjunto de datos en este nodo
        _, conteos = np.unique(etiquetas, return_counts=True)
        probabilidades = conteos / conteos.sum()
        return 1 - np.sum(probabilidades ** 2)

    def _calcular_mse(self, etiquetas):
        """
        Devuelve la impureza utilizando las probabilidades de cada etiqueta usando MSE, donde y_hat es el promedio de y.
        """
        # Calcula el error cuadrático medio (MSE) del conjunto de datos en este nodo
        if etiquetas.size == 0:
            return 0
        media_etiquetas = np.mean(etiquetas)
        return np.mean((etiquetas - media_etiquetas) ** 2)

    def predict(self, caracteristicas):
        """
        Devuelve las predicciones de cada instancia del Dataset.
        """
        # Crea un array con las características de los datos de prueba
        caracteristicas = np.array(caracteristicas)
        predicciones = []
        probabilidades = []
        # Genera predicciones para cada instancia en el conjunto de datos
        for caracteristica in caracteristicas:
            prediccion, probabilidad = self._predict_individual(caracteristica, self.raiz)
            predicciones.append(prediccion)
            probabilidades.append(probabilidad)
        return predicciones, probabilidades

    def predict_class_probabilities(self):
        """
        Devuelve las probabilidades de clase para cada instancia del Dataset.
        """
        probabilidades = []
        # Genera probabilidades de clase para cada instancia en el conjunto de datos
        for caracteristica in self.caracteristicas:
            probabilidad = self._predict_class_probabilities_individual(caracteristica, self.raiz)
            probabilidades.append(probabilidad)
        return np.array(probabilidades)

    def predict_class_probabilities_X(self, X):
        """
        Devuelve las probabilidades de clase para un conjunto de características dado.
        """
        caracteristicas = np.array(X)
        probabilidades = []
        # Genera probabilidades de clase para cada instancia en el conjunto de datos proporcionado
        for caracteristica in caracteristicas:
            probabilidad = self._predict_class_probabilities_individual(caracteristica, self.raiz)
            probabilidades.append(probabilidad)
        return np.array(probabilidades)
    
    def _predict_individual(self, caracteristica, nodo):
        """
        Determina la predicción para una instancia del dataset dependiendo si sus características cumplen
        las reglas de los nodos del árbol.
        """
        # Si el nodo es una hoja, devuelve la etiqueta como predicción
        if nodo.es_hoja:
            return nodo.etiqueta, self._calcular_probabilidad(nodo.etiquetas, nodo.etiqueta)
        # Si se cumple con la regla, recursivamente valida por el subárbol izquierdo
        if self._seguir_regla(caracteristica, nodo.regla):
            return self._predict_individual(caracteristica, nodo.izquierda)
        # Si no se cumple la regla, recursivamente valida por el subárbol derecho
        else:
            return self._predict_individual(caracteristica, nodo.derecha)

    def _predict_class_probabilities_individual(self, caracteristica, nodo):
        """
        Determina las probabilidades de clase para una instancia del dataset dependiendo si sus características
        cumplen las reglas de los nodos del árbol.
        """
        # Si el nodo es una hoja, devuelve las probabilidades de clase
        if nodo.es_hoja:
            return self._calcular_probabilidad_clase(nodo.etiquetas)
        # Si se cumple con la regla, recursivamente valida por el subárbol izquierdo
        if self._seguir_regla(caracteristica, nodo.regla):
            return self._predict_class_probabilities_individual(caracteristica, nodo.izquierda)
        # Si no se cumple la regla, recursivamente valida por el subárbol derecho
        else:
            return self._predict_class_probabilities_individual(caracteristica, nodo.derecha)

    def _calcular_probabilidad_clase(self, etiquetas):
        """
        Calcula las probabilidades de clase en un conjunto de etiquetas.
        """
        valores, conteos = np.unique(etiquetas, return_counts=True)
        probabilidades = np.zeros(len(self.etiquetas_originales))
        # Calcula las probabilidades para cada clase en el conjunto de etiquetas
        for i, valor in enumerate(self.etiquetas_originales):
            if valor in valores:
                probabilidades[i] = conteos[np.where(valores == valor)[0][0]] / etiquetas.size
        return probabilidades

    def _calcular_probabilidad(self, etiquetas, etiqueta):
        """
        Calcula la probabilidad de una etiqueta específica en un conjunto de etiquetas.
        """
        valores, conteos = np.unique(etiquetas, return_counts=True)
        indice = np.where(valores == etiqueta)[0][0]
        return conteos[indice] / conteos.sum()

    def _seguir_regla(self, caracteristica, regla):
        """
        Devuelve el booleano que indica si cumple o no la regla dependiendo si la regla es <= o ==.
        """
        # Comprueba si la característica cumple con la regla a seguir
        indice_columna, condicion, valor = regla
        if condicion == '==':
            return caracteristica[indice_columna] == valor
        elif condicion == '<=':
            return caracteristica[indice_columna] <= valor
        else:
            return caracteristica[indice_columna] > valor

    def imprimir_arbol(self, nodo=None, profundidad=0, condicion="Raíz"):
        """
        Imprime el árbol mediante prints.
        """
        # Si no se proporciona un nodo, se usa la raíz
        if nodo is None:
            nodo = self.raiz
        # Imprime el nodo y sus subárboles de manera recursiva
        if nodo.es_hoja:
            print(f"{'|   ' * profundidad}{condicion} -> Hoja: {nodo.etiqueta}")
        else:
            nombre_columna = self.nombres_caracteristicas[nodo.regla[0]]
            condicion_str = f"{nombre_columna} {nodo.regla[1]} {nodo.regla[2]}"
            print(f"{'|   ' * profundidad}{condicion} -> {condicion_str}")
            self.imprimir_arbol(nodo.izquierda, profundidad + 1, f"{condicion_str}")
            self.imprimir_arbol(nodo.derecha, profundidad + 1, f"No {condicion_str}")

    def get_tree_structure(self, nodo=None):
        """
        Usa recursión para devolver la estructura completa de un árbol, incluyendo en cada
        nodo información relevante dependiendo si es un nodo hoja o un nodo de decisión que
        representa una regla/pregunta.
        """
        if nodo is None:
            nodo = self.raiz
        # Se obtienen los valores únicos de las etiquetas con sus cantidades
        etiquetasUnicas, cuenta = np.unique(nodo.etiquetas, return_counts=True)
        # Determina si las etiquetas son continuas
        es_continua = not (len(self.etiquetas_originales) <= 2 or isinstance(self.etiquetas_originales[0], str))
        # Muestra MSE si el atributo es continuo y el criterio especificado por el usuario en la creación de DT si
        # el atributo es binario o categórico multiclase
        if es_continua:
            if isinstance(self.criterio_continuo, str):
                criterio_a_mostrar = f"MSE: {round(nodo.impureza, 3)}"
            else:
                criterio_a_mostrar = f"{self.criterio_continuo.__name__}: {round(nodo.impureza, 3)}"
        else:
            if isinstance(self.criterio, str):
                criterio_a_mostrar = f"{self.criterio}:{round(nodo.impureza, 3)}"
            else:
                criterio_a_mostrar = f"{self.criterio.__name__}:{round(nodo.impureza, 3)}"
        # Se comprueba la cantidad de valores para graficar diferente los values de cada nodo
        valor = f"[{', '.join(str(cuenta[np.where(etiquetasUnicas == etiqueta)[0][0]]) if etiqueta in etiquetasUnicas else '0' for etiqueta in self.etiquetas_originales)}]"
        # Si es una hoja retorna la siguiente información
        if nodo.es_hoja:
            # Se obtiene la impureza del nodo y se redondea a solo 3 decimales
            numeroImpureza = round(nodo.impureza, 3)
            # Se comprueba que sea mayor a "-0.0" para establecer el valor en 0 si no es el caso
            if numeroImpureza <= -0.0:
                numeroImpureza = 0
            # Guarda la información relevante del nodo
            return {
                "tipo": "Hoja",
                "criterio": criterio_a_mostrar,
                "muestras": f"muestras: {nodo.muestras}",
                "valor": f"valor: {valor}",
                "clase": f"clase: {nodo.etiqueta}"
            }
        else:
            nombre_columna = self.nombres_caracteristicas[nodo.regla[0]]
            # Devuelve la información relevante del nodo
            return {
                "tipo": "Decision",
                "reglaDescritiva": f"{nombre_columna} {nodo.regla[1]} {nodo.regla[2]}",
                "regla": f"{nodo.regla[0]} {nodo.regla[1]} {nodo.regla[2]}",
                "izquierda": self.get_tree_structure(nodo.izquierda), # Obtiene la estructura izquierda
                "derecha": self.get_tree_structure(nodo.derecha), # Obtiene la estructua derecha
                "criterio": criterio_a_mostrar,
                "muestras": f"muestras: {nodo.muestras}",
                "valor": f"valor: {valor}",
                "clase": f"clase: {nodo.etiqueta}",
            }
