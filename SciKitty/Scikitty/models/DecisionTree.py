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
import pandas as pd

class Nodo:
    """
        Clase que conforma un nodo de un árbol de decisión, almacenando información como
        la regla utilizada para comparar y separar los datos de un split, la etiqueta o valor
        que más se repite en el split, la impureza de los datos, y el listado de etiquetas o
        el "y_train" con el split del nodo actual. 

        Atributos
        ---------
        es_hoja: atributo que indica si el nodo es una hoja o no. Parametro que sirve para
        tratar de forma diferente los nodos en diferentes funciones de graficación.

        regla: atributo que almacena la regla utilizada para partir los datos del nodo actual.
        Se almacena una tupla en él con la regla a utilizar, por ejemplo: para la regla 
        "Numero1 <= Numero2", la tupla sería (Numero1, <=, Numero2). 

        etiquetas: atibuto que almacena los datos del split actual en el nodo. Se almacenan los
        datos que quedaron como resultado del split anterior. 

        impureza: atributo que almacena la impureza de los datos guardados en el atributo etiquetas.
        El tipo o críterio con el que se calcula este valor dependerá del críterio que el árbol de
        decisión tenga seleccionado.

        etiqueta: atributo que almacena la etiqueta más común de los datos almacenados en etiquetas.

        muestras: atibuto que almacena la cantidad de muestras con las que se va a trabajar el split
        actual. Este valor se calcula en base al atributo "etiquetas".

        izquierda: atributo que contiene el subárbol izquierdo del nodo actual.

        derecha: atributo que contiene el subárbol derecho del nodo actual.

        Ejemplos
        --------
        >>> from Scikitty.models.DecisionTree import Nodo
        >>> ...
        >>> nodo = Nodo(es_hoja=True, regla=None, impureza=nodo_impureza, etiquetas=etiquetas)
        >>> ...
    """


    # Función de construción de la clase Nodo.
    def __init__(self, es_hoja=False, regla=None, etiquetas=np.array([]), impureza=0):
        """
            Inicializa un nodo del árbol de decisión, cada nodo tiene su regla de división además
            de un atributo que indica si es un nodo final (hoja). Nuestro algoritmo está diseñado
            para que las ramas del nodo siempre sean binarias, por lo que si hay un atributo multiclase
            los hijos del nodo corresponderían a si dicho atributo presenta la subclase o no.
            Etiqueta es el nombre del target, se utilizaa solo cuando el nodo es hoja al momento de
            representarlo gráficamente.
        """

        # Atributo del nodo para saber si se trata de un nodo hoja o no.
        self.es_hoja = es_hoja

        # Atributo que almacena la regla que el nodo realiza para hacer el split.
        self.regla = regla

        # Atributo que almacena la impureza del split del nodo actual.
        self.impureza = impureza

        # Atributo que alamacena las etiquetas que componen el split.
        self.etiquetas = etiquetas

        # Atributo que almacena la etiqueta más común del split.
        self.etiqueta = self._etiqueta_mas_comun(etiquetas)

        # Atributo que representa la cantidad de muestras que hay total en el split actual.
        self.muestras = etiquetas.size

        # Atributo que almacena el subárbol izquierdo del nodo.
        self.izquierda = None

        # Atributo que almacena el subárbol derecho del nodo.
        self.derecha = None


    # Función de impresión de la clase nodo.
    def __str__(self):
        """
            Describe a un nodo dependiendo si es hoja o no. En caso de no ser hoja, se muestra la regla
            y en caso de ser hoja, se muestra la etiqueta (nombre del target).
        """
        return f"Hoja: {self.etiqueta}" if self.es_hoja else f"Regla: {self.regla}"
    

    # Función que encuentra la etiqueta más comun del atributo numpy array "etiquetas" del nodo. 
    def _etiqueta_mas_comun(self, etiquetas):
        """
            Devuelve la etiqueta más común en un conjunto de etiquetas.

            Parametros
            ----------

            etiquetas: parametro que contiene el atributo numpy array "etiquetas" del nodo. 
            Puede utilizar otro numpy array arbitrario que se le pase por parametro.

            Ejemplos
            --------
            >>> from Scikitty.models.DecisionTree import Nodo
            >>> ...
            >>> nodo = Nodo(es_hoja=True, regla=None, impureza=nodo_impureza, etiquetas=etiquetas)
            >>> nodo._etiqueta_mas_comun(etiquetas=etiquetas)
        """

        if len(etiquetas) == 0:
            return None  # Si no hay etiquetas, retornar None o un valor indicativo

        # Se crea un array que contiene la cantidad de conteos de cada etiqueta.
        valores, conteos = np.unique(etiquetas, return_counts=True)

        # Se obtiene el valor máximo del array.
        return valores[np.argmax(conteos)]


class DecisionTree:
    """
        Definición del algoritmo de aprendizaje automático "Árbol de Decisión". La idea es construir un árbol donde 
        los nodos son preguntas o reglas sobre alguna característica del conjunto de datos (DS), dichas reglas,
        dividirán al DS en subconjuntos más pequeños según las preguntas o reglas que mejor dividan al DS.

        Nuestro árbol funciona escogiendo las características que generen los subconjuntos con menor impureza respecto a
        las etiquetas a predict, utilizando criterios como "gini" o "entropy" si los datos son multiclase o binarios
        y MSE si los datos son contínuos y requieren de técnicas de regresión.

        Atributos
        ---------
        caracteristicas: atributo del árbol que almacena el "x_train" del modelo. Utilizado en diferentes funciones
        para calcular parametros como la impureza de los datos o la mejor pregunta que se puede realizar en un split.

        etiquetas: atributo del árbol que almacena el "y_train" del modelo. Al igual que el atributo "caracteristicas"
        es utilizado en diferentes funciones para calcular parametros como la impureza o mejor pregunta.

        criterio: atributo que indica cual es le críterio que utilizará el árbol para calcular la impureza de los datos
        de un split dado.

        min_muestras_div: hiperparametro del árbol que indica el minimo de muestras por split en cada nodo del árbol.

        max_profundidad: hiperparametro del árbol que indica el máximo de profundidad al cual el árbol se puede generar.

        raiz: atributo que almacena el nodo raiz del árbol.

        etiquetas_originales: atributo que almacena las etiquetas unicas del "y_train" del árbol. Utilizado para comprobar
        si se trata de un split izquierdo o derecho a la hora de graficar los "valores" de cada nodo.

        criterio_continuo: atributo que indica cuál es el criterio que se utilizará para calcular la impureza de los datos
        de un split dado cuando los datos son continuos.

        Ejemplos
        --------
        >>> from Scikitty.models.DecisionTree import DecisionTree
        >>> from Scikitty.model_selection.train_test_split import train_test_split
        >>> import pandas as pd
        
        >>> # Se almacena el nombre del archivo donde se guarda el dataset.
        >>> file_name = 'fictional_disease'

        >>> # Se cargan los datos.
        >>> data = pd.read_csv(f'../datasets/{file_name}.csv')

        >>> # Se preparan los datos.
        >>> features = data.drop('Disease', axis=1)
        >>> labels = data['Disease']

        >>> # Se dividen los datos.
        >>> X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

        >>> # Se crea e instancia el árbol de decisión.
        >>> dt = DecisionTree(X_train, y_train, criterio='entropy', min_muestras_div=2, max_profundidad=5)
    """


    # Función de construcción de la clase DecisionTree.
    def __init__(self, caracteristicas, etiquetas, criterio='entropy', min_muestras_div=2, max_profundidad=None, criterio_continuo='MSE'):
        """
            El árbol recibirá una lista de características que dividirá en los nombres de dichas características y sus
            valores, además de las etiquetas correctas a predict en estructuras de numpy.

            El árbol recibe como parámetro escrito si el criterio de impureza a utilizar es "gini" o "entropy" además de 
            hiperparámetros de regularización que el usuario definirá para controlar el modelo, como el máximo de niveles
            de profundidad del árbol y el mínimo de muestras que debe haber para hacer una nueva división de características.

            Se inicializa un nodo raíz que, en el proceso de fit al modelo, definirá su regla de división y nodos hijo.
        """

        # Atributo Numpy array con las caracteristicas o x_train del árbol.
        self.caracteristicas = np.array(caracteristicas)

        # Atributo Numpy array con las etiquetas o y_train del árbol.
        self.etiquetas = np.array(etiquetas)

        # Se comprueba si el numpy array de carcteristicas tiene un tamaño superior a 0.
        if len(caracteristicas) > 0:
            # Se asignan los nombres de las caracteristicas.
            self.nombres_caracteristicas = caracteristicas.columns.tolist() if isinstance(caracteristicas,
            pd.DataFrame) else [f'Característica[{i}]' for i in range(np.array(caracteristicas).shape[1])]
            # Si no se cumple, se dejan los nombre de las caracteristicas como un array vacío.
        else: self.nombres_caracteristicas = []

        # Atributo que almacena el criterio a utilizar para calcular la impureza de los splits.
        self.criterio = criterio
        
        # Atributo que almacena el hiperparametro de mínimo de muestras por split.
        self.min_muestras_div = min_muestras_div

        # Atributo que almacena el hiperparametro del máximo de profundidad del árbol.
        self.max_profundidad = max_profundidad

        # Atributo que almacena la raiz del árbol.
        self.raiz = None

        # Atributo que almacena las etiquetas originales o las etiquetas unicas del y_train.
        self.etiquetas_originales = np.unique(etiquetas)

        # Atributo que almacena el criterio continuo.
        self.criterio_continuo = criterio_continuo

        # Inicializar estructuras para atributos categóricos/binarios y continuos
        self.atributos_binarios_categoricos = []
        self.atributos_continuos = []

        # Clasificar atributos en categoricos/binarios-continuos al inicio del árbol
        self._clasificar_atributos()


    def get_tree(self, nodo=None):
        """
            Función que retorna un diccionario con los datos completos del árbol actual. Se van
            obteniendo los datos de cada nodo del árbol de forma recursiva.

            Parametros
            ---------- 

            nodo: se obtiene un nodo con toda su información. Se extrae de dicho nodo, todos sus
            datos para almacenarlos en un diccionario que creará toda la estructura del árbol.

            Ejemplos
            -------- 
            >>> from Scikitty.models.DecisionTree import DecisionTree
            >>> from Scikitty.model_selection.train_test_split import train_test_split
            >>> import pandas as pd

            >>> # Se almacena el nombre del archivo donde se guarda el dataset.
            >>> file_name = 'CO2_car_emision'

            >>> # Se cargan los datos.
            >>> data = pd.read_csv(f'../datasets/{file_name}.csv')

            >>> # Se preparan los datos.
            >>> features = data.drop('CO2', axis=1)
            >>> labels = data['CO2']

            >>> # Se dividen los datos.
            >>> X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

            >>> # Se crea e instancia el árbol de decisión.
            >>> dt = DecisionTree(X_train, y_train, criterio='Entropy', min_muestras_div=2, max_profundidad=5)
            >>> dt.fit()

            >>> # Se visualiza el árbol.
            >>> tree_structure = dt.get_tree_structure()
        """

        # Se comprueba si el nodo es igual para None. Dado el caso se iguala a la raiz del árbol.
        if nodo is None:
            nodo = self.raiz

        # Se crea un diccionario con todos lo datos del nodo.
        nodo_dict = {
            'es_hoja': nodo.es_hoja,
            'regla': nodo.regla,
            'etiqueta': nodo.etiqueta,
            'impureza': nodo.impureza,
            'etiquetas': nodo.etiquetas.tolist(),
            'muestras': nodo.muestras
        }

        # Se comprueba si el nodo no es de tipo hoja para añadirle al diccionario sus subárboles.
        if not nodo.es_hoja:
            nodo_dict['izquierda'] = self.get_tree(nodo.izquierda)
            nodo_dict['derecha'] = self.get_tree(nodo.derecha)
        
        return nodo_dict


    def set_tree(self, root_node):
        """
            Función que setea la raiz del árbol en base a un nodo pasado por parametro.

            Parametros
            ----------

            root_node: nodo raiz pasado por parametro el cual se quiere establecer como la raiz
            del árbol de decisión actual.

            Ejemplos
            -------- 

            >>> from Scikitty.models.DecisionTree import DecisionTree
            >>> from Scikitty.model_selection.train_test_split import train_test_split
            >>> import pandas as pd

            >>> # Se almacena el nombre del archivo donde se guarda el dataset
            >>> file_name = 'playTennis'

            >>> # Se cargan los datos.
            >>> data = pd.read_csv(f'../datasets/{file_name}.csv')

            >>> # Se preparan los datos.
            >>> features = data.drop('Play Tennis', axis=1)  # Asume que 'Play Tennis' es la columna objetivo
            >>> labels = data['Play Tennis']

            >>> # Se dividen los datos.
            >>> X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

            >>> # Se crea e instancia el árbol de decisión.
            >>> dt = DecisionTree(X_train, y_train, criterio='gini', min_muestras_div=2, max_profundidad=5)
            >>> dt.fit()

            >>> # Se guarda el árbol en un archivo JSON.
            >>> TreePersistence.save_tree(dt, 'playTennis.json')

            >>> # Se carga el árbol desde el archivo JSON.
            >>> nueva_raiz = TreePersistence.load_tree('playTennis.json')
            >>> nuevo_dt = DecisionTree(X_train, y_train, criterio='gini', min_muestras_div=2, max_profundidad=5)
            >>> nuevo_dt.set_tree(nueva_raiz)
        """
        self.raiz = root_node
        

    def is_balanced(self, umbral=0.5):
        """
            Evalúa si el dataset está balanceado basándose en un umbral de balance.
            Un dataset se considera balanceado si la proporción de la clase minoritaria respecto
            a la clase mayoritaria, del target o etiquetas a predecir, es mayor o igual al umbral
            establecido por el usuario. Devuelve true o false según el dataset esté balanceado o
            no.

            Parametros
            ----------
            umbral: parametro que indica el umbral de balance entre los datos del dataset.

            Ejemplos
            --------
            
        """

        # Se obtiene la cantidad total de cada elemento diferente del y_train.
        _, conteos = np.unique(self.etiquetas, return_counts=True)

        # Si solo hay un solo tipo de elemento, el dataset no está balanceado.
        if len(conteos) == 1:
            return False
        
        # Se calcula la proporción entre el mínimo y el máximo.
        proporción_min_max = conteos.min() / conteos.max()

        # Se retorna un True si la proporción es mayor o igual al umbral planteado o False en caso contrario.
        return proporción_min_max >= umbral
    

    def fit(self):
        """
            Entrena el árbol de decisión utilizando los datos proporcionados. Llama al proceso de construir un árbol
            con las características y etiquetas.

            Parametros
            ----------
            Función sin parametros. Utiliza los atributos propios del árbol. 
            
            Ejemplos
            --------
            >>> from Scikitty.models.DecisionTree import DecisionTree
            >>> from Scikitty.model_selection.train_test_split import train_test_split
            >>> import pandas as pd

            >>> # Se almacena el nombre del archivo donde se guarda el dataset
            >>> file_name = 'fictional_reading_place'

            >>> # Se cargan los datos.
            >>> data = pd.read_csv(f'../datasets/{file_name}.csv')

            >>> # Se preparan los datos.
            >>> features = data.drop('user_action', axis=1)  # Características del dataset
            >>> labels = data['user_action']  # Etiquetas del dataset

            >>> # Se dividen los datos en conjuntos de entrenamiento y prueba.
            >>> X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

            >>> # Se crea e instancia el árbol de decisión.
            >>> dt = DecisionTree(X_train, y_train, criterio='entropy', min_muestras_div=2, max_profundidad=5)
            >>> dt.fit()  # Entrenar el árbol de decisión
        """

        # Se realizá el llamado a la función encargada de construir el árbol.
        self.raiz = self._construir_arbol(self.caracteristicas, self.etiquetas, 0)


    def _clasificar_atributos(self):
        """
            Clasifica los atributos en continuos y categóricos/binarios al inicio del entrenamiento del árbol.

            Parametros
            ----------
            Función sin parametros. Utiliza los atributos propios del árbol. 
            
            Ejemplos
            --------
            >>> ...
            >>> # Clasificar Atributos es una función interna de DecisionTree. No debe ser utilizada fuera de la clase.
            >>> self._clasificar_atributos()
            >>> ...
        """
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

            Parametros
            ----------
            caracteristicas: parametro que contiene un numpy array con las caracteristicas del split actual.
            Es el restante de X_train del split anterior utilizado para calcular la impureza, mejor pregunta
            y crear el nodo con la información relevante.

            etiquetas: parametro que contiene un numpy array con las etiquetas del split actual.
            Es el restante de Y_train del split anterior utilizado para calcular la impureza, mejor pregunta
            y crear el nodo con la información relevante.

            profundidad_actual: parametro que almacena la profundad actual del árbol. Parametro importante
            para saber si detener la generación del árbol o seguir.

            Ejemplos
            --------
            >>> ...
            >>> # Construir Árbol es una función interna de DecisionTree. No debe ser utilizada fuera de la clase.
            >>> self.raiz = self._construir_arbol(self.caracteristicas, self.etiquetas, 0)
            >>> ...
        """

        # Se obtiene la impureza del split actual.
        nodo_impureza = self._calcular_impureza(etiquetas)

        # Caso base para la recursividad.
        # Valida los hiperparámetros: profundidad y la cantidad de la muestra.
        if self._detener_division(etiquetas, caracteristicas.shape[0], profundidad_actual):
            return Nodo(es_hoja=True, impureza=nodo_impureza, etiquetas=etiquetas)
        
        # Se obtiene la característica con la mejor pureza.
        mejor_regla, _ = self._elegir_mejor_regla(caracteristicas, etiquetas)

        # Se comprueba si no exite la mejor regla para poder crear el nodo si es el caso.
        if not mejor_regla:
            return Nodo(es_hoja=True, impureza=nodo_impureza, etiquetas=etiquetas)
        
        # Se inicializan dos variables que almacenaran los indicies de cada lado del split.
        indices_izquierda = indices_derecha = 0

        # Se comprueba si existe la mejor regla para obtener los indices de cada lado del split.
        if mejor_regla:
            # Se generan las divisiones recursivamente
            indices_izquierda, indices_derecha = self._dividir(caracteristicas, mejor_regla)
        
        # Se genera el subárbol izquierdo del nodo actual.
        subarbol_izquierdo = self._construir_arbol(
            caracteristicas[indices_izquierda], etiquetas[indices_izquierda], profundidad_actual + 1)
        
        # Se genera el subárbol derecho del nodo actual.
        subarbol_derecho = self._construir_arbol(
            caracteristicas[indices_derecha], etiquetas[indices_derecha], profundidad_actual + 1)
        
        # Al nodo raíz se le asigna la mejor característica según su impureza.
        nodo = Nodo(regla=mejor_regla, impureza=nodo_impureza, etiquetas=etiquetas)

        # Se le agrega el subárbol izquierdo.
        nodo.izquierda = subarbol_izquierdo

        # Se le agrega el subárbol derecho.
        nodo.derecha = subarbol_derecho

        return nodo


    def _detener_division(self, etiquetas, num_muestras, profundidad_actual):
        """
            Indica si hay alguna razón para detener el split, ya sea debido a hiperparámetros o debido a que el
            conjunto ya es totalmente puro.

            Parametros
            ----------

            etiquetas: parametro que almacena las etiquetas con las cuales se realizára el split. Necesario
            para saber si ya no existen más de un solo tipo de datos en el array de etiquetas.

            num_muestras: parametro que almacena el numero de muestras que las etiquetas almacenan. Utilizando
            este parametro se comprueba si el numero de muestras es menor al hiperametro "min_muestras_div".

            profundidad_actual: parametro que almacena la profundidad del árbol actual para saber si detener
            la creación del arbol en base al hiperparametro "max_profundidad".

            Ejemplos
            --------
            >>> ...
            >>> # Detener Division es una función interna de DecisionTree. No debe ser utilizada fuera de la clase.
            >>> if self._detener_division(etiquetas, caracteristicas.shape[0], profundidad_actual):
            >>> ...
        """

        # Si sólo hay una etiqueta o que el número de muestras sean menores al hiperparámetro.
        if len(np.unique(etiquetas)) == 1 or num_muestras < self.min_muestras_div:
            return True
        
        # Verifica que la profundidad actual sea mayor o igual a la máxima profundidad.
        if self.max_profundidad is not None and profundidad_actual >= self.max_profundidad:
            return True
        
        return False


    def _calcular_impureza_y_probabilidad(self, etiquetas, mascara):
        """
            Calcula la impureza y la probabilidad de una división dado un conjunto de etiquetas y una máscara.

            Parameters
            ----------
            etiquetas : numpy array
                Etiquetas del conjunto de datos.
            mascara : numpy array
                Máscara booleana para dividir el conjunto de etiquetas.

            Returns
            -------
            impureza : float
                Impureza de la división.
            probabilidad : float
                Probabilidad de la división.
        """
        etiquetas_divididas = etiquetas[mascara]
        probabilidad = len(etiquetas_divididas) / len(etiquetas)
        impureza = self._calcular_impureza(etiquetas_divididas)
        return impureza, probabilidad

    def _calcular_impureza_division(self, etiquetas, mascara_division):
        """
            Calcula la impureza de una división dado un conjunto de etiquetas, una característica y una máscara de división.

            Parameters
            ----------
            etiquetas : numpy array
                Etiquetas del conjunto de datos.
            caracteristica : numpy array
                Característica del conjunto de datos.
            mascara_division : numpy array
                Máscara booleana para dividir el conjunto de datos.

            Returns
            -------
            impureza : float
                Impureza de la división.
        """
        impureza_valor, probabilidad_valor = self._calcular_impureza_y_probabilidad(
            etiquetas, mascara_division)
        impureza_no_valor, probabilidad_no_valor = self._calcular_impureza_y_probabilidad(
            etiquetas, ~mascara_division)
        impureza = probabilidad_valor * impureza_valor + \
            probabilidad_no_valor * impureza_no_valor
        return impureza

    def _elegir_mejor_regla(self, caracteristicas, etiquetas):
        """
            Encuentra la regla que genera la menor impureza respecto a las etiquetas a predict.

            Parameters
            ----------
            caracteristicas : numpy array
                Características del conjunto de datos.
            etiquetas : numpy array
                Etiquetas del conjunto de datos.

            Returns
            -------
            mejor_regla : tuple
                La mejor regla de división encontrada.
            mejor_impureza : float
                La menor impureza encontrada.
        """
        mejor_impureza = float('inf')
        mejor_regla = None

        # Iterar sobre atributos categóricos/binarios
        for indice in self.atributos_binarios_categoricos:
            caracteristica = caracteristicas[:, indice]
            valores_unicos = np.unique(caracteristica)
            for valor in valores_unicos:
                mascara_division = caracteristica == valor
                impureza = self._calcular_impureza_division(
                    etiquetas, mascara_division)
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
                impureza = self._calcular_impureza_division(
                    etiquetas, mascara_division)
                if impureza < mejor_impureza:
                    mejor_impureza = impureza
                    mejor_regla = (indice, '<=', punto)

        return mejor_regla, mejor_impureza


    def _dividir(self, caracteristicas, regla):
        """
            Divide el conjunto de datos dependiendo si cumplen la regla o no.

            Parametros
            ----------
            caracteristicas: parametro que contiene un numpy array con las caracteristicas del split actual.
            Es el restante de X_train del split anterior utilizado realizar un nuevo split.

            regla: parametro que contine la información de la regla o condición a cumplir para separar los
            datos.

            Ejemplos
            --------
            >>> ...
            >>> # Dividir es una función interna de DecisionTree. No debe ser utilizada fuera de la clase.
            >>> indices_izquierda, indices_derecha = self._dividir(caracteristicas, mejor_regla)
            >>> ...
        """

        # Se obtiene el indice de la columna, la condición y el valor a comparar de la regla.
        indice_columna, condicion, valor = regla

        # Encuentra los índices que cumplen la regla (izquierda) y las que no (derecha).
        if(condicion == '<='):
            indices_izquierda = np.where(
                caracteristicas[:, indice_columna] <= valor)[0]
            indices_derecha = np.where(
                caracteristicas[:, indice_columna] > valor)[0]
        elif(condicion == '=='):
            indices_izquierda = np.where(
                caracteristicas[:, indice_columna] == valor)[0]
            indices_derecha = np.where(
                caracteristicas[:, indice_columna] != valor)[0]
        return indices_izquierda, indices_derecha


    def _calcular_impureza(self, etiquetas):
        """
            Escoge que criterio usar y devuelve la impureza calculada respecto a las etiquetas
            dependiendo del criterio escogido por el usuario en la definición del árbol de decisión
            para etiquetas multiclase o binarias, o MSE para etiquetas contínuas (target contínuo).

            Parametros
            ----------
            etiquetas: parametro que contiene un numpy array con las etiquetas del split actual.
            Es el restante de Y_train del split anterior utilizado para calcular su impureza.

            Ejemplos
            --------
            >>> ...
            >>> # Calcular Impureza es una función interna de DecisionTree. No debe ser utilizada fuera de la clase.
            >>> impureza_valor = self._calcular_impureza(etiquetas_divididas)
            >>> ...
        """

        # Se optienen los valores unicos de las etiquetas.
        valores_unicos = np.unique(etiquetas)

        # Se comprueba si la cantidad de etiquetas es igual a cero para retornar 0 dado el caso.
        if etiquetas.size == 0:
            return 0
        
        # Se determina si las etiquetas son continuas (no categoricas y no binarias).
        es_binaria = len(valores_unicos) <= 2
        es_categorica = isinstance(etiquetas[0], str) and len(valores_unicos) > 2

        # Dependiento del "es_binaria" o "es_categorica" y del críterio del árbol
        # se calcula la impureza con un críterio diferente.
        if not (es_binaria or es_categorica):
            if isinstance(self.criterio_continuo, str):
                if self.criterio_continuo == 'MSE':
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

            Parametros
            ----------
            etiquetas: parametro que contiene un numpy array con las etiquetas del split actual.
            Es el restante de Y_train del split anterior utilizado para calcular su entropía.

            Ejemplos
            --------
            >>> ...
            >>> # Calcular Entropia es una función interna de DecisionTree. No debe ser utilizada fuera de la clase.
            >>> impureza_valor = self._calcular_entropia(etiquetas)
            >>> ...
        """

        # Se obtienen las cantidades totales de los valores unicos de las etiquetas.
        _, conteos = np.unique(etiquetas, return_counts=True)

        # Se calculan las probabilidades de cada etiqueta.
        probabilidades = conteos / conteos.sum()

        # Se calcula la entropia de las etiquetas.
        return -np.sum(probabilidades * np.log2(probabilidades))

    def _calcular_gini(self, etiquetas):
        """
            Devuelve la impureza utilizando las probabilidades de cada etiqueta usando el criterio gini.

            Parametros
            ----------
            etiquetas: parametro que contiene un numpy array con las etiquetas del split actual.
            Es el restante de Y_train del split anterior utilizado para calcular su indice gini.

            Ejemplos
            --------
            >>> ...
            >>> # Calcular Gini es una función interna de DecisionTree. No debe ser utilizada fuera de la clase.
            >>> impureza_valor = self._calcular_gini(etiquetas)
            >>> ...
        """

        # Se obtienen las cantidades totales de los valores unicos de las etiquetas.
        _, conteos = np.unique(etiquetas, return_counts=True)

        # Se calculan las probabilidades de cada etiqueta.
        probabilidades = conteos / conteos.sum()

        # Se calcula el indice gini de las etiquetas.
        return 1 - np.sum(probabilidades ** 2)


    def _calcular_mse(self, etiquetas):
        """
            Devuelve la impureza utilizando las probabilidades de cada etiqueta usando MSE, donde y_hat es el promedio de y.
        
            Parametros
            ----------
            etiquetas: parametro que contiene un numpy array con las etiquetas del split actual.
            Es el restante de Y_train del split anterior utilizado para calcular su MSE.

            Ejemplos
            --------
            >>> ...
            >>> # Calcular MSE es una función interna de DecisionTree. No debe ser utilizada fuera de la clase.
            >>> impureza_valor = self._calcular_mse(etiquetas)
            >>> ...
        """

        # Se comprueba que la cantidad de etiquetas sea igual a cero para retornar 0 dado el caso.
        if etiquetas.size == 0:
            return 0
        
        # Se calcula la media de las etiquetas.
        media_etiquetas = np.mean(etiquetas)

        # Se calcula el mse de las etiquetas.
        return np.mean((etiquetas - media_etiquetas) ** 2)


    def predict(self, caracteristicas):
        """
            Devuelve las predicciones de cada instancia del Dataset.

            Parametros
            ----------
            caracteristicas: parametro que contiene un numpy array con las caracteristicas del split actual.
            Es el  X_Test del modelo utilizado para predecir los posibles datos.

            Ejemplos
            --------
            >>> from Scikitty.models.DecisionTree import DecisionTree
            >>> from Scikitty.model_selection.train_test_split import train_test_split
            >>> import pandas as pd

            >>> # Se almacena el nombre del archivo donde se guarda el dataset.
            >>> file_name = 'fictional_reading_place'

            >>> # Se cargan los datos.
            >>> data = pd.read_csv(f'../datasets/{file_name}.csv')

            >>> # Se preparan los datos.
            >>> features = data.drop('user_action', axis=1)  # Características del dataset
            >>> labels = data['user_action']  # Etiquetas del dataset

            >>> # Se dividen los datos en conjuntos de entrenamiento y prueba.
            >>> X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

            >>> # Se crea e instanciar el árbol de decisión.
            >>> dt = DecisionTree(X_train, y_train, criterio='entropy', min_muestras_div=2, max_profundidad=5)
            >>> dt.fit()  # Entrenar el árbol de decisión

            >>> # Se imprimen los resultados de evaluación del modelo.
            >>> y_pred = dt.predict(X_test)
        """

        # Se crea un numpu array utilizando las caracteristicas.
        caracteristicas = np.array(caracteristicas)

        # Se calculan las predicciones del modelos en base al x_train.
        return [self._predict_individual(caracteristica, self.raiz) for caracteristica in caracteristicas]


    def _predict_individual(self, caracteristica, nodo):
        """
            Determina la predicción para una instancia del dataset dependiendo si sus características cumplen
            las reglas de los nodos del árbol.

            Parametros
            ----------
            caracteristicas: parametro que contiene un numpy array con las caracteristicas del split actual.
            Es el X_Test del modelo utilizado para predecir los posibles datos.

            nodo: parametro que contiene el nodo raiz del árbol utilizado para llamar de forma recursiva
            está función para realizar las predicciones en cada uno de los nodos y recopilar los datos.

            Ejemplos
            --------
            >>> ...
            >>> # Predict Individual es una función interna de DecisionTree. No debe ser utilizada fuera de la clase.
            >>> self._predict_individual(caracteristica, self.raiz)
            >>> ...
        """

        # Si es una hoja devuelve la etiqueta como predicción
        if nodo.es_hoja: 
            return nodo.etiqueta
        # Si se cumple con la regla, recursivamente valide por al subárbol izquierdo
        if self._seguir_regla(caracteristica, nodo.regla):
            return self._predict_individual(caracteristica, nodo.izquierda)
        # Si no se cumple la regla, recursivamente valide por el subárbol derecho
        else:
            return self._predict_individual(caracteristica, nodo.derecha)


    def _seguir_regla(self, caracteristica, regla):
        """
            Devuelve el booleano que indica si cumple o no la regla dependiendo si la regla es <= o ==.
       
            Parametros
            ----------
            caracteristicas: parametro que contiene un numpy array con las caracteristicas del split actual.

            regla: parametro que contiene la regla a evaluar en la función para saber si las caracteristicas
            cumplen o no con ella.

            Ejemplos
            --------
            >>> ...
            >>> # Seguir Regla es una función interna de DecisionTree. No debe ser utilizada fuera de la clase.
            >>> if self._seguir_regla(caracteristica, nodo.regla):
            >>> ...
        """

        # Se obtiene el indice de la columna, la condición y el valor del elemento a comparar en base a la regla.
        indice_columna, condicion, valor = regla

        # Comprueba si la característica cumple con la regla a seguir.
        if condicion == '==':
            return caracteristica[indice_columna] == valor
        elif condicion == '<=':
            return caracteristica[indice_columna] <= valor
        else:
            return caracteristica[indice_columna] > valor


    def imprimir_arbol(self, nodo=None, profundidad=0, condicion="Raíz"):
        """
            Imprime el árbol mediante prints.

            Parametros
            ----------
            nodo: se obtiene el nodo raiz del árbol para recorrer todos los nodos del árbol. Esto
            ayuda a imprimir cada nodo de forma separada.

            profundida: se obtiene la profundiad actual a la cual está recorriendo la función. Esto
            ayuda a imprimir y llevar el control de la profundidad actual del árbol y de los nodos.

            condicion: se obtien la condición que se utilizo en el nodo actual del árbol para poder
            imprimirlo como parte de los datos mostrados.

            Ejemplos
            --------
            >>> from Scikitty.models.DecisionTree import DecisionTree
            >>> from Scikitty.model_selection.train_test_split import train_test_split
            >>> import pandas as pd

            >>> # Se almacena el nombre del archivo donde se guarda el dataset.
            >>> file_name = 'fictional_reading_place'

            >>> # Se cargan los datos.
            >>> data = pd.read_csv(f'../datasets/{file_name}.csv')

            >>> # Se preparan los datos.
            >>> features = data.drop('user_action', axis=1)  # Características del dataset
            >>> labels = data['user_action']  # Etiquetas del dataset

            >>> # Se dividen los datos en conjuntos de entrenamiento y prueba.
            >>> X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

            >>> # Se crea e instanciar el árbol de decisión.
            >>> dt = DecisionTree(X_train, y_train, criterio='entropy', min_muestras_div=2, max_profundidad=5)
            >>> dt.fit()  # Entrenar el árbol de decisión

            >>> # Se imprime el árbol de decisión.
            >>> dt.imprimir_arbol()
        """

        # Se comprueba si el nodo es igual a None para igualar el nodo a la raiz.
        if nodo is None:
            nodo = self.raiz

        # Se comprueba si el nodo es una hoja para imprimir el fortmato de la hoja
        # en caso contrario se imprime en el formato del nodo.
        if nodo.es_hoja:
            print(f"{'|   ' * profundidad}{condicion} -> Hoja: {nodo.etiqueta}")
        else:
            # Toma el nombre de la característica según la regla actual
            nombre_columna = self.nombres_caracteristicas[nodo.regla[0]]
            condicion_str = f"{nombre_columna} {nodo.regla[1]} {nodo.regla[2]}"
            print(f"{'|   ' * profundidad}{condicion} -> {condicion_str}")
            # Llama recursivamente a la función con respectoal subárbol izquierdo y derecho
            self.imprimir_arbol(
                nodo.izquierda, profundidad + 1, f"{condicion_str}")
            self.imprimir_arbol(
                nodo.derecha, profundidad + 1, f"No {condicion_str}")


    def get_tree_structure(self, nodo=None):
        """
            Usa recursión para devolver la estructura completa de un árbol, incluyendo en cada
            nodo información relevante dependiendo si es un nodo hoja o un nodo de decisión que
            representa una regla/pregunta.

            Parametros
            ----------
            nodo: atributo que obtiene la raiz del árbol para poder obtener toda la estructura
            de dicho árbol en base a este atributo.

            Ejemplos
            --------
            >>> from Scikitty.models.DecisionTree import DecisionTree
            >>> from Scikitty.model_selection.train_test_split import train_test_split
            >>> import pandas as pd

            >>> # Se almacena el nombre del archivo donde se guarda el dataset
            >>> file_name = 'playTennis'

            >>> # Se cargan los datos.
            >>> data = pd.read_csv(f'../datasets/{file_name}.csv')

            >>> # Se preparan los datos.
            >>> features = data.drop('Play Tennis', axis=1)  # Asume que 'Play Tennis' es la columna objetivo
            >>> labels = data['Play Tennis']

            >>> # Se dividen los datos.
            >>> X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

            >>> # Se crea e instancia el árbol de decisión.
            >>> dt = DecisionTree(X_train, y_train, criterio='entropy', min_muestras_div=2, max_profundidad=5)
            >>> dt.fit()

            >>> # Se visualiza el árbol de decisión.
            >>> tree_structure = dt.get_tree_structure()
        """
        if nodo is None:
            nodo = self.raiz
            
        # Se obtienen los valores unicos de las etiquetas con sus cantidades.
        etiquetasUnicas, cuenta = np.unique(nodo.etiquetas, return_counts=True)
        
        # Determina si las etiquetas son continuas.
        es_continua = not (len(self.etiquetas_originales) <= 2 or isinstance(self.etiquetas_originales[0], str))

        # Muestra MSE si el atributo es continuo y el criterio especificado por el usuario en la creación de DT si
        # el atributo es binario o categórico multiclase.
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
                
        # Se comprueba la cantidad de valores para graficar diferente los values de cada nodo.
        valor = f"[{', '.join(str(cuenta[np.where(etiquetasUnicas == etiqueta)[0][0]]) if etiqueta in etiquetasUnicas else '0' for etiqueta in self.etiquetas_originales)}]"
       
        # Si es una hoja retorna la siguiente información.
        if nodo.es_hoja:

            # Se obtiene la impureza del nodo y se redondea a solo 3 decimales.
            numeroImpureza = round(nodo.impureza, 3)

            # Se comprueba que sea mayor a "-0.0" para establecer el valor en 0 si no es el caso.
            if numeroImpureza <= -0.0:
                numeroImpureza = 0
            # Guarda la información relevante del nodo.
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

