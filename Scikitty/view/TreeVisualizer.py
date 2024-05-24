# --------------------------------------------------------------------------------- #
"""
    Autores:
    1) Nombre: John Rojas Chinchilla
       ID: 118870938
       Correo: john.rojas.chinchilla@est.una.ac.cr
       Horario: 1pm

    2) Nombre: Abigail Salas
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
import graphviz


class TreeVisualizer:
    """
    Clase para visualizar un árbol de decisión utilizando Graphviz.

    Atributos
    ---------
    grafo: atributo que crea un grafo utilizando la libreria de graphviz en el formato png.

    Ejemplos
    --------
    >>> from Scikitty.models.DecisionTree import DecisionTree
    >>> from Scikitty.view.TreeVisualizer import TreeVisualizer
    >>> from Scikitty.model_selection.train_test_split import train_test_split
    >>> import pandas as pd

    >>> # Se almacena el nombre del archivo donde se guarda el dataset
    >>> file_name = 'CO2_car_emision'

    >>> # Se cargan los datos.
    >>> data = pd.read_csv(f'../datasets/{file_name}.csv')

    >>> # Se preparan los datos.
    >>> features = data.drop('CO2', axis=1)
    >>> labels = data['CO2']

    >>> # Se dividen los datos.
    >>> X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    >>> # Se crea e instancia el árbol de decisión.
    >>> dt = DecisionTree(X_train, y_train, criterio='entropy', min_muestras_div=2, max_profundidad=5)
    >>> dt.fit()

    >>> # Se visualiza el árbol utilziando la clase TreeVisualizer
    >>> tree_structure = dt.get_tree_structure()
    >>> visualizer = TreeVisualizer()
    >>> visualizer.graph_tree(tree_structure)
    >>> visualizer.get_graph(f'{file_name}_tree', ver=True)
    """


    def __init__(self):
        """
        Inicializa un objeto Graphviz Digraph para visualizar el árbol de decisión.
        """
        self.grafo = graphviz.Digraph(format='png')


    def graph_tree(self, estructura_arbol, padre=None, etiqueta_arista='', nivel=0, posicion=0):
        """
        Método recursivo para graficar el árbol de decisión a partir de su estructura utilizando Graphviz.
        Dibuja el nodo actual y llama recursivamente a sí mismo para dibujar los nodos hijos.

        Parametros
        ----------
        estructura_arbol: dict, la estructura del árbol de decisión definiendo cada nodo y los atributos a mostrar por nodo.

        padre: str, el nodo padre en el grafo.

        etiqueta_arista: str, la etiqueta de la arista que conecta al nodo padre con el nodo actual. (Se usa true y false predeterminadamente).

        nivel: int, el nivel del nodo en el árbol.

        posicion: int, la posición del nodo en el nivel.

        Ejemplos
        -------- 
        >>> from Scikitty.models.DecisionTree import DecisionTree
        >>> from Scikitty.view.TreeVisualizer import TreeVisualizer
        >>> from Scikitty.model_selection.train_test_split import train_test_split
        >>> import pandas as pd

        >>> # Se almacena el nombre del archivo donde se guarda el dataset
        >>> file_name = 'CO2_car_emision'

        >>> # Se cargan los datos.
        >>> data = pd.read_csv(f'../datasets/{file_name}.csv')

        >>> # Se preparan los datos.
        >>> features = data.drop('CO2', axis=1)
        >>> labels = data['CO2']

        >>> # Se dividen los datos.
        >>> X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

        >>> # Se crea e instancia el árbol de decisión.
        >>> dt = DecisionTree(X_train, y_train, criterio='entropy', min_muestras_div=2, max_profundidad=5)
        >>> dt.fit()

        >>> # Se visualiza el árbol utilziando la clase TreeVisualizer
        >>> tree_structure = dt.get_tree_structure()
        >>> visualizer = TreeVisualizer()
        >>> visualizer.graph_tree(tree_structure)
        >>> visualizer.get_graph(f'{file_name}_tree', ver=True)
        """

        # Dibuja un nodo usando informacion relevante de estructura_arbol según si es hoja o regla.
        if estructura_arbol['tipo'] == 'Hoja':
            nombre_nodo = f"hoja_{id(estructura_arbol)}"
            color = '#ffa500'
        else:
            nombre_nodo = f"decision_{id(estructura_arbol)}"
            color = '#5a9ad5'
            self.graph_tree(estructura_arbol['izquierda'],
                            padre=nombre_nodo,
                            etiqueta_arista='True',
                            nivel=nivel+1,
                            posicion=posicion-1)
            self.graph_tree(estructura_arbol['derecha'],
                            padre=nombre_nodo,
                            etiqueta_arista='False',
                            nivel=nivel+1,
                            posicion=posicion+1)

        # LLama al método que formatea la informació a desplegar en el nodo.
        etiqueta_nodo = self._formato_etiqueta(estructura_arbol)
        self.grafo.node(nombre_nodo, label=etiqueta_nodo,
                        shape='box', style='filled', fillcolor=color)
        if padre:
            self.grafo.edge(padre, nombre_nodo, label=etiqueta_arista)


    def _formato_etiqueta(self, nodo):
        """
        Crea una etiqueta formateada con los detalles del nodo centrados y en líneas separadas.
        Divide el texto de 'regla' en líneas individuales y las muestra como etiquetas del nodo.

        Parametros
        ----------
        nodo: dict, La información del nodo a formatear.

        Ejemplos
        --------
        >>> ...
        >>> # Formato Etiqueta es una función interna de DecisionTree. No debe ser utilizada fuera de la clase.
        >>> etiqueta_nodo = self.formato_etiqueta(estructura_arbol)
        >>> ...
        """

        # Formatea la información a desplegar en el nodo según si es hoja o, si no es hoja, es regla.
        tipo = nodo.get('tipo', "")
        if tipo == 'Hoja':
            criterio = nodo.get('criterio', "")
            muestras = nodo.get('muestras', "")
            valor = nodo.get('valor', "")
            clase = nodo.get('clase', "")
            etiqueta_nodo = f"{criterio}\n{muestras}\n{valor}\n{clase}"
        else:
            regla = nodo.get('reglaDescritiva', "")
            criterio = nodo.get('criterio', "")
            muestras = nodo.get('muestras', "")
            valor = nodo.get('valor', "")
            clase = nodo.get('clase', "")
            etiqueta_nodo = f"{regla}\n{criterio}\n{muestras}\n{valor}\n{clase}"
        return etiqueta_nodo

    def get_graph(self, nombre_archivo='arbol', ver=True):
        """
        Renderiza el grafo como un archivo de imagen y muestra el grafo.

        Parametros
        ----------
        nombre_archivo: str, El nombre del archivo de imagen a crear.

        ver: bool, Si es True, abre el archivo de imagen después de crearlo.

        Ejemplos
        --------
        >>> from Scikitty.models.DecisionTree import DecisionTree
        >>> from Scikitty.view.TreeVisualizer import TreeVisualizer
        >>> from Scikitty.model_selection.train_test_split import train_test_split
        >>> import pandas as pd

        >>> # Se almacena el nombre del archivo donde se guarda el dataset
        >>> file_name = 'CO2_car_emision'

        >>> # Se cargan los datos.
        >>> data = pd.read_csv(f'../datasets/{file_name}.csv')

        >>> # Se preparan los datos.
        >>> features = data.drop('CO2', axis=1)
        >>> labels = data['CO2']

        >>> # Se dividen los datos.
        >>> X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

        >>> # Se crea e instancia el árbol de decisión.
        >>> dt = DecisionTree(X_train, y_train, criterio='entropy', min_muestras_div=2, max_profundidad=5)
        >>> dt.fit()

        >>> # Se visualiza el árbol utilziando la clase TreeVisualizer
        >>> tree_structure = dt.get_tree_structure()
        >>> visualizer = TreeVisualizer()
        >>> visualizer.graph_tree(tree_structure)
        >>> visualizer.get_graph(f'{file_name}_tree', ver=True)
        """
        self.grafo.render(
            nombre_archivo, view=ver)  # Método de Digraph que se utiliza para generar (o "renderizar") y guardar el grafo creado en un archivo de salida
        # Usamos ver como true por defecto para abrir el archivo, eso lo mandeja Digraph.


"""
Modo de uso:

# Obtener la estructura del árbol
tree_structure = dt.get_tree_structure()

# Crear un visualizador del árbol
visualizer = TreeVisualizer()

# Graficar el árbol
visualizer.graph_tree(tree_structure)

# Renderizar y mostrar el grafo del árbol
visualizer.get_graph(f'{file_name}_tree', ver=True)
"""
