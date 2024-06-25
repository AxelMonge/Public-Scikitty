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
import json
import numpy as np
import sys
sys.path.append('..')
from ..models.DecisionTree import Nodo

class TreePersistence:
    """
    Clase encargada de persistir el nodo en formato JSON para guardarlo,
    reconstruirlo o ser traducido a un programa de Prolog.
    """

    @staticmethod
    def save_tree(tree, filename):
        """
        Serializa el árbol a un archivo JSON.

        Parámetros:
        tree: Objeto del árbol de decisión (clase DecisionTree), debe tener un método get_tree() como nuestra implementación
        que retorne la estructura del árbol en formato de diccionario.
        filename: str, Nombre del archivo donde se guardará el árbol serializado.

        Funcionalidad:
        Convierte el árbol de decisión en un diccionario usando el método get_tree() 
        del objeto tree y guarda este diccionario en un archivo JSON especificado por filename.
        """
        tree_dict = tree.get_tree()

        # Convertir numpy.int64 a int
        def convert_np_int64(obj):
            if isinstance(obj, np.int64):
                return int(obj)
            raise TypeError

        with open(filename, 'w') as f:
            json.dump(tree_dict, f, indent=4, default=convert_np_int64)
    
    @staticmethod
    def load_tree(filename):
        """
        Deserializa el archivo JSON a una estructura de árbol.

        Parámetros:
        filename: str, Nombre del archivo desde donde se cargará el árbol serializado.

        Retorna:
        Nodo: Raíz del árbol de decisión reconstruido con sus hijos reconstruidos recursivamente con la misma función. (Árbol completo).

        Funcionalidad:
        Carga el contenido de un archivo JSON especificado por filename, reconstruye la 
        estructura del árbol de decisión y devuelve el nodo raíz.
        """
        with open(filename, 'r') as f:
            tree_dict = json.load(f) # Este método de la librería JSON rehace el objeto que había antes de serializarlo, en nuestro caso, siempre un dictionary.
        
        def _reconstruir_nodo(nodo_dict):
            """
            Función auxiliar para reconstruir recursivamente un nodo y sus subárboles a partir
            de un diccionario.

            Parámetros:
            nodo_dict: dict, Diccionario que representa un nodo del árbol.

            Retorna:
            Nodo: Nodo reconstruido con sus subárboles.
            """
            # Se reconstruyen los nodos recursivamente, usa la definición de la clase "Nodo" de models.DecisionTree.
            nodo = Nodo(
                es_hoja=nodo_dict['es_hoja'],
                regla=nodo_dict.get('regla'),
                impureza=nodo_dict['impureza'],
                etiquetas=np.array(nodo_dict['etiquetas'])
            )
            if not nodo.es_hoja:
                nodo.izquierda = _reconstruir_nodo(nodo_dict['izquierda'])
                nodo.derecha = _reconstruir_nodo(nodo_dict['derecha'])
            return nodo

        return _reconstruir_nodo(tree_dict)

"""
Modo de uso:

# Supongamos que ya tienes un árbol de decisión entrenado llamado 'arbol_decision'

# Para guardar el árbol en un archivo JSON
TreePersistence.save_tree(arbol_decision, 'arbol_decision.json')

# Para cargar el árbol desde un archivo JSON
raiz_cargada = TreePersistence.load_tree('arbol_decision.json')

# Asignar la raíz cargada al árbol de decisión
arbol_decision.set_tree(raiz_cargada) # Este método set_tree es único de la implementación de DT de scikitty.

# Ahora puedes usar el árbol cargado
arbol_decision.imprimir_arbol()
"""