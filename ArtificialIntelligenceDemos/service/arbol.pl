
:- use_module(library(http/json)).
:- use_module(library(http/json_convert)).

% Definición del nodo del árbol de decisión
:- json_object nodo(
    es_hoja:bool,
    regla:list,
    etiqueta:string,
    impureza:float,
    etiquetas:list,
    muestras:int,
    izquierda:dict,
    derecha:dict
).

% Leer el archivo JSON
leer_json(File, JSON) :-
    open(File, read, Stream),
    json_read_dict(Stream, JSON),
    close(Stream).

% Reconstruir el árbol de decisión
reconstruir_nodo(Dict, Nodo) :-
    _{es_hoja: EsHoja, regla: Regla, etiqueta: Etiqueta, impureza: Impureza, etiquetas: Etiquetas, muestras: Muestras} :< Dict,
    (   EsHoja = true
    ->  Nodo = nodo(EsHoja, Regla, Etiqueta, Impureza, Etiquetas, Muestras, _, _)
    ;   _{izquierda: IzquierdaDict, derecha: DerechaDict} :< Dict,
        reconstruir_nodo(IzquierdaDict, Izquierda),
        reconstruir_nodo(DerechaDict, Derecha),
        Nodo = nodo(EsHoja, Regla, Etiqueta, Impureza, Etiquetas, Muestras, Izquierda, Derecha)
    ).

% Cargar el árbol desde un archivo JSON
cargar_arbol(File, Arbol) :-
    leer_json(File, JSON),
    reconstruir_nodo(JSON, Arbol).
	
% Función para evaluar una regla
evaluar_regla([Atributo, Operador, Valor], Datos) :-
    nth0(Atributo, Datos, Dato),
    (   Operador == "==" -> Dato == Valor
    ;   Operador == "!=" -> Dato \== Valor
    ;   Operador == "<"  -> Dato < Valor
    ;   Operador == ">"  -> Dato > Valor
    ;   Operador == "=<" -> Dato =< Valor
    ;   Operador == ">=" -> Dato >= Valor
    ).

% Función para hacer preguntas al árbol de decisión
preguntar_arbol(nodo(true, _, Etiqueta, _, _, _, _, _), _, Etiqueta). % Si es una hoja, devolver la etiqueta
preguntar_arbol(nodo(false, Regla, _, _, _, _, Izquierda, Derecha), Datos, Etiqueta) :-
    (   evaluar_regla(Regla, Datos)
    ->  preguntar_arbol(Izquierda, Datos, Etiqueta) % Seguir a la izquierda
    ;   preguntar_arbol(Derecha, Datos, Etiqueta)  % Seguir a la derecha
    ).
	
init(File, Pregunta, Respuesta) :-
	cargar_arbol(File, Arbol),
	preguntar_arbol(Arbol, Pregunta, Respuesta)
.
