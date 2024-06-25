% --------------------------------------------------------------------------------- %
/*
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
*/
% --------------------------------------------------------------------------------- %

% Descripción del módulo:
% Este módulo 'utils' proporciona diversas funciones de utilidad para la manipulación de listas y otros
% auxiliares necesarios para el algoritmo A*. Estas funciones son utilizadas en varios contextos para
% gestionar listas, generar identificadores únicos, y manipular estructuras de datos en el tablero.

% Definición del módulo utils, que expone las funciones públicas.
:- module(utils, [
    zip/3,
    enumerate/2,
    index_of/3,
    new_id/2,
    list_split/4,
    list_set_value/4,
    list_at/3,
    list_swap/4,
    swap_vertical/5
]).

%%%%%%%%%%%%%%%%%%%% UTILS %%%%%%%%%%%%%

% Une dos listas en una lista de pares.
zip(List1, List2, ZippedList) :-
    % Utiliza maplist para combinar elementos de List1 y List2 en pares.
    maplist([X, Y, [X, Y]] >> true, List1, List2, ZippedList).

% Enumera una lista añadiendo índices a sus elementos.
enumerate(List, EnumeratedList) :-
    % Calcula la longitud de la lista.
    length(List, Length),
    % Genera una lista de índices del 1 al Length.
    numlist(1, Length, Indices),
    % Combina los índices y los elementos de la lista.
    zip(Indices, List, EnumeratedList).

% Encuentra el índice de un valor en una lista.
index_of(Value, List, Index) :-
    % Utiliza nth1 para encontrar el índice del valor.
    nth1(Index, List, Value).

% Genera un nuevo identificador único basado en un prefijo.
new_id(Base, Id) :-
    % Utiliza gensym para generar un identificador único.
    gensym(Base, Id).

% Divide una lista en un prefijo, un elemento y un sufijo.
list_split(List, Prefix, Element, Suffix) :-
    % Utiliza append para dividir la lista en Prefijo, Elemento y Sufijo.
    append(Prefix, [Element | Suffix], List).

% Establece un valor en una lista en una posición específica. Usa el list_split
list_set_value(List, Index, Value, ResultList) :-
    % Divide la lista en Prefijo y Sufijo.
    list_split(List, Prefix, _, Suffix),
    % Calcula el índice ajustado para 0-based indexing.
    Index1 is Index - 1, Index1 >= 0,
    % Verifica que la longitud del Prefijo sea correcta.
    length(Prefix, Index1),
    % Combina Prefijo, Valor y Sufijo en la lista resultado.
    append(Prefix, [Value | Suffix], ResultList).

% Obtiene el valor en una lista en una posición específica.
list_at(List, Index, Value) :-
    % Utiliza nth1 para obtener el valor en la posición Index.
    nth1(Index, List, Value).

% Intercambia dos elementos en una lista.
list_swap(List, Index1, Index2, SwappedList):-
    % Obtiene el valor en Index1.
    list_at(List, Index1, Value1),
    % Obtiene el valor en Index2.
    list_at(List, Index2, Value2),
    % Establece el valor en Index2 en Index1.
    list_set_value(List, Index1, Value2, TempList),
    % Establece el valor en Index1 en Index2.
    list_set_value(TempList, Index2, Value1, SwappedList).

% Intercambia elementos verticalmente entre dos filas en un tablero. Toma en cuenta que las rows son hechos en prolog con assert, entonces debe buscar según id entre distintas rows.
swap_vertical(BoardId, Row1, Col1, Row2, Col2) :-
    % Obtiene la fila Row1 del tablero.
    board_row(BoardId, Row1, RowData1),
    % Obtiene la fila Row2 del tablero.
    board_row(BoardId, Row2, RowData2),
    % Obtiene el valor en Col1 de Row1.
    list_at(RowData1, Col1, Value1),
    % Obtiene el valor en Col2 de Row2.
    list_at(RowData2, Col2, Value2),
    % Establece el valor Value2 en Col1 de Row1.
    list_set_value(RowData1, Col1, Value2, UpdatedRow1),
    % Establece el valor Value1 en Col2 de Row2.
    list_set_value(RowData2, Col2, Value1, UpdatedRow2),
    % Actualiza la fila Row1 en el tablero.
    update_board_row(BoardId, Row1, UpdatedRow1),
    % Actualiza la fila Row2 en el tablero.
    update_board_row(BoardId, Row2, UpdatedRow2).
