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
% Este módulo implementa el algoritmo de búsqueda A* con dos tipos de heurísticas:
% la distancia Euclidiana y la distancia Manhattan. Está diseñado para resolver el
% problema del 8-puzzle no trivial, buscando desde un estado inicial hasta la posición final deseada.
% El algoritmo A* es una técnica de búsqueda informada que utiliza una función heurística para estimar
% el costo total (f = g + h) desde el nodo actual hasta el objetivo. La heurística h puede ser 
% la distancia Manhattan o la Euclidiana en este contexto.

% Definición del módulo heuristic_astar, que expone las funciones públicas.
:- module(heuristic_astar, [
    manhattan_distance/3,
    euclidean_distance/3,
    heuristic/2,
    current_position/3
]).

% Importa el módulo 'utiles' y selecciona las funciones específicas para usarlas.
% Este módulo proporciona funciones de utilidad para manipulación de listas y otros auxiliares.
:- use_module(utiles, [
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

% Importa el módulo 'guess' y selecciona las funciones específicas para usarlas.
% Este módulo incluye funciones para gestionar el estado del tablero, validar movimientos y realizar operaciones necesarias para el algoritmo A*.
:- use_module(guess, [
    get_goal_board/1,
    clear_visited_boards/0,
    clear_goal_found/0,
    clear_goal_board/0,
    save_goal_board/1,
    set_board_as_visited/1,
    is_board_visited/1,
    save_total_colum/1,
    save_total_row/1,
    get_total_colum/1,
    get_total_row/1,
    board_row_size/3,
    board_total_rows/2,
    save_goal_colums/1,
    save_goal_rows/1,
    get_goal_colums/1,
    get_goal_rows/1,
    goal_row_size/3,
    goal_total_rows/2,
    get_goal_row/3,
    create_board_from_list/2,
    create_goal_board_from_list/2,
    convert_board_to_list/2,
    store_goal_positions/0,
    store_row_positions/2,
    get_all_goal_rows/2,
    generate_new_board_id/1,
    clear_all_boards/0,
    clear_goal/0,
    clear_goal_details/1,
    clear_board/1,
    clone_board/2,
    add_row_to_board/2,
    update_board_row/3,
    add_empty_to_board/3,
    add_row_to_goal/2,
    add_empty_to_goal/3,
    display_board/1,
    display_goal/1,
    get_valid_move/3,
    valid_move/4,
    apply_move_to_board/2,
    apply_horizontal_move/2,
    apply_vertical_move/2,
    generate_child_board/3,
    goal_reached/1,
    compare_all_rows/2,
    compare_rows/2,
    start_astar/1,
    save_last_state/1,
    get_last_state/1,
    astar/1,
    goal_search/1,
    goal_achieved/3,
    process_board/5,
    expand_children/6,
    display_search_state/5,
    display_and_store_board/2,
    build_matrix/2,
    ejemplo/1,
    test_astar/0,
    goal_position/2,
    board_row/3
]).

%%%%%%%%%%%%%%%%%%%% HEURISTIC; MANHATTAN AND EUCLIDEAN DISTANCE %%%%%%%%%%%%%

% Calcula la distancia Manhattan entre dos puntos (X1, Y1) y (X2, Y2).
manhattan_distance([X1, Y1], [X2, Y2], Distance) :-
    % Calcula la distancia Manhattan
    Distance is abs(X1 - X2) + abs(Y1 - Y2).

% Calcula la distancia Euclidiana entre dos puntos (X1, Y1) y (X2, Y2).
euclidean_distance([X1, Y1], [X2, Y2], Distance) :-
    % Calcula la distancia Euclidiana
    Distance is sqrt((X1 - X2) * (X1 - X2) + (Y1 - Y2) * (Y1 - Y2)).

% Calcula el valor heurístico basado en las distancias Manhattan y Euclidiana para un tablero dado.
heuristic(BoardId, HeuristicValue) :-
    % Encuentra todas las distancias Manhattan entre las posiciones actuales y las posiciones objetivo
    findall(Distance, (
        goal_position(Piece, GoalPos),
        current_position(BoardId, Piece, CurrPos),
        manhattan_distance(GoalPos, CurrPos, Distance)
    ), ManhattanDistances),
    % Suma todas las distancias Manhattan
    sumlist(ManhattanDistances, ManhattanHeuristicValue),
    
    % Encuentra todas las distancias Euclidianas entre las posiciones actuales y las posiciones objetivo
    findall(Distance, (
        goal_position(Piece, GoalPos),
        current_position(BoardId, Piece, CurrPos),
        euclidean_distance(GoalPos, CurrPos, Distance)
    ), EuclideanDistances),
    % Suma todas las distancias Euclidianas
    sumlist(EuclideanDistances, EuclideanHeuristicValue),

    % Combina los valores heurísticos Manhattan y Euclidiana
    HeuristicValue = [ManhattanHeuristicValue, EuclideanHeuristicValue].

% Obtiene la posición actual de una pieza en un tablero dado.
current_position(BoardId, Piece, [RowIndex, ColIndex]) :-
    % Encuentra la fila que contiene la pieza
    board_row(BoardId, RowIndex, Row),
    % Encuentra el índice de la pieza en la fila
    nth1(ColIndex, Row, Piece).
