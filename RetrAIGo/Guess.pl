% --------------------------------------------------------------------------------- %
/*
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
*/
% --------------------------------------------------------------------------------- %

% Descripción del Módulo:
%
% Este módulo, `guess`, está diseñado para implementar y gestionar el algoritmo de búsqueda A* con heurísticas 
% de distancia Manhattan y Euclidiana para resolver un problema de rompecabezas 8-puzzle. El módulo contiene 
% una variedad de predicados que facilitan la creación, manipulación y evaluación de tableros de juego, así 
% como la ejecución del algoritmo A*. 
%
% A continuación se describen las principales funcionalidades del módulo:
%
% 1. Gestión de Tableros y Objetivos:
%    - `create_board_from_list/2` y `create_goal_board_from_list/2`: Estos predicados crean tableros de juego 
%      y tableros objetivo a partir de listas, respectivamente.
%    - `display_board/1` y `display_goal/1`: Permiten visualizar el estado actual de un tablero de juego y un 
%      tablero objetivo.
%    - `clone_board/2`, `clear_board/1`, y `clear_all_boards/0`: Facilitan la clonación y limpieza de tableros.
%    - `save_goal_board/1` y `get_goal_board/1`: Guardan y recuperan el tablero objetivo actual.
%
% 2. Movimientos y Validaciones:
%    - `get_valid_move/3` y `valid_move/4`: Determinan los movimientos válidos desde una posición vacía en un 
%      tablero.
%    - `apply_move_to_board/2`, `apply_horizontal_move/2`, y `apply_vertical_move/2`: Aplican movimientos 
%      específicos (horizontales o verticales) a un tablero.
%    - `generate_child_board/3`: Genera un nuevo tablero hijo basado en un movimiento válido.
%
% 3. Heurísticas y Evaluaciones:
%    - `heuristic/2`: Calcula las heurísticas de distancia Manhattan y Euclidiana para un tablero dado.
%    - `goal_reached/1`, `compare_all_rows/2`, y `compare_rows/2`: Determinan si se ha alcanzado el objetivo 
%      comparando filas de tableros.
%
% 4. Algoritmo A*:
%    - `start_astar/1` y `astar/1`: Inician y ejecutan el algoritmo A* utilizando una cola de prioridad.
%    - `process_board/5` y `expand_children/6`: Procesan tableros y expanden nodos hijos durante la búsqueda 
%      del algoritmo A*.
%    - `goal_search/1` y `goal_achieved/3`: Gestionan la búsqueda del objetivo y las acciones a realizar una 
%      vez alcanzado el objetivo.
%    - `display_search_state/5`: Muestra el estado actual de la búsqueda, incluyendo el costo del movimiento y 
%      los valores heurísticos.
%
% 5. Utilidades y Funciones Auxiliares:
%    - `check_solvable/1` y `count_inversions/2`: Verifican si un tablero es solucionable contando las 
%      inversiones.
%    - `display_and_store_board/2` y `build_matrix/2`: Muestran y almacenan el estado de un tablero en una 
%      matriz.
%    - `test_astar/0`: Predicado de prueba que ejecuta el algoritmo A* con un tablero inicial y un tablero 
%      objetivo predeterminados.
%
% Este módulo se basa en las funcionalidades proporcionadas por otros módulos auxiliares (`utiles` y 
% `heuristic_astar`) para realizar diversas tareas como la manipulación de listas, la generación de identificadores 
% únicos y el cálculo de heurísticas. En conjunto, estas herramientas permiten una implementación eficiente y 
% modular del algoritmo A* para resolver el problema del 8-puzzle.

% Definición del módulo y los predicados exportados
:- module(guess, [
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
    board_row/3,
    goal_position/2
]).

% Importar módulos necesarios
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
:- use_module(heuristic_astar, [
    manhattan_distance/3,
    euclidean_distance/3,
    heuristic/2,
    current_position/3
]).

% Declarar hechos dinámicos
:- dynamic visited_board/1.
:- dynamic goal_found/0.
:- dynamic goal_board/1.

% Obtener el tablero objetivo
get_goal_board(X) :-
    goal_board(X).

% Limpiar los tableros visitados
clear_visited_boards :-
    retractall(visited_board(_)).

% Limpiar el estado de objetivo encontrado
clear_goal_found :-
    retractall(goal_found).

% Limpiar el tablero objetivo
clear_goal_board :-
    retractall(goal_board()).

% Guardar el tablero objetivo
save_goal_board(GoalBoard):-
    retractall(goal_board()),
    assert(goal_board(GoalBoard)).

% Marcar un tablero como visitado
set_board_as_visited(BoardId) :-
    convert_board_to_list(BoardId, List),
    term_hash(List, HashId),
    assert(visited_board(HashId)).

% Verificar si un tablero ha sido visitado
is_board_visited(BoardId) :-
    convert_board_to_list(BoardId, List),
    term_hash(List, HashId),
    visited_board(HashId).

% Declarar dinámicos para columnas y filas
:- dynamic(total_colum/1).
:- dynamic(total_row/1).

:- dynamic(goal_colums/1).
:- dynamic(goal_rows/1).

% Guardar el número total de columnas
save_total_colum(X) :-
    retractall(total_colum(_)),
    assert(total_colum(X)).

% Guardar el número total de filas
save_total_row(X) :-
    retractall(total_row(_)),
    assert(total_row(X)).

% Obtener el número total de columnas
get_total_colum(X) :-
    total_colum(X).

% Obtener el número total de filas
get_total_row(X) :-
    total_row(X).

% Obtener el tamaño de una fila del tablero
board_row_size(BoardId, RowIndex, Size) :-
    board_row(BoardId, RowIndex, Row),
    length(Row, Size),
    save_total_colum(Size).

% Obtener el número total de filas del tablero
board_total_rows(BoardId, TotalRows) :-
    findall(RowIndex, board_row(BoardId, RowIndex, _), Rows),
    length(Rows, TotalRows),
    save_total_row(TotalRows).

% Guardar las columnas del objetivo
save_goal_colums(X) :-
    retractall(goal_colums(_)),
    assert(goal_colums(X)).

% Guardar las filas del objetivo
save_goal_rows(X) :-
    retractall(goal_rows(_)),
    assert(goal_rows(X)).

% Obtener las columnas del objetivo
get_goal_colums(X) :-
    goal_colums(X).

% Obtener las filas del objetivo
get_goal_rows(X) :-
    goal_rows(X).

% Obtener el tamaño de una fila del objetivo
goal_row_size(BoardId, RowIndex, Size) :-
    goal_row(BoardId, RowIndex, Row),
    length(Row, Size),
    save_goal_colums(Size).

% Obtener el número total de filas del objetivo
goal_total_rows(BoardId, TotalRows) :-
    findall(RowIndex, goal_row(BoardId, RowIndex, _), Rows),
    length(Rows, TotalRows),
    save_goal_rows(TotalRows).

% Obtener una fila del objetivo
get_goal_row(BoardId, RowIndex, Row) :-
    goal_row(BoardId, RowIndex, Row).

% Crear un tablero a partir de una lista
create_board_from_list(List, BoardId) :-
    generate_new_board_id(BoardId),
    clear_board(BoardId),
    enumerate(List, EnumeratedList),
    forall(member(Row, EnumeratedList), add_row_to_board(BoardId, Row)),
    board_total_rows(BoardId, _),
    board_row_size(BoardId, 1, _).

% Crear un tablero objetivo a partir de una lista
create_goal_board_from_list(List, 'GOAL_1') :-
    clear_goal_details('GOAL_1'),
    enumerate(List, EnumeratedList),
    forall(member(Row, EnumeratedList), add_row_to_goal('GOAL_1', Row)),
    goal_total_rows('GOAL_1', _),
    goal_row_size('GOAL_1', 1, _),
    save_goal_board('GOAL_1'),
    store_goal_positions.

% Convertir un tablero a una lista
convert_board_to_list(BoardId, List) :-
    findall(Row, (board_row(BoardId, _, Row)), List).

% Declarar dinámicos para filas y vacíos del tablero y objetivo
:- dynamic board_row/3.
:- dynamic board_empty/3.

:- dynamic goal_row/3.
:- dynamic goal_empty/3.

:- dynamic goal_position/2.

% Guardar las posiciones del objetivo
store_goal_positions :-
    get_goal_board(Goal),
    get_all_goal_rows(Goal, TotalRows),
    forall(between(1, TotalRows, RowIndex),
           (get_goal_row(Goal, RowIndex, Row),
            store_row_positions(Row, RowIndex))).

% Guardar las posiciones de una fila del objetivo
store_row_positions(Row, RowIndex) :-
    length(Row, TotalColumns),
    forall(between(1, TotalColumns, ColIndex),
           (nth1(ColIndex, Row, Value),
            assert(goal_position(Value, [RowIndex, ColIndex])))).

% Obtener todas las filas del objetivo
get_all_goal_rows(Goal, TotalRows) :-
    findall(RowIndex, get_goal_row(Goal, RowIndex, _), RowIndexes),
    length(RowIndexes, TotalRows).

% Generar un nuevo ID para un tablero
generate_new_board_id(BoardId) :- new_id('board_', BoardId).

% Limpiar todos los tableros
clear_all_boards :-
    retractall(board_row(_, _, _)),
    retractall(board_empty(_, _, _)).

% Limpiar el objetivo
clear_goal :-
    retractall(goal_row(_, _, _)),
    retractall(goal_empty(_, _, _)).

% Limpiar detalles de un objetivo específico
clear_goal_details(BoardId) :-
    retractall(goal_row(BoardId, _, _)),
    retractall(goal_empty(BoardId, _, _)).

% Limpiar un tablero específico
clear_board(BoardId) :-
    retractall(board_row(BoardId, _, _)),
    retractall(board_empty(BoardId, _, _)).

% Clonar un tablero
clone_board(BoardId, ClonedBoardId) :-
    findall([RowId, Row], board_row(BoardId, RowId, Row), EnumeratedList),
    generate_new_board_id(ClonedBoardId),
    forall(member(RowClone, EnumeratedList), add_row_to_board(ClonedBoardId, RowClone)),
    board_empty(BoardId, EmptyRow, EmptyCol),
    add_empty_to_board(ClonedBoardId, EmptyRow, EmptyCol).

% Agregar una fila a un tablero
add_row_to_board(BoardId, [RowIndex, Row]) :-
    assert(board_row(BoardId, RowIndex, Row)),
    ( index_of(empty, Row, EmptyIndex) -> add_empty_to_board(BoardId, RowIndex, EmptyIndex) ; true ).

% Actualizar una fila de un tablero
update_board_row(BoardId, RowIndex, UpdatedRow) :-
    retract(board_row(BoardId, RowIndex, _)),
    add_row_to_board(BoardId, [RowIndex, UpdatedRow]).

% Agregar un vacío a un tablero
add_empty_to_board(BoardId, RowIndex, ColIndex) :-
    retractall(board_empty(BoardId, _, _)),
    assert(board_empty(BoardId, RowIndex, ColIndex)).

% Agregar una fila a un objetivo
add_row_to_goal(BoardId, [RowIndex, Row]) :-
    assert(goal_row(BoardId, RowIndex, Row)),
    ( index_of(empty, Row, EmptyIndex) -> add_empty_to_goal(BoardId, RowIndex, EmptyIndex) ; true ).

% Agregar un vacío a un objetivo
add_empty_to_goal(BoardId, RowIndex, ColIndex) :-
    retractall(goal_empty(BoardId, _, _)),
    assert(goal_empty(BoardId, RowIndex, ColIndex)).

% Mostrar un tablero
display_board(BoardId) :-
    writeln('Board rows:'),
    findall([RowIndex, Row], board_row(BoardId, RowIndex, Row), Rows),
    sort(Rows, SortedRows),
    forall(member([Index, Row], SortedRows), writeln([Index, Row])),
    writeln('Empty at:'),
    board_empty(BoardId, EmptyRow, EmptyCol),
    write([EmptyRow, EmptyCol]), nl.

% Mostrar un objetivo
display_goal(BoardId) :-
    writeln('Board rows:'),
    findall([RowIndex, Row], goal_row(BoardId, RowIndex, Row), Rows),
    sort(Rows, SortedRows),
    forall(member([Index, Row], SortedRows), writeln([Index, Row])),
    writeln('Empty at:'),
    goal_empty(BoardId, EmptyRow, EmptyCol),
    write([EmptyRow, EmptyCol]), nl.

% Obtener un movimiento válido
get_valid_move(BoardId, Position, Direction) :-
    board_empty(BoardId, EmptyRow, EmptyCol),
    valid_move(EmptyRow, EmptyCol, Position, Direction).

% Verificar si un movimiento es válido
valid_move(EmptyRow, EmptyCol, [RowAbove, EmptyCol], up) :- EmptyRow > 1, RowAbove is EmptyRow - 1.
valid_move(EmptyRow, EmptyCol, [RowBelow, EmptyCol], down) :-
    get_total_colum(TotalColumns),
    EmptyRow < TotalColumns,
    RowBelow is EmptyRow + 1.
valid_move(EmptyRow, EmptyCol, [EmptyRow, ColLeft], left) :- EmptyCol > 1, ColLeft is EmptyCol - 1.
valid_move(EmptyRow, EmptyCol, [EmptyRow, ColRight], right) :-
    get_total_row(TotalRows),
    EmptyCol < TotalRows,
    ColRight is EmptyCol + 1.

% Aplicar un movimiento a un tablero
apply_move_to_board(BoardId, Direction) :-
    apply_horizontal_move(BoardId, Direction);
    apply_vertical_move(BoardId, Direction).

% Aplicar un movimiento horizontal
apply_horizontal_move(BoardId, Direction) :-
    (Direction = left; Direction = right),
    get_valid_move(BoardId, [Row, NewCol], Direction),
    board_row(BoardId, Row, RowData),
    board_empty(BoardId, Row, EmptyCol),
    list_swap(RowData, EmptyCol, NewCol, UpdatedRow),
    update_board_row(BoardId, Row, UpdatedRow),
    add_empty_to_board(BoardId, Row, NewCol).

% Aplicar un movimiento vertical
apply_vertical_move(BoardId, Direction) :-
    (Direction = up; Direction = down),
    get_valid_move(BoardId, [NewRow, Col], Direction),
    board_empty(BoardId, EmptyRow, Col),
    swap_vertical(BoardId, EmptyRow, Col, NewRow, Col),
    add_empty_to_board(BoardId, NewRow, Col).

% Generar un tablero hijo
generate_child_board(BoardId, ChildBoardId, Direction) :-
    member(Direction, [left, right, up, down]),
    clone_board(BoardId, ChildBoardId),
    apply_move_to_board(ChildBoardId, Direction).

% Verificar si se ha alcanzado el objetivo
goal_reached(BoardId) :-
    compare_all_rows(BoardId, 1).

% Comparar todas las filas de un tablero
compare_all_rows(BoardId, RowIndex) :-
    get_goal_rows(TotalRows),
    compare_rows(BoardId, RowIndex),
    (RowIndex < TotalRows ->
        NextIndex is RowIndex + 1,
        compare_all_rows(BoardId, NextIndex)
    ; true).

% Comparar filas individuales
compare_rows(BoardId, RowIndex) :-
    get_goal_board(Goal),
    get_goal_row(Goal, RowIndex, GoalRow),
    board_row(BoardId, RowIndex, BoardRow),
    GoalRow = BoardRow.

% Iniciar el algoritmo A*
start_astar(BoardId) :-
    clear_visited_boards,
    clear_goal_found,
    writeln(BoardId),
    heuristic(BoardId, HValue),
    empty_heap(OpenHeap),
    add_to_heap(OpenHeap, HValue, [BoardId, 1, HValue, [], 1], UpdatedOpenHeap),
    astar(UpdatedOpenHeap).

% Declarar el último estado dinámico
:- dynamic last_state/1.

% Guardar el último estado
save_last_state(X) :-
    retractall(last_state(_)),
    assert(last_state(X)).

% Obtener el último estado
get_last_state(X) :-
    last_state(X).

% Algoritmo A*
astar(OpenHeap) :-
    goal_found; goal_search(OpenHeap).

% Buscar el objetivo
goal_search(OpenHeap) :-
    get_from_heap(OpenHeap, _, [BoardId, GValue, [ManhattanValue, EuclideanValue], Path, Depth], RestHeap),
    display_search_state(BoardId, GValue, [ManhattanValue, EuclideanValue], Path, Depth),
    ( goal_reached(BoardId) -> goal_achieved(Path, BoardId, Depth)
    ; process_board(BoardId, GValue, Path, Depth, RestHeap)
    ).

% Lograr el objetivo
goal_achieved(Path, BoardId, Depth) :-
    format('Objetivo alcanzado en la ruta: ~w!~nGOAL ALCANZADO!!!~n', [Path]),
    display_board(BoardId),
    save_last_state(BoardId),
    get_last_state(NewBoard),
    ejemplo(NewBoard),
    format('Profundidad del arbol: ~d~n', [Depth]),
    assert(goal_found), !.

% Procesar un tablero
process_board(BoardId, GValue, Path, Depth, RestHeap) :-
    \+ is_board_visited(BoardId),
    set_board_as_visited(BoardId),
    findall([ChildBoardId, ChildDirection], generate_child_board(BoardId, ChildBoardId, ChildDirection), Children),
    NewDepth is Depth + 1,
    expand_children(Children, GValue, Path, NewDepth, RestHeap, NewHeap),
    astar(NewHeap).

process_board(_, _, _, _, RestHeap) :-
    astar(RestHeap).

% Expandir tableros hijos
expand_children([], _, _, _, OpenHeap, OpenHeap).
expand_children([[ChildId, ChildDir]|RestChildren], GValue, Path, Depth, OpenHeap, NewHeap) :-
    heuristic(ChildId, [ManhattanValue, EuclideanValue]),
    NewGValue is GValue + 1,
    FValue is NewGValue + ManhattanValue, % Usamos la heuristica Manhattan para la prioridad de la cola
    append(Path, [ChildDir], NewPath),
    add_to_heap(OpenHeap, FValue, [ChildId, NewGValue, [ManhattanValue, EuclideanValue], NewPath, Depth], UpdatedHeap),
    expand_children(RestChildren, GValue, Path, Depth, UpdatedHeap, NewHeap).

% Mostrar el estado de busqueda
display_search_state(BoardId, GValue, [ManhattanValue, EuclideanValue], Path, Depth) :-
    format('Estado actual del tablero (Costo de movimientos: ~d, H. Manhattan: ~d, H. Euclidiana: ~2f, Profundidad: ~d): ~w~n',
           [GValue, ManhattanValue, EuclideanValue, Depth, Path]),
    display_board(BoardId),
    format('Total heuristica Manhattan: ~d~n', [ManhattanValue]),
    format('Total heuristica Euclidiana: ~2f~n', [EuclideanValue]).

% Mostrar y guardar el tablero
display_and_store_board(BoardId, Matrix) :-
    writeln('Board rows:'),
    findall([RowIndex, Row], board_row(BoardId, RowIndex, Row), Rows),
    sort(Rows, SortedRows),
    writeln('Sorted rows:'),
    forall(member([Index, Row], SortedRows), writeln([Index, Row])),
    writeln('Empty at:'),
    board_empty(BoardId, EmptyRow, EmptyCol),
    write([EmptyRow, EmptyCol]), nl,
    % Construir la matriz de filas
    build_matrix(SortedRows, Matrix).

% Construir una matriz a partir de las filas ordenadas del tablero
build_matrix([], []).
build_matrix([[_, Row]|RestRows], [Row|MatrixRest]) :-
    build_matrix(RestRows, MatrixRest).

% Ejemplo de uso
ejemplo(BoardId) :-
    display_and_store_board(BoardId, Matrix),
    % Aqui puedes usar la matriz Matrix como necesites
    writeln('Matriz de filas del tablero:'),
    maplist(writeln, Matrix).

% Convierte el tablero a una lista unidimensional
flatten_board([Row1, Row2, Row3], FlatList) :-
    append(Row1, Row2, TempList),
    append(TempList, Row3, FlatList).

% Filtra el elemento 'empty' y lo convierte a 0
filter_empty([], []).
filter_empty([empty|Tail], [0|FilteredTail]) :-
    filter_empty(Tail, FilteredTail).
filter_empty([Head|Tail], [Head|FilteredTail]) :-
    filter_empty(Tail, FilteredTail).

% Cuenta las inversiones en una lista y almacena los pares que causan inversiones
count_inversions(List, Count, Inversions) :-
    count_inversions(List, 0, Count, [], Inversions).

count_inversions([], Count, Count, Inversions, Inversions).
count_inversions([Head|Tail], Acc, Count, InvAcc, Inversions) :-
    findall((Head, X), (member(X, Tail), X \= 0, Head \= 0, Head > X), CurrentInversions),
    length(CurrentInversions, CurrentCount),
    NewAcc is Acc + CurrentCount,
    append(InvAcc, CurrentInversions, NewInvAcc),
    count_inversions(Tail, NewAcc, Count, NewInvAcc, Inversions).

% Verifica si el tablero inicial es solucionable
check_solvable(InitialBoard) :-
    (is_3x3_board(InitialBoard) ->
        flatten_board(InitialBoard, FlatList),
        filter_empty(FlatList, FilteredList),
        count_inversions(FilteredList, Count, Inversions),
        writeln('Inversions:'), writeln(Inversions),
        writeln('Number of inversions:'), writeln(Count),
        (0 is Count mod 2
            -> writeln('The puzzle is solvable.')
            ;  writeln('The puzzle is not solvable.'), abort
        )
    ;
        writeln('El tablero no es de tamaño 3x3, no se verificará si es solucionable.')
    ).

% Verifica el tamaño del tablero
is_3x3_board(Board) :-
    length(Board, 3),
    maplist(length_(3), Board).

length_(Length, List) :-
    length(List, Length).

% Prueba del algoritmo A*
test_astar :-
    % Suponiendo que clear_visited_boards, clear_all_boards, clear_goal_board y clear_goal están definidos en otro lugar
    clear_visited_boards,
    clear_all_boards,
    clear_goal_board,
    clear_goal,
    InitialBoard = [
        [1, 2, 3],
        [4, 5, 6],
        [7 ,8, empty]
    ],
    GoalBoard = [
        [1, 2, 3],
        [8, empty, 4],
        [7, 6, 5]
    ],
    % Verificar si el tablero inicial es solucionable
    check_solvable(InitialBoard),
    % Suponiendo que create_board_from_list y create_goal_board_from_list están definidos en otro lugar
    create_board_from_list(InitialBoard, BoardId),
    create_goal_board_from_list(GoalBoard, NewGoalBoard),
    start_astar(BoardId), !.

