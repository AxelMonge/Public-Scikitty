:- module(guess, [
    clear_visited_boards/0,
    clear_goal_found/0,
    set_board_as_visited/1,
    is_board_visited/1,
    save_total_colum/1,
    save_total_row/1,
    get_total_colum/1,
    get_total_row/1,
    board_row_size/3,
    board_total_rows/2,
    create_board_from_list/2,
    convert_board_to_list/2,
    generate_new_board_id/1,
    clear_all_boards/0,
    clear_board/1,
    clone_board/2,
    add_row_to_board/2,
    update_board_row/3,
    add_empty_to_board/3,
    get_valid_move/3,
    valid_move/4,
    apply_move_to_board/2,
    apply_horizontal_move/2,
    apply_vertical_move/2,
    generate_child_board/3,
    goal_row/3,
    goal_reached/1,
    start_astar/1,
    test_astar/7,
    astar/1,
    goal_search/1,
    goal_achieved/3,
    goal_position/2,
    process_board/5,
    expand_children/6,
    display_search_state/1,
    board_row/3
]).

:-use_module(utiles, [
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
:-use_module(heuristic_astar, [
    manhattan_distance/3,
	heuristic/2, 
	current_position/3
]).
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
:- dynamic visited_board/1.
:- dynamic goal_found/0.
:- dynamic goal_board/1.
:- dynamic goal_metrics/4.
:- dynamic heuristics/3. 

% Obtener el tablero objetivo
get_goal_board(X) :-
    goal_board(X).

clear_visited_boards :-
    retractall(visited_board(_)).

clear_goal_found :-
    retractall(goal_found).

% Limpiar el tablero objetivo
clear_goal_board :-
    retractall(goal_board()).

% Guardar el tablero objetivo
save_goal_board(GoalBoard):-
    retractall(goal_board()),
    assert(goal_board(GoalBoard)).

set_board_as_visited(BoardId) :-
    convert_board_to_list(BoardId, List),
    term_hash(List, HashId),
    assert(visited_board(HashId)).

is_board_visited(BoardId) :-
    convert_board_to_list(BoardId, List),
    term_hash(List, HashId),
    visited_board(HashId).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
:- dynamic(total_colum/1).
:- dynamic(total_row/1).

:- dynamic(goal_colums/1).
:- dynamic(goal_rows/1).

save_total_colum(X) :-
    retractall(total_colum(_)),
    assert(total_colum(X)).

save_total_row(X) :-
    retractall(total_row(_)),
    assert(total_row(X)).	

get_total_colum(X) :-
    total_colum(X).

get_total_row(X) :-
    total_row(X).
	
board_row_size(BoardId, RowIndex, Size) :-
    board_row(BoardId, RowIndex, Row),
    length(Row, Size),
	save_total_colum(Size).
	
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
	
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

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

convert_board_to_list(BoardId, List) :-
    findall(Row, (board_row(BoardId, _, Row)), List).

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

generate_new_board_id(BoardId) :- new_id('board_', BoardId).

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

clear_board(BoardId) :-
    retractall(board_row(BoardId, _, _)),
    retractall(board_empty(BoardId, _, _)).

clone_board(BoardId, ClonedBoardId):-
    findall([RowId, Row], board_row(BoardId, RowId, Row), EnumeratedList),
    generate_new_board_id(ClonedBoardId),
    forall(member(RowClone, EnumeratedList), add_row_to_board(ClonedBoardId, RowClone)),
    board_empty(BoardId, EmptyRow, EmptyCol),
    add_empty_to_board(ClonedBoardId, EmptyRow, EmptyCol).

add_row_to_board(BoardId, [RowIndex, Row]) :-
    assert(board_row(BoardId, RowIndex, Row)),
    ( index_of(empty, Row, EmptyIndex) -> add_empty_to_board(BoardId, RowIndex, EmptyIndex) ; true ).

update_board_row(BoardId, RowIndex, UpdatedRow) :-
    retract(board_row(BoardId, RowIndex, _)),
    add_row_to_board(BoardId, [RowIndex, UpdatedRow]).

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

get_valid_move(BoardId, Position, Direction) :-
    board_empty(BoardId, EmptyRow, EmptyCol),
    valid_move(EmptyRow, EmptyCol, Position, Direction).

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

apply_move_to_board(BoardId, Direction) :-
    apply_horizontal_move(BoardId, Direction);
    apply_vertical_move(BoardId, Direction).

apply_horizontal_move(BoardId, Direction) :-
    (Direction = left; Direction = right),
    get_valid_move(BoardId, [Row, NewCol], Direction),
    board_row(BoardId, Row, RowData),
    board_empty(BoardId, Row, EmptyCol),
    list_swap(RowData, EmptyCol, NewCol, UpdatedRow),
    update_board_row(BoardId, Row, UpdatedRow),
    add_empty_to_board(BoardId, Row, NewCol).

apply_vertical_move(BoardId, Direction) :-
    (Direction = up; Direction = down),
    get_valid_move(BoardId, [NewRow, Col], Direction),
    board_empty(BoardId, EmptyRow, Col),
    swap_vertical(BoardId, EmptyRow, Col, NewRow, Col),
    add_empty_to_board(BoardId, NewRow, Col).

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

start_astar(BoardId) :-
    clear_visited_boards,
    clear_goal_found,
    heuristic(BoardId, HValue),
    empty_heap(OpenHeap),
    add_to_heap(OpenHeap, HValue, [BoardId, 1, HValue, [], 1], UpdatedOpenHeap),
    astar(UpdatedOpenHeap).
%%%%%%%%%%%%%%%%%%%%%%%agregar
astar(OpenHeap) :-
    goal_found; goal_search(OpenHeap).

goal_search(OpenHeap) :-
    get_from_heap(OpenHeap, _, [BoardId, GValue, [ManhattanValue, EuclideanValue], Path, Depth], RestHeap),
    display_search_state([ManhattanValue, EuclideanValue]),
    ( goal_reached(BoardId) -> goal_achieved(Path, BoardId, Depth)
    ; process_board(BoardId, GValue, Path, Depth, RestHeap)
    ).

goal_achieved(Path, BoardId, Depth) :-
    generate_board_matrix(BoardId, M),
    assert(goal_metrics(1, Path, Depth, M)),
    assert(goal_found), !.

% Predicado principal para generar la matriz del tablero
generate_board_matrix(BoardId, M) :- 
    % Obtener todas las filas del tablero
    findall([RowIndex, Row], board_row(BoardId, RowIndex, Row), Rows),
    % Ordenar las filas por su índice
    sort(Rows, SortedRows),
    % Crear una lista con las filas del tablero
    build_matrix(SortedRows, Matrix),
    % Asignar la matriz resultante al parámetro M
    M = Matrix.

% Predicado auxiliar para construir la matriz a partir de las filas ordenadas
build_matrix([], []).
build_matrix([[_, Row]|Rest], [Row|MatrixRest]) :-
    build_matrix(Rest, MatrixRest).

process_board(BoardId, GValue, Path, Depth, RestHeap) :-
    \+ is_board_visited(BoardId),
    set_board_as_visited(BoardId),
    findall([ChildBoardId, ChildDirection], generate_child_board(BoardId, ChildBoardId, ChildDirection), Children),
    NewDepth is Depth + 1,
    expand_children(Children, GValue, Path, NewDepth, RestHeap, NewHeap),
    astar(NewHeap).

process_board(_, _, _, _, RestHeap) :-
    astar(RestHeap).

expand_children([], _, _, _, OpenHeap, OpenHeap).
expand_children([[ChildId, ChildDir]|RestChildren], GValue, Path, Depth, OpenHeap, NewHeap) :-
    heuristic(ChildId, [ManhattanValue, EuclideanValue]),
    NewGValue is GValue + 1,
    FValue is NewGValue + ManhattanValue, % Usamos la heurística Manhattan para la prioridad de la cola
    append(Path, [ChildDir], NewPath),
    add_to_heap(OpenHeap, FValue, [ChildId, NewGValue, [ManhattanValue, EuclideanValue], NewPath, Depth], UpdatedHeap),
    expand_children(RestChildren, GValue, Path, Depth, UpdatedHeap, NewHeap).

display_search_state([ManhattanValue, EuclideanValue]) :-
    retractall(heuristics(1)),
    assert(heuristics(1, ManhattanValue, EuclideanValue))
.

% Verificar si un tablero es solucionable
check_solvable(Board) :-
   (is_3x3_board(Board) ->
        flatten(Board, FlatBoard),
        exclude(=(empty), FlatBoard, NumberedTiles),
        count_inversions(NumberedTiles, InversionCount),
        ( 0 is InversionCount mod 2 -> true
        ; abort
        )
    ; true).

count_inversions(List, Count) :-
    findall(1, (nth1(I, List, X), nth1(J, List, Y), I < J, X > Y), Inversions),
    length(Inversions, Count).

% Verificar si un tablero es de 3x3
is_3x3_board(Board) :-
    length(Board, 3),
    forall(member(Row, Board), length(Row, 3)).

test_astar(InitialBoard, GoalBoard, Path, Depth, Goal, ManhattanValue, EuclideanValue) :-
    clear_visited_boards,
    clear_all_boards,
    clear_goal_board,
    clear_goal,
    create_board_from_list(InitialBoard, BoardId),
    create_goal_board_from_list(GoalBoard, _),
    % check_solvable(BoardId),
    start_astar(BoardId),
    goal_metrics(1, Path, Depth, Goal),
    heuristics(1, ManhattanValue, EuclideanValue),
    !.
