:- module(heuristic_astar, [
    manhattan_distance/3,
	heuristic/2, 
	current_position/3
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
:-use_module(guess, [
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
    goal_position/2,
    start_astar/1,
    test_astar/7,
	astar/1,
	goal_search/1,
	goal_achieved/3,
	process_board/5,
	expand_children/6,
	display_search_state/1,
	board_row/3
]).
%%%%%%%%%%%%%%%%%%%% HEURISTIC AND MANHATTAN DISTANCE %%%%%%%%%%%%%

manhattan_distance([X1, Y1], [X2, Y2], Distance) :-
    Distance is abs(X1 - X2) + abs(Y1 - Y2).

euclidean_distance([X1, Y1], [X2, Y2], Distance) :-
    Distance is sqrt((X1 - X2) * (X1 - X2) + (Y1 - Y2) * (Y1 - Y2)).

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

current_position(BoardId, Piece, [RowIndex, ColIndex]) :-
    board_row(BoardId, RowIndex, Row),
    nth1(ColIndex, Row, Piece).
