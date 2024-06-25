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
zip(List1, List2, ZippedList) :- maplist([X, Y, [X, Y]] >> true, List1, List2, ZippedList).

enumerate(List, EnumeratedList) :-
    length(List, Length), numlist(1, Length, Indices), zip(Indices, List, EnumeratedList).

index_of(Value, List, Index) :- nth1(Index, List, Value).

new_id(Base, Id) :- gensym(Base, Id).

list_split(List, Prefix, Element, Suffix) :- append(Prefix, [Element | Suffix], List).

list_set_value(List, Index, Value, ResultList) :-
    list_split(List, Prefix, _, Suffix),
    Index1 is Index - 1, Index1 >= 0,
    length(Prefix, Index1),
    append(Prefix, [Value | Suffix], ResultList).

list_at(List, Index, Value) :-
    nth1(Index, List, Value).

list_swap(List, Index1, Index2, SwappedList):-
    list_at(List, Index1, Value1), list_at(List, Index2, Value2),
    list_set_value(List, Index1, Value2, TempList),
    list_set_value(TempList, Index2, Value1, SwappedList).

swap_vertical(BoardId, Row1, Col1, Row2, Col2) :-
    board_row(BoardId, Row1, RowData1),
    board_row(BoardId, Row2, RowData2),
    list_at(RowData1, Col1, Value1),
    list_at(RowData2, Col2, Value2),
    list_set_value(RowData1, Col1, Value2, UpdatedRow1),
    list_set_value(RowData2, Col2, Value1, UpdatedRow2),
    update_board_row(BoardId, Row1, UpdatedRow1),
    update_board_row(BoardId, Row2, UpdatedRow2).
