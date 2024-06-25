
# RetrAIGo

**RetrAIGo** es un paquete de Prolog que incluye tres módulos principales: `utils`, `heuristic_astar` y `guess`. Estos módulos implementan colectivamente el algoritmo de búsqueda A* (búsqueda informada) con heurísticas Manhattan y Euclidiana para resolver el problema del 8-puzzle desde un estado inicial hasta un estado objetivo. La idea es comprender como funciona el enfoque de IA simbólica "Rule Based" en una situación como resolver un juego tipo 8-puzzle con dimensiones y estado inicial y final no triviales.

## Tabla de Contenidos
- [Descripción](#descripción)
- [Módulos](#módulos)
  - [utils](#utils)
  - [heuristic_astar](#heuristic_astar)
  - [guess](#guess)
- [Instalación](#instalación)
- [Uso](#uso)
  - [Ejecutar una Prueba](#ejecutar-una-prueba)
  - [Cambiar Entrada y Salida](#cambiar-entrada-y-salida)

## Descripción

El paquete RetrAIGo está diseñado para resolver el problema del 8-puzzle utilizando el algoritmo de búsqueda A*. Los módulos proporcionan funcionalidades para gestionar tableros de juego, realizar movimientos, calcular heurísticas y ejecutar la búsqueda A*. El módulo `utils` proporciona predicados auxiliares para operaciones de listas y manipulaciones de tableros. El módulo `heuristic_astar` calcula las distancias Manhattan y Euclidiana y el valor heurístico total para el algoritmo A*. El módulo `guess` implementa el núcleo del algoritmo A* y gestiona el estado de los tableros durante el proceso de búsqueda.

## Módulos

### utils

El módulo `utils` incluye varios predicados de utilidad para manipulaciones de listas y tableros.

```prolog
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
```

### heuristic_astar

El módulo `heuristic_astar` proporciona funciones para calcular distancias Manhattan y Euclidianas y computar el valor heurístico para el algoritmo A*.

```prolog
:- module(heuristic_astar, [
    manhattan_distance/3,
    euclidean_distance/3,
    heuristic/2,
    current_position/3
]).
```

### guess

El módulo `guess` gestiona el algoritmo de búsqueda A*, incluyendo la creación de tableros, la verificación de objetivos y la gestión del estado durante el proceso de búsqueda.

```prolog
:- module(guess,[
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
```

## Instalación

Para instalar y ejecutar el paquete RetrAIGo, debe de seguir los siguientes pasos:

###**1.Instalar SWI-Prolog**: 
Asegúrese de tener SWI-Prolog instalado en el sistema. Puede descargarlo desde [SWI-Prolog website](https://www.swi-prolog.org/Download.html).

###**2. Ubicar la linea de comando en la ruta RetrAIGo**: 
Primero es necesario abrir una consola de comandos, en el directorio de RetrAIGo, en donde se descomprimió el archivo y se encuentran los 3 módulos. En caso de ser necesario puede seguir los siguientes pasos:
* Abrir la línea de comando
* Abra el explorador de archivo y úbiquese en el archivo de RetrAIGo. Copie la ruta que se encuentra en el buscador de archivos.
* Diríjase a la linea de comando e ingrese el comando cd y consecutivamente la ruta:


###**3. Cargar los Módulos**: 
Desde la misma línea de comando debe de iniciar SWI-Prolog y carga los módulos de Retraigo.
   ```sh
   swipl
   ```

   En la consola de SWI-Prolog, carga los módulos:
   ```prolog
   ?- [utiles].
   ?- [heuristic_astar].
   ?- [guess].
   ```
Es necesario destacar que para poder compilar cada uno de los módulos debe de encontrarse en la ruta en donde se encuentran los tres archivos.

## Ejecución

Asegúrese haber seguido los pasos de la guía de instalación y haber cargado los módulos indicados en el paso 3 de la misma.

Para ejecutar una prueba del algoritmo A* en un tablero de 8-puzzle predefinido, usa el predicado `test_astar/0` del módulo `guess`:

```prolog
?- test_astar.
```

###Salidas
En la salida por consola se mostrará el tablero inicial, el tablero objetivo y los pasos tomados por el algoritmo A* para alcanzar el objetivo.
#####1. Tablero inicial y tablero meta:
Anotación: Si el tablero es 3x3 se determina si es solucionable y sus inversiones. En cualquier otro caso diferente al mencionado esto no se realizará.
El ejemplo a continuación demuestra la salida del inicio de A* y luego de otro paso siguiente de ese estado inicial.
Ejemplo del tablero inicial, inversiones (si es 3x3) y tablero meta:
```
2 ?- test_astar.
Inversions:
[(8,4),(8,5),(8,7),(8,6),(7,6)]
Number of inversions:
5
The puzzle is not solvable.
Tablero inicial:
Board rows:
[1,[1,2,3]]
[2,[8,4,5]]
[3,[empty,7,6]]
Empty at:
[3,1]
Tablero meta:
Board rows:
[1,[1,2,3]]
[2,[8,empty,4]]
[3,[7,6,5]]
Empty at:
[2,2]
```
#####2. Algoritmo A* 
A continuación se muestra la salida principal del algoritmo de A*, en donde se muestra:
1. **Nombre del tablero**
2. **Estado actual con la siguiente información: **
* Costo de movimiento.
* Valor de las heuristicas de Manhattan y Euclidiana.
* Profudidad del nodo. 
* Path: Movimientos realizados, es necesario recordar que no se ha hecho ningún movimiento por lo que saldrá vacío.
* Se muestra inicialmente el tablero inicial.
* Posición en donde se encuentra la "ficha" empty 
* Calculo de los valores de las heuristicas desde el tablero inicial y el tablero meta
```
Iniciando A*:
board_1
Estado actual del tablero (Costo de movimientos: 1, H. Manhattan: 6, H. Euclidiana: 5.41, Profundidad: 1): []
Board rows:
[1,[1,2,3]]
[2,[8,4,5]]
[3,[empty,7,6]]
Empty at:
[3,1]
Total heuristica Manhattan: 6
Total heuristica Euclidiana: 5.41
```
En este caso sólo se muestra la primera parte del algoritmo A* y se calculan las inversiones si el tablero es 3x3, a continuación se muestra la salida después de el primer movimiento:
```
Estado actual del tablero (Costo de movimientos: 2, H. Manhattan: 4, H. Euclidiana: 4.00, Profundidad: 2): [right]
Board rows:
[1,[1,2,3]]
[2,[8,4,5]]
[3,[7,empty,6]]
Empty at:
[3,2]
Total heuristica Manhattan: 4
Total heuristica Euclidiana: 4.00
```
En este caso pordemos observar los siguientes cambios:
1. El costo del movimiento ha aumentado. 
2. El valor de las heuristicas se ha recalculado con el nuevo estado
3. Se ha ingresado el movimiento realizado en el path. 
4. Se muestra el estado actual del tablero con el nuevo movimiento.

### Cambiar Entrada y Salida

Asegúrese de haber seguido los pasos de la guía de instalación y haber cargado los módulos indicados en el paso 3 de la misma.

Puede cambiar la entrada (tablero inicial) y la salida (tablero objetivo) modificando el predicado `test_astar/0` en el módulo `guess`.

1. **Abrir el módulo `guess`**: Localice el predicado `test_astar/0`.

2. **Modificar los Tableros Inicial y Objetivo**: Cambie las variables `InitialBoard` y `GoalBoard` a tus configuraciones deseadas.
   ```prolog
   test_astar :-
    clear_visited_boards,
    clear_all_boards,
    clear_goal_board,
    clear_goal,
    InitialBoard = [
        [2, 8, 3],
        [1, 6, 4],
        [7 , empty, 5]
    ],
    GoalBoard = [
        [1, 2, 3],
        [8, empty, 4],
        [7, 6, 5]
    ],
    create_board_from_list(InitialBoard, BoardId),
    create_goal_board_from_list(GoalBoard, NewGoalBoard),
    writeln('Tablero inicial:'),
    display_board(BoardId),
    writeln('Tablero meta:'),
    display_goal(NewGoalBoard),
    % Prueba del tamano de la fila
    writeln('\nIniciando A*:'),
    start_astar(BoardId), !.
   ```

3. **Guardar y Recargar**: Guarde los cambios y recargue el módulo `guess` en SWI-Prolog.
   ```prolog
   ?- [guess].
   ```

4. **Ejecutar la Prueba**: Ejecute el predicado `test_astar/0` nuevamente para ver los resultados con los nuevos tableros.
   ```prolog
   ?- test_astar.
   ```

Siguiendo estas instrucciones, puede ejecutar y modificar con éxito el algoritmo de búsqueda A* para diferentes configuraciones del problema del 8-puzzle utilizando el paquete RetrAIGo.