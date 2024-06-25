#  Artificial Intelligence Demos
Artificial Intelligence Demos es una página web con diferentes demos de inteligencia artificial utilizando una librería creada desde cero en Python llamada SciKitty. Esta librería está basada en otra muy popular en el ámbito de la IA llamada Scikit-learn.

## Inicialización de la Página Web
Antes de poder utilizar la página web, es necesario tener algunas herramientas instaladas, así como hacer algunos procedimientos importantes para su uso.

Primero, hay que descargar el repositorio y descomprimirlo usando una aplicación como WinRAR o 7-Zip. Una vez descomprimido el proyecto, dentro de la carpeta principal, se deben observar dos subcarpetas llamadas client y service.
Es necesario tener instalado Node.js. Se puede instalar desde su página oficial https://nodejs.org/en. Una vez instalado el "Node.js (LTS)" bastará con ejecutarlo y presionar en "siguiente" en todas las ventanas que aparecerán en su instalador.
Una vez instalado Node.js, es necesario ubicarse en la carpeta client desde una consola utilizando cd o abriéndola directamente. Una vez ubicado sobre la carpeta client, se ejecutará el comando npm install para poder instalar todas las dependencias de la página web.
```
npm pip install
```
4. En el caso de que todas las dependencias fueran instaladas correctamente, solo queda ejecutar o levantar la página web, utilizando npm run dev. Esto inicializará la página en el puerto localhost:3000. Se debe ingresar dicho puerto en la URL de algún navegador para poder acceder a la página web.
```
localhost:3000
```

## Inicialización del Servidor
El servidor permitirá a la web interactuar con código Prolog y Python que ayudará a generar los Árboles de Decisión, las Regresiones Lineales, una IA Basada en Reglas, entre otras cosas.

1. Es necesario tener alguna versión de Python instalada. Para poder descargar alguna, se puede realizar desde la página principal https://www.python.org/downloads/. Una vez instalado y ejecutado el instalador, quedará listo Python para usarse.
2. Se necesitará de la librería Graphviz el cual se puede descargar su instalador desde la página oficial https://graphviz.org/download/. Es importante a la hora de instalarlo, seleccionar la opción que añade graphviz al PATH.
3. Al igual que con la carpeta client, se debe ubicar en la carpeta service utilizando una consola y el comando cd.
4. Es necesario crear un environment para poder manejar las dependencias del servidor alojado en Python. Primero es necesario instalar la dependencia virtualenv utilizando el comando python -m pip install --user virtualenv. Ya solo falta crear el ambiente utilizando python -m virtualenv <nombre> y activarlo usando .<nombre>\Scripts\activate.
```
python -m pip install --user virtualenv
python -m virtualenv <nombre>
 .\<nombre>\Scripts\activate
```
5. Una vez se está en el ambiente creado, se tendrán que instalar las dependencias utilizando pip install -r .\requirements.txt. Cuando las dependencias estén instaladas, se deben ejecutar los comandos python manage.py migrate y python manage.py makemigrations.
```
pip install -r .\requirements.txt
python manage.py migrate
python manage.py makemigrations
```
6. Finalmente, si todo ha salido bien, se deberá ejecutar el comando .\runServices.bat. Este comando ejecutará un script que se encargará de montar el servidor de Django en el puerto 127.0.0.1:8000 y otro servidor en Prolog en el puerto 127.0.0.1:8001. Ya solo queda entrar en la página web en el puerto localhost:3000 y usar los distintos demos que ahí se presentan.

## Demos
En la página principal de la web, se encontrarán 5 demos que se pueden utilizar para probar distintos conceptos de la inteligencia artificial.

### 1. SciKitty Decision Tree
En esta demo, se podrá entrenar un modelo de un Árbol de Decisión utilizando un dataset en el formato CSV.
1. Lo primero que se verá en esta demo es un espacio para subir un dataset en el formato CSV o la opción de seleccionar un dataset guardado en la aplicación web.
2. Una vez seleccionado y subido el dataset, la página mostrará dos selectores, uno para seleccionar el feature target y otro para seleccionar el criterio de impureza que se utilizará en el modelo del Árbol.
3. Se deberá presionar sobre el botón "Send dataset to train a decision tree" para entrenar el Árbol de Decisión. Este botón enviará una petición HTTP que se realizará en el servidor de Django el Árbol de Decisión utilizando el dataset subido, el target y el criterio de impureza seleccionado, utilizando el paquete de SciKitty instalado en Python.
4. Una vez el modelo esté creado, la página web mostrará las métricas del Árbol de Decisión, la matriz de confusión y la gráfica con el Árbol que se generó. Esto tanto para el Árbol de Decisión de SciKitty como el Árbol de Decisión de Scikit-learn.
5. Finalmente, en la página aparecerá un formulario con el cual seleccionar los features con los cuales realizar una predicción en el Árbol de Decisión. Se deberá presionar sobre el botón 'Predict' el cual enviará la petición HTTP a un servidor Prolog que se encargará de parsear el Árbol de Decisión que se encuentra en formato JSON, a un objeto de Prolog con el cual realizar la predicción.

### 2. RetrAiGo
Es un demo que permitirá observar el rendimiento de una IA Basada en Reglas utilizando una búsqueda informada conocida como A*, utilizando las heurísticas Manhattan y Euclidiana.
1. La página mostrará un input con el cual el usuario podrá cambiar el tamaño de los tableros que se utilizarán para realizar la búsqueda informada. Esto permite probar tableros de tamaños en un rango de 2x2 hasta 10x10, limitado para no afectar el rendimiento de la página.
2. Justo debajo de dicho input, se observarán dos tableros, uno nombrado Initial Board y otro Goal Board. Cada celda de estos tableros puede ser modificada por el usuario, presionando sobre dichas celdas y escribiendo el número que desea. Los números que se podrán ingresar son entre el rango de 0 hasta el tamaño total del tablero - 1, o sea (NxN) - 1. El cero será tomado como la celda vacía del juego.
3. Habrá un botón "Send Boards to Play", el cual, al ser presionado, enviará una petición HTTP al servidor de Prolog, en el cual se realizará la búsqueda informada, usando el initial board para intentar llegar al goal board.
4. Finalmente, una vez la búsqueda informada es realizada, el servidor devolverá parámetros como las heurísticas, la profundidad de la búsqueda, el camino que se realizó, así como el board resultante de la búsqueda.

### 3. SciKitty Tree Gradient Boosting
Se podrá entrenar un modelo de un Árbol de Decisión usando Decision Tree Gradient Boosting por medio de un dataset en el formato CSV. 
1. Lo primero que se verá en esta demo es un espacio para subir un dataset en el formato CSV o la opción de seleccionar un dataset guardado en la aplicación web.
2. Una vez seleccionado y subido el dataset, la página mostrará dos selectores, uno para seleccionar el feature target y otro para seleccionar el criterio de impureza que se utilizará en el modelo del Árbol.
3. Se deberá presionar sobre el botón "Send dataset to train a decision tree" para entrenar el Árbol de Decisión. Este botón enviará una petición HTTP que se realizará en el servidor de Django el Árbol de Decisión utilizando el dataset subido, el target y el criterio de impureza seleccionado, utilizando el paquete de SciKitty instalado en Python.
4. Una vez el modelo esté creado, la página web mostrará las métricas del Árbol de Decisión, la matriz de confusión y la gráfica con el Árbol que se generó. Esto tanto para el Árbol de Decisión de SciKitty como el Árbol de Decisión de Scikit-learn.

### 4. SciKitty Linear Regression
En esta demo, se podrá probar una regresión lineal implementada en el paquete SciKitty utilizando un dataset en CSV.
1. Lo primero que se verá en esta demo es un espacio para subir un dataset en el formato CSV o la opción de seleccionar un dataset guardado en la aplicación web.
2. Una vez seleccionado y subido el dataset, la página mostrará un selector para seleccionar el feature target que se utilizará en el modelo de Regresión Lineal de SciKitty.
3. La página muestra dos botones, uno que dice "Send Dataset to Create the Linear Regression" y otro que dice "Try with Sklearn California Housing Dataset". Utilizando el primer botón, se enviará una petición HTTP al servidor en Django que permitirá generar el modelo de Regresión Lineal utilizando el dataset subido y el target seleccionado. Por otro lado, el segundo botón, permitirá probar el modelo de Regresión Lineal de SciKitty usando el dataset California Housing de Sklearn.
4. Finalmente, en la página web se verán las métricas del modelo así como el gráfico que se generó.

### 5. SciKitty Logistic Regression
Se podrá probar el modelo de regresión logística implementado en el paquete SciKitty utilizando un dataset en CSV.
1. Lo primero que se verá en esta demo es un espacio para subir un dataset en el formato CSV o la opción de seleccionar un dataset guardado en la aplicación web.
2. Una vez seleccionado y subido el dataset, la página mostrará un selector para seleccionar el feature target que se utilizará en el modelo de Regresión Logística de SciKitty.
3. La página muestra dos botones, uno que dice "Send Dataset to Create the Logistic Regression" y otro que dice "Try with Sklearn Breast Cancer Dataset". Utilizando el primer botón, se enviará una petición HTTP al servidor en Django que permitirá generar el modelo de Regresión Logística utilizando el dataset subido y el target seleccionado. Por otro lado, el segundo botón, permitirá probar el modelo de Regresión Logística de SciKitty usando el dataset Breast Cancer de Sklearn.
4. Finalmente, en la página web se verán las métricas del modelo así como el gráfico que se generó.

## Información Adicional
Es un proyecto desarrollado por Axel Monge Ramírez, Andrel Ramírez Solis, John Rojas Chinchilla y Abigail Salas Ramírez. Estudiantes de la Universidad Nacional de Costa Rica, matriculados en el curso de Inteligencia Artificial (EIF420-O) durante el primer semestre del 2024 en la Facultad de Ciencias Exactas. Este proyecto representa sus esfuerzos de colaboración y puede comunicarse con ellos por correo electrónico:  axel.monge.ramirez@est.una.ac.cr, andrel.ramirez.solis@est.una.ac.cr, john.rojas.chinchilla@est.una.ac.cr y abigail.salas.ramirez@est.una.ac.cr.
