# Scikitty

## 1) Descripción del sprint #1 del proyecto
En el siguiente sprint se podrán observar la primera parte de la implementación y la creación del modelo de árbol de decisión para variables categóricas multiclase, continuas y binarias. En donde se busca poder entrenar y regular (por medio de hiperparámetros) un modelo por medio de un dataset y mediante variables de testeo poder observar las decisiones tomadas por medio del árbol, además de poder serializar el árbol en un documento .json. 

## 2) Ambiente:
Este proyecto se deberá de correr desde el prompt Anaconda, en donde se deberá de tener instalado las librerias graphviz, pandas y numpy. Siguiendo los siguientes pasos, se prepará el ambiente para poder utilizar las demos del proyecto:
1. Descargar e instalar el ambiente Anaconda en caso de no tenerlo desde su página oficial: https://www.anaconda.com/download
2. Una vez instalado, en el buscardor de aplicaciones del sistema operativo, buscar **"Anaconda Prompt"** y abrir dicha aplicación.
3.  Se puede verificar las liberias instaladas en el ambiente Anaconda utilizando el siguiente comando en la consola de la aplicación:
```
conda list
```
4. Si en dicha lista no se encuentra las librerias **graphviz, pandas y numpy** desde la misma consola de Anaconda se podrán instalar mediante los siguientes comandos:
```
conda install graphviz
conda install pandas
conda install numpy
```
**Nota Importante:** durante la instalación de las librerias saldrá el mensaje:
```
Proceed ([y]/n)?
```
Es importante introduccir **"y"** y presionar la tecla intro para poder instalar las versiones más nuevas y dependencias necesarias de cada uno de los paquetes a instalar.
## 3) Datasets

En este proyecto se estarán utilizando diferentes datasets. Para temas del primer seguimiento se estarán utilizando tres en específicos:
1. **fictional_disease.csv:**
* **Target:** Disease.
* **Características:** Age, Gender, SmokerHistory.
El modelo busca predecir si una persona es enferma o no, por medio de características como la edad, el género y su historial de fumador.

2. **fictional_reading_place.csv:** 
* **Target:** User action.
* **Características:** Author, Thread, Length, Where read.
El modelo busca predecir si un lector lee o se salta una lectura, según características como el autor, hilo, longitud y el lugar de lectura. 

3. **playTennis.csv:** 
* **Target:** Play Tennis.
* **Características:** Outlook, Temperature, Humidity, Wind
El modelo busca predecir si una persona juega o no tennis por medio de características como el pronóstico del clima, la temperatura, la humedad y el viento.

El modelo creado para este proyecto tendrá la capacidad de poder validar las variables categóricas binarios, multiclase y continuas. Por lo que sí tomará en cuenta estas variables de los dataset y podrá tomar las desiciones en base a estas. En consecuencia se puede tomar en cuenta el siguiente dataset:

1. **CO2_car_emision.csv:**
* **Target:** CO2.
* **Características:** Car, Model, Volume, Weight.
El modelo busca predecir la cantidad de CO2 que genera un automóvil, según marca, modelo, volumen y peso del automóvil.

## 4) ¿Cómo ejecutar el proyecto?
Dentro de la carpeta raiz del proyecto **"Scikitty"**, se encontrá una carpeta llamada **"demos"**. Aquí se almacenan cada uno de los scripts con las demos para cada uno de los datasets planteados. Tanto su versión utilizando la libreria **"Scikitty"** como la versión utilizando **"Scikitlearn**. Para poder ejecuta cada uno de los scripts, es necesario seguir cada uno de los siguientes pasos:
1. Abrir Anaconda Prompt. 
2. Se debe de ubicar en el directorio Scikitty en la caperta 'demos' utilizando el comando **cd** en la consola de Anaconda Prompt, en donde se encuentran los scripts correspondiente a cada uno de los dataset. 
3. Una vez ubicado en el directorio de las **'demos'**, se podrá ejecutar cada uno de los scripts. Es necesario colocar los siguientes comando para ejecutarlos:
*   Para poder ejecutar el árbol de decisión de **fictional_disease.csv: **
```
python fictional_disease.py
```
* Para poder ejecutar el árbol de decisión de **fictional_reading_place.csv** 
```
python fictional_reading_place.py
```
* Para poder ejecutar el árbol de decisión de **playTennis.csv: **
```
python playTennis.py
```
* Para poder ejecutar el árbol de decisión de **CO2_car_emision.csv**
```
python CO2_car_emision.py
```
En el caso de querer compararlo con las salidas de la librería de **Scikit Learn**, puede ejecutar los siguientes scripts, para comparar los resultados de las dos librerias. Tener en cuenta que los siguientes scripts utilizan la libreria Matplotlib para realizar la grafica del árbol, esto genera que hasta que no se cierre la ventana emergente del grafico del árbol, el flujo de la consola de Anaconda no sigue.

* Para poder ejecutar el resultado de **scikitlearn del dataset fictional_disease.csv: **
```
python fictional_disease-scikitlearn.py
```
* Para poder ejecutar el resultado de *scikitlearn* del dataset **fictional_reading_place.csv:**
```
python fictional_reading_place-scikitlearn.py
```
* Para poder ejecutar el resultado de *scikitlearn* del dataset **playTennis.csv:**
```
python playTennis-scikitlearn.py
```
* Para poder ejecutar el árbol de decisión de *scikitlearn* del dataset **CO2_car_emision.csv**
```
python CO2_car_emision-scikitlearn.py
```

## 5) Salidas:
Con respecto de los resultados de las métricas se utilizó la librería de Sklearn.metrics. Para visualizar el árbol se usará la clase TreeVisualizer, el cual se encargará de generar una imágen .png, con ayuda de la librería Graphviz. Esta información, tanto las metricas que se muestran el la consola de Anaconda, como las graficas las cuales se mostraran en un png emergente, se muestran al ejecutar los scripts de cada uno de las demos de los datasets. 

**SCI-KITTY: **
Se mostrará los datos del árbol entrenado original y una vez que se cree el archivo JSON se recuperará y se guardará el modelo entrenado en un nuevo árbol, en donde se mostrarán y se compararán los siguientes datos del árbol original y del recuperado:
* **Exactitud:** Muestra las etiquetas que han predicho correctamente.
* **Precisión:**  Muestra las etiquetas predichas positivas que son correcta.
* **Recall:** Muestra las etiquetas positivas reales que se han predicho correctamente.
* **F1-score:** Muestra la predicción que tuvo el árbol en la fase de prueba
* **Matriz de confusión: ** Se mostrará la matriz de confusión.
* **Etiquetas predichas: **Se mostrará cuáles fueron las etiquetas que predijo el modelo.
* **Etiquetas reales:** Se mostrará cuáles son las etiquetas reales. Con el fin de poder compararlas con las etiquetas predichas.
* **Visualización del árbol:** Para poder visualizar el árbol entrenado original, se mostrará en un png emergente con el respectivo grafico del árbol.
* **Archivo JSON:** Se crea un archivo JSON en donde se guardarán los nodos con información relevante para poder recontruir el árbol, como lo es:
1. Es_hoja: True/ False
2. Regla: [índice, condición, valor ]
3. La etiqueta
4. Impureza
5. Etiquetas: Los valores de las muestras
6. La cantidad de muestras
* **Visaulización del árbol recuperado:** Se visualizará el árbol recuperado por medio del archivo JSON, se mostrará en un png emergente con el respectivo grafico del árbol.

**SCI-KIT LEARN**
Se mostrarán los mismos datos que en el caso de SCI-KITTY exceptuando el árbol cargado. Para visualizar el árbol se usará la librería de Matplotlib.

## 6) Hiperparámetros
En caso de querer cambiar los hiperparámetros, se tendrá que modificar directamente el archivo a ejecutar en donde se podrán reconocer las variables:
* **max_depth:** Recibirá el número máximo de profundidad que contendrá el árbol de decisión.
* **min_samples_split:** Recibirá el número mínimo de ejemplares que debe de tener un nodo para poder generar un split.

Estos hiperparámetros, tendrán que se remplazados los símbolos '?' y colocar el número que sea deseado. Con esto se podrán ajustar la generación del árbol de decisión:

```
dt = DecisionTreeClassifier(criterion='entropy', min_samples_split=?, max_depth=?)
	
```

## Información Adicional
Scikitty es una libreria de Python que simula el comportamiento de la libreria Scikitlearn. Es un proyecto desarrollado por Axel Monge Ramírez, Andrel Ramírez Solis, John Rojas Chinchilla y Abigail Salas Ramírez. Estudiantes de la Universidad Nacional de Costa Rica, matriculados en el curso de Inteligencia Artificial (EIF420-O) durante el primer semestre del 2024 en la Facultad de Ciencias Exactas. Este proyecto representa sus esfuerzos de colaboración y puede comunicarse con ellos por correo electrónico:  axel.monge.ramirez@est.una.ac.cr, andrel.ramirez.solis@est.una.ac.cr, john.rojas.chinchilla@est.una.ac.cr y abigail.salas.ramirez@est.una.ac.cr.