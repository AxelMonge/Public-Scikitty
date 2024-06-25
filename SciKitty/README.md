# Proyecto
En el siguiente trabajo se implementarán dos modelos de machine learning, como lo son Decision Tree y Decision Tree Gradient Boosting. Además, como extras: Logistic Regression y Linear Regression.  En este apartado se explicará la implementación de las dependencias y cómo paso a paso se podrá ejecutar cada uno de los demos de este proyecto, también se comentarán a detalle cada uno de los datasets que se estarán usando en los modelos, así como el significado de cada salida mostrada en consola, teniendo la posibilidad de poder compararlo con las implementaciones que cuenta SKLearn. 
Es necesario tener en cuenta que la compilación del árbol a prolog se demuestra en el proyecto de la pagina web, el cual tiene su propio README.
A continuación se hablará en más detalle los modelos creados

# Scikitty

## 1) Descripción del sprint #2 del proyecto
En el siguiente sprint se podrán observar la continuación de la implementación y la creación del modelo de árbol de decisión para variables categóricas multiclase, continuas y binarias. Además de la implementación del un modelo de Decisition Tree Gradient Boosting para las mismas variables, que permite observar cada una de las iteraciones, así como los resultados que va obteniendo cada stump del modelo.
Como extra, se podrá ejecutar Linear Regression para el dataset california_housing por medio de la librería Sklearn, así mismo la implementación y ejecución de Logistic Regression para el dataset breast_cancer, tomado de la misma librería.

## 2) Ambiente:
Este proyecto se deberá de correr desde el prompt Anaconda, en donde se deberá de tener instalado las librerias graphviz, pandas y numpy. Siguiendo los siguientes pasos, se prepará el ambiente para poder utilizar las demos del proyecto:
1. Descargar e instalar el ambiente Anaconda en caso de no tenerlo desde su página oficial: https://www.anaconda.com/download
2. Una vez instalado, en el buscardor de aplicaciones del sistema operativo, buscar **"Anaconda Prompt"** y abrir dicha aplicación.
3. Se puede verificar las liberias instaladas en el ambiente Anaconda utilizando el siguiente comando en la consola de la aplicación:
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
En este proyecto se estarán utilizando diferentes datasets. 
**1. fictional_disease.csv:**

El modelo busca predecir si una persona es enferma o no, por medio de características como la edad, el género y su historial de fumador.

* **Target:** Disease.
* **Características:** Age, Gender, SmokerHistory.

**2. fictional_reading_place.csv:** 

El modelo busca predecir si un lector lee o se salta una lectura, según características como el autor, hilo, longitud y el lugar de lectura. 

* **Target:** User action.
* **Características:** Author, Thread, Length, Where read.


**3. playTennis.csv:** 

El modelo busca predecir si una persona juega o no tennis por medio de características como el pronóstico del clima, la temperatura, la humedad y el viento.

* **Target:** Play Tennis.
* **Características:** Outlook, Temperature, Humidity, Wind


**4. CO2_car_emision.csv:**

El modelo busca predecir la cantidad de CO2 que genera un automóvil, según marca, modelo, volumen y peso del automóvil.

* **Target:** CO2.
* **Características:** Car, Model, Volume, Weight.

En este proyecto también se incluirán datasets de los cuales no se contarán con demos, como lo son: titanic, penguins_gender, diabetes, cancer, diabetes, iris, entre otros.

## 4) ¿Cómo ejecutar el proyecto?
Dentro de la carpeta raiz del proyecto **"Scikitty"**, se encontrá una carpeta llamada **"demos"**. Aquí se almacenan cada uno de los scripts con las demos para cada uno de los datasets planteados. Siendo estas las implementaciones de  **"Scikitty"**,  **"Scikitlearn**,  **"Boosting"**,  **"Linear Regression"** y  **"Logistic Regression"**. Para poder ejecutar cada uno de los scripts, es necesario seguir cada uno de los siguientes pasos:
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
* **Visualización del árbol recuperado:** Se visualizará el árbol recuperado por medio del archivo JSON, se mostrará en un png emergente con el respectivo grafico del árbol.

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
#Decision Tree Gradient Boosting (DTGB)
La implementación de este modelo tiene como objetivo el entrenar diversos árboles débiles (altura 1) en donde cada uno de ellos obtiene los residuos del anterior, en donde se implementa el MSE (Error Promedio Cuadrático) y el GD (Descenso por gradiente) buscando mejorar los errores del árbol anterior, mejorando las predicciones, permitiendo crear un modelo fuerte.
##1) Datasets
En esta parte del proyecto se estarán utilizando algunos dataset ya antes mencionados para Scikitty. Para temas de boosting se estarán utilizando tres en específicos:

**1. playTennis_boosting.csv:**

El modelo busca predecir si una persona juega o no tennis por medio de características como el pronóstico del clima, la temperatura, la humedad y el viento.

* **Target:** Play Tennis.
* **Características:** Outlook, Temperature, Humidity, Wind

**2. fake_weights_boosting.csv:** 

El modelo busca predecir si 

* **Target:** 
* **Características:** .

**3. fictional_disease.csv:**

El modelo busca predecir si una persona es enferma o no, por medio de características como la edad, el género y su historial de fumador.

* **Target:** Disease.
* **Características:** Age, Gender, SmokerHistory.

## 2) Ejecución del Decision Tree Gradient Boosting (DTGB)
Para poder ejecutar cada uno de los scripts, es necesario seguir cada uno de los siguientes pasos:
1. Abrir Anaconda Prompt. 
2. Se debe de ubicar en el directorio Scikitty en la caperta 'demos' utilizando el comando **cd** en la consola de Anaconda Prompt, en donde se encuentran los scripts correspondiente a cada uno de los dataset. 
3. Una vez ubicado en el directorio de las **'demos'**, se podrá ejecutar cada uno de los scripts. Es necesario colocar los siguientes comando para ejecutarlos:
*   Para poder ejecutar el DTGB de ** playTennis_boosting.csv: **
```
python  playTennis_boosting.py
```
* Para poder ejecutar el  DTGB de **fake_weights_boosting.csv** 
```
python fake_weights_boosting.py
```
* Para poder ejecutar el  DTGB de **fictional_disease_boosting.csv: **
```
python fictional_disease_boosting.py
```

## 3) Salidas
La salida esperada muestra los ciclos de entrenamiento que contiene el modelo, permitiendo observar los residuos y el mejoramiento de las predicciones. En donde cada stump mejora los errores del anterior por medio del descenso por gradiente, logrando acercarse al objetivo
Las salidas esperadas para el DTGB serán:
####Salida para variables categóricas y continuas
* **Target continuo:**  True o False dependiendo de la naturaleza de este.
* **Target a predecir en fit:**  Muestra las etiquetas verdaderas para el conjunto de entrenamiento
* **Regla Inicial del Stump:** Se muestran feature inicial donde se especifica el número del feature, la operación y el valor. En el caso de las variables continuas se escoge el de mayor probabilidad
* **Inicialización de probabilidades de clase:** En el caso de las categoricas está basado en las probabilidades en cada clase según vaya prediciendo. Mientras que para el caso de los continuos se saca la media.
* **Residuales: **Se muestran los residuos después de la primera iteración, calculados como la diferencia entre las predicciones actuales y las etiquetas verdaderas. Los residuos muestran qué tan lejos están las predicciones actuales de las etiquetas verdaderas para cada ejemplo.
* **Reglas para los otros features del Stump:** Se muestran los features restantes  donde se especifica el número del feature, la operación y el valor.
* **Predicciones del stump para una clase:** Se muestran las predicciones iniciales para el stump.
* **Nuevas predicciones para una clase :** Se muestra las nuevas predicciones tomando en cuenta los residuos del stump anterior.
* **Exactitud:** Muestra las etiquetas que han predicho correctamente.
* **Precisión:**  Muestra las etiquetas predichas positivas que son correcta.
* **Recall:** Muestra las etiquetas positivas reales que se han predicho correctamente.
* **F1-score:** Muestra la predicción que tuvo el árbol en la fase de prueba
* **Matriz de confusión: ** Se mostrará la matriz de confusión.
* **Etiquetas predichas: **Se mostrará cuáles fueron las etiquetas que predijo el modelo.
* **Etiquetas reales:** Se mostrará cuáles son las etiquetas reales. Con el fin de poder compararlas con las etiquetas predichas.

####Salida para variables continuas
* **Gamma:** Medida de regularización.
# Linear Regresion

##1) Dataset
**1. california_housing:** Es un dataset tomado de la libreria de Sklearn, el cual como
* **Target:**  MedHouseVal: Valor promedio de las casas en la zona (en dólares).
* **Características:**
1. MedInc: Ingreso medio de los hogares en la zona (en dólares).
2. HouseAge: Promedio de los años de las casas en la zona.
3. AveRooms: Promedio del número de habitaciones por hogar.
4. AveBedrms: Promedio del número de dormitorios por hogar.
4. Population: Población de la zona.
5. AveOccup: Promedio del número de ocupantes por hogar.
6. Latitude: Latitud de la zona.
7. Longitude: Longitud de la zona.

## 2) Ejecución
Para poder ejecutar el script correspondiente, es necesario seguir cada uno de los siguientes pasos:
1. Abrir Anaconda Prompt. 
2. Se debe de ubicar en el directorio Scikitty en la caperta 'demos' utilizando el comando **cd** en la consola de Anaconda Prompt, en donde se encuentran el script correspondiente al dataset. 
3. Una vez ubicado en el directorio de las **'demos'**, se podrá ejecutar  el script. Es necesario colocar los siguientes comando para ejecutarlo:
*   Para poder ejecutar el DTGB de ** california_housing.csv: **
```
python california_housing.py
```

## 2) Salida
* **Generación de imagen:** Se muestra una gráfica, mostrando las predicciones y los valores verdaderos con una tendencia lineal.
* **Mean Squared Error:** Se muestra el error cuadrado medio

#Logistic Regresion
##1) Dataset
**1. breast_cancer:** Es un dataset tomado de Sklearn el cual contiene 
* **Target:** Tipo cancer: Maligno = 0 y Benigno = 1.
* **Características:** Medias: radius,  texture,  perimeter, area,  smoothness,  compactness,  concavity, concave points, symmetry, fractal dimension; erores: radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, fractal dimension; peor: radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, fractal dimension.

## 2) Ejecución
Para poder ejecutar el script correspondiente, es necesario seguir cada uno de los siguientes pasos:
1. Abrir Anaconda Prompt. 
2. Se debe de ubicar en el directorio Scikitty en la caperta 'demos' utilizando el comando **cd** en la consola de Anaconda Prompt, en donde se encuentran el script correspondiente al dataset. 
3. Una vez ubicado en el directorio de las **'demos'**, se podrá ejecutar  el script. Es necesario colocar los siguientes comando para ejecutarlo:
*   Para poder ejecutar el DTGB de ** breast_cancer: **
```
python breast_cancer.py
```

## 2) Salida
* **Exactitud:** Muestra las etiquetas que han predicho correctamente.
* **Precisión:**  Muestra las etiquetas predichas positivas que son correcta.
* **Recall:** Muestra las etiquetas positivas reales que se han predicho correctamente.
* **F1-score:** Muestra la predicción que tuvo el árbol en la fase de prueba
* **Matriz de confusión: ** Se mostrará la matriz de confusión.

## Información Adicional
Scikitty es una libreria de Python que simula el comportamiento de la libreria Scikitlearn. Es un proyecto desarrollado por Axel Monge Ramírez, Andrel Ramírez Solis, John Rojas Chinchilla y Abigail Salas Ramírez. Estudiantes de la Universidad Nacional de Costa Rica, matriculados en el curso de Inteligencia Artificial (EIF420-O) durante el primer semestre del 2024 en la Facultad de Ciencias Exactas. Este proyecto representa sus esfuerzos de colaboración y puede comunicarse con ellos por correo electrónico:  axel.monge.ramirez@est.una.ac.cr, andrel.ramirez.solis@est.una.ac.cr, john.rojas.chinchilla@est.una.ac.cr y abigail.salas.ramirez@est.una.ac.cr.