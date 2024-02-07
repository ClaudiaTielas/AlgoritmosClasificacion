# -*- coding: utf-8 -*-
"""AlgoritmosClasificacion.ipynb


# Predicción de dolencias cardiacas a partir de un electrocardiograma




    Claudia Tielas Sáez, Enero de 2024

[1. Introducción](#seccion-1)

[2. Exploración de datos](#seccion-2)

[3. k-Nearest Neighbour](#seccion-3)

[4. Naive Bayes](#seccion-4)

[5. Artificial Neural Network](#seccion-5)

[6. Support Vector Machine](#seccion-6)

[7. Árbol de Clasificación](#seccion-7)

[8. Random Forest](#seccion-8)

[9. Discusión](#seccion-9)

[10. Referencias](#seccion-10)

##<h2 id="seccion-1">Introducción</h2>

**¿Qué sabemos de nuestros datos?**


Disponemos del resultado de 1200 electrocardiogramas (ECG) en pacientes con alguno de estos 4  problemas cardíacos:
* Arrhythmia (ARR)
* Congestive Heart Failure (CHF)
* Atrial Fibrillation (AFF)
* Normal Sinus Rhythm (NSR).

**¿Cuál es nuestro objetivo?**

El objetivo de este análisis es predecir el tipo de dolencia cardíaca (clase) de los pacientes a partir de la información recogida en el ECG.

Analizaremos los datos mediante la implementación de los diferentes algoritmos estudiados:
*k-Nearest Neighbour, Naive Bayes, Artificial Neural Network, Support Vector Machine, Arbol de Decisión* y
*Random Forest* para predecir el tipo de docencia cardíaca.

---

##<h2 id="seccion-2">Exploración de datos</h2>


El análisis de datos exploratorios o (EDA) consiste en comprender los conjuntos de datos resumiendo sus características principales y, a menudo, representándolos visualmente. Este paso es muy importante especialmente cuando llegamos a modelar los datos para aplicar el aprendizaje automático (Google Colaboratory, s. f.-a).

**Carga de datos**

Importaremos las librerías necesarias para el desarrollo
"""

# Commented out IPython magic to ensure Python compatibility.
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# %matplotlib inline
sns.set(color_codes=True)
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report
from tabulate import tabulate
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier

"""Para obtener o cargar el conjunto de datos en el cuaderno, seleccionamos la carpeta Archivos del panel izquierdo y cargamos nuestro archivo."""

df = pd.read_csv("ECGCvdata.csv") # Leemos el archivo y lo transformamos a dataframe de pandas

df.head(5)

"""Para ver la columna más relevante (ECG_signal) que hace referencia al tipo de dolencia, la mostaremos junto con las últimas variables explicativas numéricas"""

ultimas_columnas = df.iloc[:, -5:]
print(ultimas_columnas)

"""

---

"""

df.shape

"""Nuestro dataframe está constituido por 1200 filas y 56 columnas.

**Duplicados**
"""

duplicate_rows_df = df[df.duplicated()]
print("number of duplicate rows: ", duplicate_rows_df.shape)

"""Ninguna de las filas corresponde a registros duplicados, algo a esperar ya que estos están identificados con valores únicos en la variable RECORD

**Missing values**

Ahora, analizaremos la existencia de missing values en nuestro dataset, y eliminaremos aquellas variables que los contengan
"""

print(df.isnull().sum())

"""Hay 9 variables con valores ausentes, vamos a proceder a eliminarlas:"""

# Con .dropna se suelen eliminar las filas con valores ausentes, pero al indicar
# el parámetro axis=1, eliminamos las columnas con NA's
df_1 = df.dropna(axis=1)

print("Número de filas y columnas antes de eliminar valores perdidos:", df.shape)
print("Número de filas y columnas después de eliminar valores perdidos:", df_1.shape)

"""**Outliers**

La detección y eliminación de outliers (valores atípicos) es un proceso importante en el análisis de datos.

Mostraremos las distribuciones de cada variable con un gráfico de cajas en el que podremos observar si existen valores atípicos
"""

columnas_a_mostrar = df_1.columns[1:-1]  # Excluimos la primera y última columna
plt.figure(figsize=(15, 6))
sns.boxplot(data=df_1[columnas_a_mostrar])
plt.xticks(rotation=45, ha='right')        # etiquetas del eje x legibles
plt.title('Boxplots de las variables numéricas')
plt.show()

"""Las variables RRmean, PPmean, SDRR, IBIM, IBISD, SDSD, RMSSD contienen numerosos valores atípicos

Otras como pNN50 no se aprecian tan bien en conjunto, pero si analizamos su caso:
"""

sns.boxplot(x=df_1['pNN50'])

"""Vemos que también contiene valores atípicos"""

Q1 = df_1.select_dtypes(include=['number']).quantile(0.25)
Q3 = df_1.select_dtypes(include=['number']).quantile(0.75)

IQR = Q3 - Q1

df_2 = df_1[~((df_1 < (Q1 - 1.5 * IQR)) | (df_1 > (Q3 + 1.5 * IQR))).any(axis=1)]

print(df_1.shape)

print(df_2.shape)

"""Este método utiliza el concepto de IQR y elimina filas que caen fuera de 1.5 veces el IQR. En nuestro caso se han eliminado 444 filas correspondientes a outliers"""

columnas_a_mostrar = df_2.columns[1:-1]  # Excluimos la primera y última columna
plt.figure(figsize=(15, 6))
sns.boxplot(data=df_2[columnas_a_mostrar])
plt.xticks(rotation=45, ha='right')        # etiquetas del eje x legibles
plt.title('Distribuciones tras eliminar outliers')
plt.show()

"""La dimensiolalidad del eje y se ha reducido al eliminar los outliers

**Matriz de correlación**
"""

columnas_a_mostrar = df_2.columns[1:-1]  # Excluimos la primera y última columnas
# Calcular la matriz de correlación solo para columnas numéricas
c = df_2[columnas_a_mostrar].select_dtypes(include=['number']).corr()

# Creamos un mapa de calor
sns.heatmap(c, cmap="YlGnBu", annot=False)
plt.title('Matriz de Correlación')
plt.show()

"""**Gráfico de violín: Tipo de Dolencia VS Heart beat per minute**

"""

# Crearemos un gráfico de violín utilizando Seaborn
# x='hbpermin': Variable en el eje x
# y='ECG_signal': Variable en el eje y
# inner='box': Mostraremos un diagrama de caja dentro del violín
# palette='Dark2': Paleta de colores
# hue='ECG_signal': Agrupamos distintos colores por 'ECG_signal'
# legend=False: No mostramos la leyenda en el gráfico
sns.violinplot(data=df, x='hbpermin', y='ECG_signal', inner='box',
               palette='Dark2', hue='ECG_signal', legend=False)
# Dibujar líneas verticales en los valores 60 y 100
plt.axvline(x=60, color='g', linestyle='--')
plt.axvline(x=100, color='g', linestyle='--')

# Quitar información no deseada de los ejes
sns.despine(top=True, right=True, bottom=True, left=True)
# Mostrar el gráfico
plt.show()

"""Una frecuencia cardíaca en reposo normal para los adultos oscila entre 60 y 100 latidos por minuto (Mayo Clinic, 2022). En el caso del tipo de dolencia CHF (Congestive Heart Failure), los datos se sitúan en torno a los valores más altos del rango. Una frecuencia cardíaca alta puede provocar insuficiencias congestivas (Mayo Clinic, 2023)

**Análisis de Componentes Principales**


Aplicaremos el Análisis de Componentes Principales (PCA) para reducir la dimensionalidad de las variables numéricas y estudiaremos la relación entre estas y EGC_sygnal en el contexto de PCA:
"""

# Separamos las variables numéricas y categóricas
df_numericas = df_2.select_dtypes(include=['number'])
df_categoricas = df_2['ECG_signal']

# Estandarizamos las variables numéricas con StandardScaler para que tengan
# media cero y desviación estándar uno.
scaler = StandardScaler()
df_numericas_estandarizadas = scaler.fit_transform(df_numericas)

# Aplicamos PCA solo a las variables numéricaspara reducir su dimensionalidad
pca = PCA(n_components=2)
pca_resultados_numericas = pca.fit_transform(df_numericas_estandarizadas)

# LabelEncoder transforma las etiquetas categóricas en números
le = LabelEncoder()
df_categoricas_numericas = pd.DataFrame({'ECG_signal': le.fit_transform(df_categoricas)})

# Combinamos los resultados
df_final = pd.DataFrame(data=pca_resultados_numericas, columns=['Componente 1', 'Componente 2'])
df_final['ECG_signal'] = df_categoricas_numericas['ECG_signal']

# Mapeamos los nombres originales de las clases
clases_originales = dict(zip(le.transform(le.classes_), le.classes_))
df_final['Nombre Clase'] = df_final['ECG_signal'].map(clases_originales)

# Mostramos un gráfico de dispersión
sns.scatterplot(x='Componente 1', y='Componente 2', hue='Nombre Clase', data=df_final, palette='Dark2')
plt.title('Visualización PCA con ECG_signal')
plt.legend(title='Clase')
plt.show()

"""Se observan zonas con patrones de separación por clases más marcados (ARR y NSR), y otras zonas donde las clases convergen (AFF y CHF), que serán aquellas más problemáticas para la tarea de clasificación. Parece que NSR (normal Sinus Rhythm) está claramente diferenciada. Esta clase hace referencia a los casos sin dolencia cardíaca, por lo que es positivo que se diferencie claramente de los casos que si presentan patologías.

Por último dentro de esta sección, vamos a separar los datos en los conjuntos de train (66%) y test (33%) que emplearemos en la construcción de los algoritmos de clasificación
"""

df_2 = df_2.iloc[:, 1:]  # Eliminamos la primera columna ya que es identificativa

# Separamos las características (X) de la variable objeto (Y) que deseamos precedir
X = df_2.iloc[:, :-1]  # Todas las columnas excepto la última
y = df_2.iloc[:, -1]   # Solo la última columna

print(X.shape)
print(y.shape)

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=12345)

# Ahora, X_train y y_train son nuestros datos de entrenamiento
# Y X_test y y_test son nuestros datos de prueba

"""Nos aseguramos de que "X" contiene todas las filas y una columna menos (hemos eliminado RECORD).

"y" contiene únicamente la variable a predecir con todas sus filas

---

**Algoritmos de clasificación**

Los algoritmos de clasificación son técnicas en aprendizaje automático diseñadas para asignar categorías o etiquetas a datos no etiquetados. El objetivo principal es entrenar modelos que pueda generalizar patrones a partir de un conjunto de datos de entrenamiento y aplicar ese conocimiento para clasificar datos nuevos o no vistos.

Para evaluar el rendimiento de nuestros modelos, emplearemos métricas comúnmente utilizadas en problemas de clasificación:
1. Accuracy: mide la proporción de instancias clasificadas correctamente (tanto positivas como negativas) entre todas las instancias.

2. Precisión: mide la proporción de instancias positivas predichas correctamente entre todas las instancias clasificadas como positivas. Es útil cuando se quieren minimizar los falsos positivos.


3. Recall: mide la proporción de instancias positivas predichas correctamente entre todas las instancias que son verdaderamente positivas. Es útil cuando se quieren minimizar los falsos negativos.


4. F1-score: es la media armónica de precisión y recall. Proporciona un equilibrio entre ambas métricas y es útil cuando hay un desequilibrio entre las clases o cuando tanto los falsos positivos como los falsos negativos son críticos.

##<h2 id="seccion-3">k-Nearest Neighbour</h2>

Este algoritmo clasifica un punto de datos asignándole la etiqueta de la mayoría de sus k vecinos más cercanos en el espacio de características con el objetivo de identificar patrones.

Iteraremos sobre los valores de k y entrenaremos un modelo k-NN para cada uno
"""

k_values = [1, 3, 5, 7, 11]
# Listado de resultados para k-NN
results_kNN = []

# Iteramos por cada valor de k
for k in k_values:
    # Generamos el modelo
    knn_model = KNeighborsClassifier(n_neighbors=k)
    # Entrenamos el modelo
    knn_model.fit(X_train, y_train)

    # Predicciones para el conjunto de prueba
    y_pred = knn_model.predict(X_test)

    # Calculamos las métricas de rendimiento
    accuracy_kNN = accuracy_score(y_test, y_pred)
    precision_kNN = precision_score(y_test, y_pred, average='weighted')
    recall_kNN = recall_score(y_test, y_pred, average='weighted')
    f1_kNN = f1_score(y_test, y_pred, average='weighted')

    results_kNN.append([k, accuracy_kNN, precision_kNN, recall_kNN, f1_kNN])

"""Mostraremos una tabla con los resultados en forma de métricas para los distintos valores de k"""

columns = ['k', 'Accuracy', 'Precision', 'Recall', 'F1-score']
results_df_kNN = pd.DataFrame(results_kNN, columns=columns)
print(results_df_kNN)

# Definir variables para almacenar k_mejor
mejor_k = None
# Almacenamos el índice del máximo para cada métrica
indice_max_accuracy = results_df_kNN['Accuracy'].idxmax()
indice_max_precision = results_df_kNN['Precision'].idxmax()
indice_max_recall = results_df_kNN['Recall'].idxmax()
indice_max_f1 = results_df_kNN['F1-score'].idxmax()

# Localizamos el valor de k correspondiente a cada métrica
k_mejor_accuracy = results_df_kNN.loc[indice_max_accuracy, 'k']
k_mejor_precision = results_df_kNN.loc[indice_max_precision, 'k']
k_mejor_recall = results_df_kNN.loc[indice_max_recall, 'k']
k_mejor_f1 = results_df_kNN.loc[indice_max_f1, 'k']

# Definiremos una función para que seleccione y muetre el mejor k
def seleccionar_mejor_k():
    global mejor_k  # Indica que estamos utilizando la variable global
    # Seleccionamos el valor de k con mejor rendimiento entre las métricas
    k_mejor = max(k_mejor_accuracy, k_mejor_precision, k_mejor_recall, k_mejor_f1)
    mejor_k = k_mejor
    # Filtramos los resultados con el mejor k valor
    df_mejor_k = results_df_kNN[results_df_kNN['k'] == k_mejor]

    # Retorna el valor de k_mejor
    return df_mejor_k

seleccionar_mejor_k()

"""Parece que k = 1 ofrece los mejores resultados en las métricas."""

# Creamos y entrenamos el modelo k-NN con el valor óptimo de k
modelo_knn = KNeighborsClassifier(n_neighbors=mejor_k)
modelo_knn.fit(X_train, y_train)

# Realizamos predicciones en el conjunto de prueba
y_pred = modelo_knn.predict(X_test)

# Calculamos la matriz de confusión
conf_matrix = confusion_matrix(y_test, y_pred, labels=["ARR", "AFF", "CHF", "NSR"])

# Obtenemos las etiquetas de las clases
clases = y.unique()

# Creamos un DataFrame de la matriz de confusión
conf_matrix_df = pd.DataFrame(conf_matrix, index=clases, columns=clases)
# Mapa de calor
sns.heatmap(conf_matrix_df, annot=True, cmap="YlGnBu", fmt="d", linewidths=.5, cbar=False)

# Configuramos las etiquetas y el título
plt.xlabel("Predicción")
plt.ylabel("Realidad")
plt.title("Matriz de Confusión")

# Mostramos el gráfico
plt.show()

# Extraemos los valores de la matriz de confusión
TP = conf_matrix.diagonal()              # True positives
FP = conf_matrix.sum(axis=0) - TP        # False positives
FN = conf_matrix.sum(axis=1) - TP        # False negatives
TN = conf_matrix.sum() - (TP + FP + FN)  # True negatives

# Creamos un diccionario
resultados_dict = {
    'Clase': clases,
    'Verdaderos Positivos (TP)': TP,
    'Falsos Positivos (FP)': FP,
    'Falsos Negativos (FN)': FN,
    'Verdaderos Negativos (TN)': TN
}

df_resultados = pd.DataFrame(resultados_dict)

tabla_resultados = tabulate(df_resultados, headers='keys', tablefmt='pretty', showindex=False)
print(tabla_resultados)

# Imprimos un informe de clasificación
print("\nInforme de Clasificación:")
print(classification_report(y_test, y_pred))

# Almacenamos las métricas
accuracy_knn = accuracy_score(y_test, y_pred)
precision_knn = precision_score(y_test, y_pred, average='weighted')
recall_knn = recall_score(y_test, y_pred, average='weighted')
f1_knn = f1_score(y_test, y_pred, average='weighted')

"""Los resultados sugieren que el modelo es capaz de realizar predicciones precisas y tiene un buen equilibrio entre precisión y recall para todas las clases.

---

##<h2 id="seccion-4">Naive Bayes</h2>

En el clasificador Naive Bayes, se calculan las probabilidades condicionales de cada característica dado una clase. Si una característica específica no está presente en el conjunto de entrenamiento para una clase particular, la probabilidad condicional será cero, lo cual puede conducir a problemas durante la clasificación.

Para evitar este problema, se utiliza el suavizado de Laplace al agregar un valor constante (generalmente 1) a todas las frecuencias observadas. Esto garantiza que incluso si una característica no se observa en el conjunto de entrenamiento para una clase, todavía tendrá una probabilidad condicional positiva después del suavizado.

En nuestro caso, estamos tratando con variables numéricas continuas, por lo que emplearemos Naive Bayes Gaussiano:
"""

# Modelo GaussianNB
gnb = GaussianNB()
# Entrenamos el modelo
gnb.fit(X_train, y_train)
# Realizamos las predicciones para los datos de test
y_pred_gnb = gnb.predict(X_test)
# Calculamos la matriz de confusión
conf_matrix = confusion_matrix(y_test, y_pred_gnb, labels=["ARR", "AFF", "CHF", "NSR"])

# Obtenemos las etiquetas de las clases
clases = y.unique()

# Creamos un DataFrame de la matriz de confusión
conf_matrix_df = pd.DataFrame(conf_matrix, index=clases, columns=clases)
# Mapa de calor
sns.heatmap(conf_matrix_df, annot=True, cmap="YlGnBu", fmt="d", linewidths=.5, cbar=False)

# Configuramos las etiquetas y el título
plt.xlabel("Predicción")
plt.ylabel("Realidad")
plt.title("Matriz de Confusión")

# Mostramos el gráfico
plt.show()

# Extraemos los valores de la matriz de confusión
TP = conf_matrix.diagonal()              # True positives
FP = conf_matrix.sum(axis=0) - TP        # False positives
FN = conf_matrix.sum(axis=1) - TP        # False negatives
TN = conf_matrix.sum() - (TP + FP + FN)  # True negatives

# Creamos un diccionario
resultados_dict = {
    'Clase': clases,
    'Verdaderos Positivos (TP)': TP,
    'Falsos Positivos (FP)': FP,
    'Falsos Negativos (FN)': FN,
    'Verdaderos Negativos (TN)': TN
}

df_resultados = pd.DataFrame(resultados_dict)

tabla_resultados = tabulate(df_resultados, headers='keys', tablefmt='pretty', showindex=False)
print(tabla_resultados)

# Imprimimos un informe de clasificación
print("\nInforme de Clasificación:")
print(classification_report(y_test, y_pred_gnb))

# Almacenamos las métricas
accuracy_gnb = accuracy_score(y_test, y_pred_gnb)
precision_gnb = precision_score(y_test, y_pred_gnb, average='weighted')
recall_gnb = recall_score(y_test, y_pred_gnb, average='weighted')
f1_gnb = f1_score(y_test, y_pred_gnb, average='weighted')

"""Resultados:
- **ARR:** Buen rendimiento con una alta tasa de verdaderos positivos (43) y solo 1 falso positivo.
- **AFF:** Aunque tiene un alto número de verdaderos positivos (67), también muestra una cantidad significativa de falsos positivos (22).
- **CHF:** Similar a AFF, con un número considerable de falsos positivos (11).
- **NSR:** Excelente rendimiento con 53 verdaderos positivos y cero falsos positivos.

AFF y CHF presentan métricas de precisión, recall y f1-score relativamente bajas. La precisión general del modelo es del 86%, lo cual es bastante bueno. Como pudimos constatar en el Análisis de Componentes Principales de 2 componentes, las clases AFF y CHF eran las que presentaban una frontera más difusa, y por lo tanto las que más complicaciones tendrían a la hora de la clasificación.


NSR parece tener un rendimiento excepcional con cero falsos positivos, pero también debemos tener en cuenta la proporción de muestras NSR en el conjunto de datos. Si la clase NSR está sobrerrepresentada, el modelo podría sesgarse hacia esta clase.

Comprobaremos si existen clases sobrerepresentadas en nuestro conjunto:

"""

conteo_clases = y_train.value_counts()
tabla_frecuencia = pd.DataFrame({'Clase': conteo_clases.index, 'Frecuencia': conteo_clases.values})
# Mostrar la tabla de frecuencia
print(tabla_frecuencia)

"""Como podemos ver, NSR no está sobrerepresentada, de hecho contamos con mayores frecuencias en AFF y CHF

---

##<h2 id="seccion-5">Artificial Neural Network (ANN)</h2>


Este algoritmo utiliza una red de nodos (neuronas) interconectados, con capas de entrada, ocultas y de salida. Aprenden ponderaciones que minimizan la diferencia entre las predicciones y las etiquetas reales para capturar patrones complejos y no lineales en los datos.

- Red Neuronal con una Capa Oculta (15 nodos):
"""

# Configuración de la Red Neuronal con una capa oculta y 15 nodos
ann_1capa = MLPClassifier(hidden_layer_sizes=(15,), max_iter=500, random_state=12345)

# Entrenamiento del modelo
ann_1capa.fit(X_train, y_train)

# Predicciones
y_pred_1capa = ann_1capa.predict(X_test)

# Calculamos la matriz de confusión
conf_matrix = confusion_matrix(y_test, y_pred_1capa, labels=["ARR", "AFF", "CHF", "NSR"])
# Creamos un DataFrame de la matriz de confusión
conf_matrix_df = pd.DataFrame(conf_matrix, index=clases, columns=clases)
# Mapa de calor
sns.heatmap(conf_matrix_df, annot=True, cmap="YlGnBu", fmt="d", linewidths=.5, cbar=False)
# Configuramos las etiquetas y el título
plt.xlabel("Predicción")
plt.ylabel("Realidad")
plt.title("Matriz de Confusión")
# Mostramos el gráfico
plt.show()

# Extraemos los valores de la matriz de confusión
TP = conf_matrix.diagonal()              # True positives
FP = conf_matrix.sum(axis=0) - TP        # False positives
FN = conf_matrix.sum(axis=1) - TP        # False negatives
TN = conf_matrix.sum() - (TP + FP + FN)  # True negatives

# Creamos un diccionario
resultados_dict = {
    'Clase': clases,
    'Verdaderos Positivos (TP)': TP,
    'Falsos Positivos (FP)': FP,
    'Falsos Negativos (FN)': FN,
    'Verdaderos Negativos (TN)': TN
}

df_resultados = pd.DataFrame(resultados_dict)

tabla_resultados = tabulate(df_resultados, headers='keys', tablefmt='pretty', showindex=False)
print(tabla_resultados)

# Imprimimos un informe de clasificación
print("\nInforme de Clasificación:")
print(classification_report(y_test, y_pred_1capa))

# Almacenamos las métricas
accuracy_1capa = accuracy_score(y_test, y_pred_1capa)
precision_1capa = precision_score(y_test, y_pred_1capa, average='weighted')
recall_1capa = recall_score(y_test, y_pred_1capa, average='weighted')
f1_1capa = f1_score(y_test, y_pred_1capa, average='weighted')

"""- Red Neuronal con Dos Capas Ocultas (25 y 10 nodos):"""

# Configuración de la Red Neuronal con dos capas ocultas y 25 y 10 nodos respectivamente
ann_2capas = MLPClassifier(hidden_layer_sizes=(25, 10), max_iter=500, random_state=12345)

# Entrenamiento del modelo
ann_2capas.fit(X_train, y_train)

# Predicciones
y_pred_2capas = ann_2capas.predict(X_test)

# Calculamos la matriz de confusión
conf_matrix = confusion_matrix(y_test, y_pred_2capas, labels=["ARR", "AFF", "CHF", "NSR"])
# Creamos un DataFrame de la matriz de confusión
conf_matrix_df = pd.DataFrame(conf_matrix, index=clases, columns=clases)
# Mapa de calor
sns.heatmap(conf_matrix_df, annot=True, cmap="YlGnBu", fmt="d", linewidths=.5, cbar=False)
# Configuramos las etiquetas y el título
plt.xlabel("Predicción")
plt.ylabel("Realidad")
plt.title("Matriz de Confusión")
# Mostramos el gráfico
plt.show()

# Extraemos los valores de la matriz de confusión
TP = conf_matrix.diagonal()              # True positives
FP = conf_matrix.sum(axis=0) - TP        # False positives
FN = conf_matrix.sum(axis=1) - TP        # False negatives
TN = conf_matrix.sum() - (TP + FP + FN)  # True negatives

# Creamos un diccionario
resultados_dict = {
    'Clase': clases,
    'Verdaderos Positivos (TP)': TP,
    'Falsos Positivos (FP)': FP,
    'Falsos Negativos (FN)': FN,
    'Verdaderos Negativos (TN)': TN
}

df_resultados = pd.DataFrame(resultados_dict)

tabla_resultados = tabulate(df_resultados, headers='keys', tablefmt='pretty', showindex=False)
print(tabla_resultados)

# Imprimimos un informe de clasificación
print("\nInforme de Clasificación:")
print(classification_report(y_test, y_pred_2capas))

# Almacenamos las métricas
accuracy_2capas = accuracy_score(y_test, y_pred_2capas)
precision_2capas = precision_score(y_test, y_pred_2capas, average='weighted')
recall_2capas = recall_score(y_test, y_pred_2capas, average='weighted')
f1_2capas = f1_score(y_test, y_pred_2capas, average='weighted')

"""- La arquitectura con una capa oculta muestra un desempeño general sólido, con buenos resultados en precisión y sensibilidad para la mayoría de las clases.

- La arquitectura con dos capas ocultas muestra un rendimiento inferior. La sensibilidad para la clase AFF es muy baja (4%), lo que indica dificultades para identificar correctamente esta clase. La precisión para CHF es relativamente baja (48%).

A pesar de que la arquitectura con dos capas ocultas alcanza una precisión del 70%, este resultado puede estar sesgado por la alta precisión en las clases con pocos ejemplos (como ARR y NSR). La baja sensibilidad en la clase AFF indica que la red tiene dificultades para detectar positivos verdaderos en esta categoría.

La arquitectura con una capa oculta parece ser más equilibrada y generaliza mejor en comparación con la arquitectura con dos capas ocultas en este caso

---

##<h2 id="seccion-6">Support Vector Machine (SVM)</h2>

Este algoritmo busca el hiperplano que mejor separa las clases en el espacio de características. Puede emplear diferentes funciones kernel para manejar datos no lineales.

- SVM con Kernel Lineal
"""

# Crear y entrenar el modelo SVM con kernel lineal
svm_linear = SVC(kernel='linear')
svm_linear.fit(X_train, y_train)

# Predecir en el conjunto de prueba
y_pred_linear = svm_linear.predict(X_test)

# Calculamos la matriz de confusión
conf_matrix = confusion_matrix(y_test, y_pred_linear, labels=["ARR", "AFF", "CHF", "NSR"])
# Creamos un DataFrame de la matriz de confusión
conf_matrix_df = pd.DataFrame(conf_matrix, index=clases, columns=clases)
# Mapa de calor
sns.heatmap(conf_matrix_df, annot=True, cmap="YlGnBu", fmt="d", linewidths=.5, cbar=False)
# Configuramos las etiquetas y el título
plt.xlabel("Predicción")
plt.ylabel("Realidad")
plt.title("Matriz de Confusión - Kernel Lineal")
# Mostramos el gráfico
plt.show()

# Extraemos los valores de la matriz de confusión
TP = conf_matrix.diagonal()              # True positives
FP = conf_matrix.sum(axis=0) - TP        # False positives
FN = conf_matrix.sum(axis=1) - TP        # False negatives
TN = conf_matrix.sum() - (TP + FP + FN)  # True negatives

# Creamos un diccionario
resultados_dict = {
    'Clase': clases,
    'Verdaderos Positivos (TP)': TP,
    'Falsos Positivos (FP)': FP,
    'Falsos Negativos (FN)': FN,
    'Verdaderos Negativos (TN)': TN
}

df_resultados = pd.DataFrame(resultados_dict)

tabla_resultados = tabulate(df_resultados, headers='keys', tablefmt='pretty', showindex=False)
print(tabla_resultados)

# Imprimimos un informe de clasificación
print("\nInforme de Clasificación:")
print(classification_report(y_test, y_pred_linear))

# Almacenamos las métricas
accuracy_linear = accuracy_score(y_test, y_pred_linear)
precision_linear = precision_score(y_test, y_pred_linear, average='weighted')
recall_linear = recall_score(y_test, y_pred_linear, average='weighted')
f1_linear = f1_score(y_test, y_pred_linear, average='weighted')

"""-  SVM con Kernel RBF"""

# Crear y entrenar el modelo SVM con kernel RBF
svm_rbf = SVC(kernel='rbf', gamma='scale')  # gamma='scale' utiliza 1 / (n_features * X.var()) como valor de gamma
svm_rbf.fit(X_train, y_train)

# Predecir en el conjunto de prueba
y_pred_rbf = svm_rbf.predict(X_test)

# Calculamos la matriz de confusión
conf_matrix = confusion_matrix(y_test, y_pred_rbf, labels=["ARR", "AFF", "CHF", "NSR"])
# Creamos un DataFrame de la matriz de confusión
conf_matrix_df = pd.DataFrame(conf_matrix, index=clases, columns=clases)
# Mapa de calor
sns.heatmap(conf_matrix_df, annot=True, cmap="YlGnBu", fmt="d", linewidths=.5, cbar=False)
# Configuramos las etiquetas y el título
plt.xlabel("Predicción")
plt.ylabel("Realidad")
plt.title("Matriz de Confusión - SVM con Kernel RBF")
# Mostramos el gráfico
plt.show()

# Extraemos los valores de la matriz de confusión
TP = conf_matrix.diagonal()              # True positives
FP = conf_matrix.sum(axis=0) - TP        # False positives
FN = conf_matrix.sum(axis=1) - TP        # False negatives
TN = conf_matrix.sum() - (TP + FP + FN)  # True negatives

# Creamos un diccionario
resultados_dict = {
    'Clase': clases,
    'Verdaderos Positivos (TP)': TP,
    'Falsos Positivos (FP)': FP,
    'Falsos Negativos (FN)': FN,
    'Verdaderos Negativos (TN)': TN
}

df_resultados = pd.DataFrame(resultados_dict)

tabla_resultados = tabulate(df_resultados, headers='keys', tablefmt='pretty', showindex=False)
print(tabla_resultados)

# Imprimimos un informe de clasificación
print("\nInforme de Clasificación:")
print(classification_report(y_test, y_pred_rbf))

# Almacenamos las métricas
accuracy_rbf = accuracy_score(y_test, y_pred_rbf)
precision_rbf = precision_score(y_test, y_pred_rbf, average='weighted')
recall_rbf = recall_score(y_test, y_pred_rbf, average='weighted')
f1_rbf = f1_score(y_test, y_pred_rbf, average='weighted')

"""Ambos modelos muestran un rendimiento bastante sólido en términos de precisión general. El modelo SVM con kernel lineal tiene una precisión ligeramente superior (93%) en comparación con el modelo SVM con kernel RBF (91%).

Ambos tienen un rendimiento excelente en la clasificación de ARR y NSR, pero el kernel lineal destaca en la clasificación de AFF frente al kernel rbf.

---

##<h2 id="seccion-7">Árbol de Clasificación</h2>

Este algoritmo divide el espacio de características en regiones mediante decisiones basadas en características particulares. Crea una estructura de árbol para clasificar los datos.
"""

# Crear y entrenar el modelo del árbol de clasificación
tree_classifier = DecisionTreeClassifier(random_state=12345)
tree_classifier.fit(X_train, y_train)

# Realizar predicciones en el conjunto de prueba
y_pred_tree = tree_classifier.predict(X_test)

# Calculamos la matriz de confusión
conf_matrix = confusion_matrix(y_test, y_pred_tree, labels=["ARR", "AFF", "CHF", "NSR"])
# Creamos un DataFrame de la matriz de confusión
conf_matrix_df = pd.DataFrame(conf_matrix, index=clases, columns=clases)
# Mapa de calor
sns.heatmap(conf_matrix_df, annot=True, cmap="YlGnBu", fmt="d", linewidths=.5, cbar=False)
# Configuramos las etiquetas y el título
plt.xlabel("Predicción")
plt.ylabel("Realidad")
plt.title("Matriz de Confusión - Árbol de Decisión")
# Mostramos el gráfico
plt.show()

# Extraemos los valores de la matriz de confusión
TP = conf_matrix.diagonal()              # True positives
FP = conf_matrix.sum(axis=0) - TP        # False positives
FN = conf_matrix.sum(axis=1) - TP        # False negatives
TN = conf_matrix.sum() - (TP + FP + FN)  # True negatives

# Creamos un diccionario
resultados_dict = {
    'Clase': clases,
    'Verdaderos Positivos (TP)': TP,
    'Falsos Positivos (FP)': FP,
    'Falsos Negativos (FN)': FN,
    'Verdaderos Negativos (TN)': TN
}

df_resultados = pd.DataFrame(resultados_dict)

tabla_resultados = tabulate(df_resultados, headers='keys', tablefmt='pretty', showindex=False)
print(tabla_resultados)

# Imprimimos un informe de clasificación
print("\nInforme de Clasificación:")
print(classification_report(y_test, y_pred_tree))

# Almacenamos las métricas
accuracy_tree = accuracy_score(y_test, y_pred_tree)
precision_tree = precision_score(y_test, y_pred_tree, average='weighted')
recall_tree = recall_score(y_test, y_pred_tree, average='weighted')
f1_tree = f1_score(y_test, y_pred_tree, average='weighted')

"""Los resultados actuales muestran un rendimiento bastante sólido con un alto nivel de precisión, recall y f1-score para todas las clases.

Sin embargo, vamos a aplicar *boosting*, una técnica de ensamblado que combina múltiples modelos más débiles para formar un modelo fuerte. Funciona construyendo iterativamente modelos débiles, asignando más peso a las instancias clasificadas incorrectamente en iteraciones anteriores, lo que permite mejorar el rendimiento general del modelo.

Para aplicar boosting a un árbol de clasificación en scikit-learn, emplearemos el clasificador `AdaBoostClassifier`. AdaBoost construirá una secuencia de clasificadores débiles (en este caso, árboles de decisión) que se combinan para formar un clasificador fuerte.

El parámetro `n_estimators` indica el número de árboles de decisión que se construirán durante el proceso.
"""

# Crear un árbol de decisión como clasificador base
tree_classifier = DecisionTreeClassifier(random_state=12345)

# Crear el clasificador AdaBoost utilizando el árbol de decisión como base
adaboost_classifier = AdaBoostClassifier(estimator=tree_classifier, n_estimators=50, random_state=12345)

# Entrenar el modelo
adaboost_classifier.fit(X_train, y_train)

# Hacer predicciones en el conjunto de prueba
y_pred_adaboost = adaboost_classifier.predict(X_test)

# Calculamos la matriz de confusión
conf_matrix = confusion_matrix(y_test, y_pred_adaboost, labels=["ARR", "AFF", "CHF", "NSR"])
# Creamos un DataFrame de la matriz de confusión
conf_matrix_df = pd.DataFrame(conf_matrix, index=clases, columns=clases)
# Mapa de calor
sns.heatmap(conf_matrix_df, annot=True, cmap="YlGnBu", fmt="d", linewidths=.5, cbar=False)
# Configuramos las etiquetas y el título
plt.xlabel("Predicción")
plt.ylabel("Realidad")
plt.title("Matriz de Confusión - Árbol de decisión con boosting")
# Mostramos el gráfico
plt.show()

# Extraemos los valores de la matriz de confusión
TP = conf_matrix.diagonal()              # True positives
FP = conf_matrix.sum(axis=0) - TP        # False positives
FN = conf_matrix.sum(axis=1) - TP        # False negatives
TN = conf_matrix.sum() - (TP + FP + FN)  # True negatives

# Creamos un diccionario
resultados_dict = {
    'Clase': clases,
    'Verdaderos Positivos (TP)': TP,
    'Falsos Positivos (FP)': FP,
    'Falsos Negativos (FN)': FN,
    'Verdaderos Negativos (TN)': TN
}

df_resultados = pd.DataFrame(resultados_dict)

tabla_resultados = tabulate(df_resultados, headers='keys', tablefmt='pretty', showindex=False)
print(tabla_resultados)

# Imprimimos un informe de clasificación
print("\nInforme de Clasificación:")
print(classification_report(y_test, y_pred_adaboost))

# Almacenamos las métricas
accuracy_adaboost = accuracy_score(y_test, y_pred_adaboost)
precision_adaboost = precision_score(y_test, y_pred_adaboost, average='weighted')
recall_adaboost = recall_score(y_test, y_pred_adaboost, average='weighted')
f1_adaboost = f1_score(y_test, y_pred_adaboost, average='weighted')

"""Los resultados muestran que tanto con boosting como sin él, el modelo de árbol de clasificación ha logrado un rendimiento bastante alto en términos de precisión, recall y f1-score en todas las clases.

**Sin Boosting:**
- La precisión general del modelo sin boosting es alta, alcanzando el 96% de precisión.
- El modelo logra identificar correctamente todas las instancias de la clase "ARR", obteniendo un recall del 100% para esta clase.
- Las clases "AFF" y "CHF" también tienen buenos resultados, aunque con algunos falsos positivos y falsos negativos.

**Con Boosting:**
- La precisión general con boosting es del 95%, ligeramente inferior a la versión sin boosting, pero sigue siendo muy buena.
- La clase "ARR" nuevamente tiene un rendimiento perfecto con un recall del 100%.
- Las clases "AFF" y "CHF" también presentan resultados sólidos, aunque con algunos cambios en los falsos positivos y falsos negativos en comparación con la versión sin boosting.

En resumen, ambos modelos han demostrado ser efectivos, pero es importante tener en cuenta que la adición de boosting no siempre resulta en mejoras significativas. En este caso, los resultados son bastante similares entre el modelo con y sin boosting, lo que sugiere que el modelo base (árbol de decisión) ya estaba bien ajustado a los datos. Es posible que en conjuntos de datos más complejos o ruidosos, boosting proporcione mejoras más notables.

---

##<h2 id="seccion-8">Random Forest</h2>

El algoritmo se basa en un conjunto de árboles de decisión entrenados en subconjuntos aleatorios del conjunto de datos. Combina las predicciones de múltilpes árboles para mejorar la generalización y reducir el sobreajuste comparado con un solo árbol.
"""

# Configuración 1: Random Forest con 100 árboles
rf_100 = RandomForestClassifier(n_estimators=100, random_state=12345)
rf_100.fit(X_train, y_train)

# Predicciones en el conjunto de prueba
y_pred_rf_100 = rf_100.predict(X_test)

# Calculamos la matriz de confusión
conf_matrix = confusion_matrix(y_test, y_pred_rf_100, labels=["ARR", "AFF", "CHF", "NSR"])
# Creamos un DataFrame de la matriz de confusión
conf_matrix_df = pd.DataFrame(conf_matrix, index=clases, columns=clases)
# Mapa de calor
sns.heatmap(conf_matrix_df, annot=True, cmap="YlGnBu", fmt="d", linewidths=.5, cbar=False)
# Configuramos las etiquetas y el título
plt.xlabel("Predicción")
plt.ylabel("Realidad")
plt.title("Matriz de Confusión - Random forest n = 100")
# Mostramos el gráfico
plt.show()

# Extraemos los valores de la matriz de confusión
TP = conf_matrix.diagonal()              # True positives
FP = conf_matrix.sum(axis=0) - TP        # False positives
FN = conf_matrix.sum(axis=1) - TP        # False negatives
TN = conf_matrix.sum() - (TP + FP + FN)  # True negatives

# Creamos un diccionario
resultados_dict = {
    'Clase': clases,
    'Verdaderos Positivos (TP)': TP,
    'Falsos Positivos (FP)': FP,
    'Falsos Negativos (FN)': FN,
    'Verdaderos Negativos (TN)': TN
}

df_resultados = pd.DataFrame(resultados_dict)

tabla_resultados = tabulate(df_resultados, headers='keys', tablefmt='pretty', showindex=False)
print(tabla_resultados)

# Imprimimos un informe de clasificación
print("\nInforme de Clasificación:")
print(classification_report(y_test, y_pred_rf_100))

# Almacenamos las métricas
accuracy_rf_100 = accuracy_score(y_test, y_pred_rf_100)
precision_rf_100 = precision_score(y_test, y_pred_rf_100, average='weighted')
recall_rf_100 = recall_score(y_test, y_pred_rf_100, average='weighted')
f1_rf_100 = f1_score(y_test, y_pred_rf_100, average='weighted')

# Configuración 2: Random Forest con 200 árboles
rf_200 = RandomForestClassifier(n_estimators=200, random_state=12345)
rf_200.fit(X_train, y_train)

# Predicciones en el conjunto de prueba
y_pred_rf_200 = rf_200.predict(X_test)

# Calculamos la matriz de confusión
conf_matrix = confusion_matrix(y_test, y_pred_rf_200, labels=["ARR", "AFF", "CHF", "NSR"])
# Creamos un DataFrame de la matriz de confusión
conf_matrix_df = pd.DataFrame(conf_matrix, index=clases, columns=clases)
# Mapa de calor
sns.heatmap(conf_matrix_df, annot=True, cmap="YlGnBu", fmt="d", linewidths=.5, cbar=False)
# Configuramos las etiquetas y el título
plt.xlabel("Predicción")
plt.ylabel("Realidad")
plt.title("Matriz de Confusión - Random forest n = 200")
# Mostramos el gráfico
plt.show()

# Extraemos los valores de la matriz de confusión
TP = conf_matrix.diagonal()              # True positives
FP = conf_matrix.sum(axis=0) - TP        # False positives
FN = conf_matrix.sum(axis=1) - TP        # False negatives
TN = conf_matrix.sum() - (TP + FP + FN)  # True negatives

# Creamos un diccionario
resultados_dict = {
    'Clase': clases,
    'Verdaderos Positivos (TP)': TP,
    'Falsos Positivos (FP)': FP,
    'Falsos Negativos (FN)': FN,
    'Verdaderos Negativos (TN)': TN
}

df_resultados = pd.DataFrame(resultados_dict)

tabla_resultados = tabulate(df_resultados, headers='keys', tablefmt='pretty', showindex=False)
print(tabla_resultados)

# Imprimimos un informe de clasificación
print("\nInforme de Clasificación:")
print(classification_report(y_test, y_pred_rf_200))

# Almacenamos las métricas
accuracy_rf_200 = accuracy_score(y_test, y_pred_rf_200)
precision_rf_200 = precision_score(y_test, y_pred_rf_200, average='weighted')
recall_rf_200 = recall_score(y_test, y_pred_rf_200, average='weighted')
f1_rf_200 = f1_score(y_test, y_pred_rf_200, average='weighted')

"""Los resultados de los modelos Random Forest con 100 y 200 árboles son muy similares y muestran un rendimiento sobresaliente en la clasificación de las clases.

**Random Forest con 100 árboles:**
- La precisión global (accuracy) es del 97%, lo que indica que el modelo clasifica correctamente el 97% de las instancias en el conjunto de prueba.
- Para cada clase (ARR, AFF, CHF, NSR), la precisión, recall y f1-score son muy altos, todos por encima del 92%. Esto sugiere que el modelo es capaz de clasificar con precisión las instancias de cada clase.
- La clase ARR tiene un f1-score de 1.00, lo que significa que no hubo falsos negativos ni falsos positivos para esta clase.

**Random Forest con 200 árboles:**
- Los resultados son prácticamente idénticos a los del modelo con 100 árboles, lo cual es consistente con el hecho de que incrementar el número de árboles no tuvo un impacto significativo en este conjunto de datos.
- La precisión global (accuracy) es del 97%, y las métricas por clase son excelentes.

No hay una diferencia significativa al aumentar el número de árboles de 100 a 200 en este caso particular.En este escenario, ambos modelos son buenos candidatos y proporcionan resultados confiables.

##<h2 id="seccion-9">Discusión</h2>
"""

# Creamos un DataFrame para mostrar un resumen con las 4 métricas para todos los
# algoritmos implementados en la tarea de clasificación
df_resultados = pd.DataFrame({
    'k-Nearest Neighbour': [accuracy_knn, precision_knn, recall_knn, f1_knn],
    'Naive Bayes': [accuracy_gnb, precision_gnb, recall_gnb, f1_gnb],
    'ANN 1 capa': [accuracy_1capa, precision_1capa, recall_1capa, f1_1capa],
    'ANN 2 capas': [accuracy_2capas, precision_2capas, recall_2capas, f1_2capas],
    'SVM linear': [accuracy_linear, precision_linear, recall_linear, f1_linear],
    'SVM rbf': [accuracy_rbf, precision_rbf, recall_rbf, f1_rbf],
    'Árbol Clasificación': [accuracy_tree, precision_tree, recall_tree, f1_tree],
    'Árbol con boosting': [accuracy_adaboost, precision_adaboost, recall_adaboost, f1_adaboost],
    'Random forest 100': [accuracy_rf_100, precision_rf_100, recall_rf_100, f1_rf_100],
    'Random Forest 200': [accuracy_rf_200, precision_rf_200, recall_rf_200, f1_rf_200]
}, index=['Accuracy', 'Precision', 'Recall', 'F1 Score'])

# Transponemos el DataFrame para que los algoritmos estén en filas y las métricas en columnas
df_resultados = df_resultados.T
print(df_resultados)

"""- Random Forests (con 100 y 200 árboles): Destacan con una accuracy superior al 97%, indicando un rendimiento sólido en la mayoría de las métricas.

- k-Nearest Neighbour (KNN): Ofrece un rendimiento elevado en todas las métricas con una accuracy del 96%.

- SVM con kernel lineal: Muestra un buen equilibrio entre precision y recall con una accuracy del 92.8%.

- Árboles de Clasificación: alto rendimiento en todas las métricas. El boosting no ha producido un aumento en el rendimiento.


Las clases AFF y CHF son en las que se han dado mayores errores de clasificación, y, en general, el Random Forest de 100 unidades es el que mejor métricas ofrece, teniendo en cuenta que al aumentar a 200 no se produjo un aumento significativo del rendimiento.

##<h2 id="seccion-10">Referencias</h2>

Mayo Clinic. Frecuencia cardíaca: ¿Cuál es la normal? (2022, 8 octubre). https://www.mayoclinic.org/es/healthy-lifestyle/fitness/expert-answers/heart-rate/faq-20057979

Mayo Clinic. Insuficiencia cardíaca - síntomas y causas. (2023, 29 junio). https://www.mayoclinic.org/es/diseases-conditions/heart-failure/symptoms-causes/syc-20373142

Google Colaboratory. (s. f.-a). https://colab.research.google.com/github/Tanu-N-Prabhu/Python/blob/master/Exploratory_data_Analysis.ipynb
"""
