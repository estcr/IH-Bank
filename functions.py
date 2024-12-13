#------- ANÁLISIS UNIVARIABLE DE FUNCIONES NUMÉRICAS

def basic_stat(df, columna, show_outliers, bins):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns 

    

    """
    Calcula estadísticas descriptivas básicas para una columna de un DataFrame
    y genera un boxplot y un histograma para visualizar la distribución de los datos.

    Parámetros:
    df (pandas.DataFrame): El DataFrame que contiene los datos.
    columna (str): El nombre de la columna de interés dentro del DataFrame para realizar los cálculos y graficar.
    show_outliers (bool): Si es True, se mostrarán los valores atípicos en el boxplot. Si es False, no se mostrarán.
    bins (int): El número de bins (intervalos) a usar en el histograma para dividir los datos.

    No devuelve ningún valor. La función imprime las estadísticas descriptivas y muestra los gráficos de boxplot y histograma.
    
    Estadísticas calculadas:
    - Moda
    - Media
    - Mediana
    - Varianza
    - Desviación estándar
    - Rango (diferencia entre el valor máximo y mínimo)
    - Rango intercuartil (diferencia entre los cuartiles 75% y 25%)
    - Asimetría (skewness)
    - Curtosis (kurtosis)

    Los gráficos generados son:
    - Boxplot: Muestra la distribución de los datos con énfasis en los valores atípicos.
    - Histograma: Muestra la distribución de frecuencia de los datos en intervalos (bins).
    """

    # Calculamos la moda de la columna (valor más frecuente)
    # Tomamos solo el primer valor de la moda si hay varios
    moda = df[columna].mode().iloc[0]

    # Calculamos la media de la columna
    media = df[columna].mean()

    # Calculamos la mediana de la columna
    mediana = df[columna].median()

    # Calculamos la varianza de la columna (mide la dispersión de los datos)
    varianza = df[columna].var()

    # Calculamos la desviación estándar (raíz cuadrada de la varianza)
    desest = df[columna].std()

    # Calculamos el rango, que es la diferencia entre el valor máximo y mínimo de la columna
    rango = df[columna].max() - df[columna].min()

    # Calculamos el rango intercuartil (diferencia entre el cuartil 75 y el cuartil 25)
    rango_i = df[columna].quantile(0.75) - df[columna].quantile(0.25)

    # Calculamos la asimetría de la distribución de los datos (skewness)
    skewness = df[columna].skew()

    # Calculamos la curtosis de la distribución (kurtosis)
    kurtosis = df[columna].kurt()

    # Imprimimos los resultados calculados
    print(f"ANÁLISIS DE VARIABLE: {columna} \n")
    print(f"media: {media} \n mediana: {mediana} \n moda: {moda}\n varianza: {varianza}\n desviación estándar: {desest}\n rango:{rango}\n rango intercuartil: {rango_i} \n Asimetría: {skewness} \n Curtosis: {kurtosis}")

    # Creamos el boxplot con seaborn
    # El parámetro show_outliers controla si mostrar los outliers o no
    sns.boxplot(x=df[columna], color='skyblue', showfliers=show_outliers)

    # Agregamos título y etiquetas a los ejes del boxplot
    plt.title(f'Boxplot de {columna}')
    plt.xlabel(columna)  # Usamos el nombre de la columna directamente

    # Mostramos el boxplot
    plt.show()
    # Creamos el histograma con una curva de densidad KDE utilizando Seaborn
    sns.histplot(df[columna], bins=bins, kde=True, color='skyblue', edgecolor='black')

    # Agregamos título y etiquetas al histograma
    plt.title(f'Histograma de {columna}')
    plt.xlabel(columna)  # Usamos el nombre de la columna directamente
    plt.ylabel('Frecuencia')  # Etiqueta para el eje y

    # Mostramos el histograma
    plt.show()

#------- ANÁLISIS UNIVARIABLE DE FUNCIONES CATEGÓRICAS

def categ_basic_stat(df, columna):
    """
    Realiza un análisis univariado de una columna categórica en un DataFrame. 
    Calcula las frecuencias de las categorías y genera un gráfico de barras.

    Argumentos:
    df (pandas.DataFrame): El DataFrame que contiene los datos.
    columna (str): El nombre de la columna categórica que se desea analizar.

    Retorna:
    pandas.DataFrame: Un DataFrame con las frecuencias de cada categoría en la columna.
    El DataFrame contiene dos columnas:
    - 'index': Las categorías únicas de la columna.
    - 'count': La frecuencia de cada categoría. 
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns 
    
    # Calcular las frecuencias de cada categoría en la columna
    frec = df[columna].value_counts().reset_index()
    
    # Crear un gráfico de barras (barplot) con Seaborn
    sns.countplot(x=df[columna], data=df)

    # Personalizar el gráfico (opcional)
    plt.title(f'Frecuencia en {columna}')
    plt.xlabel('Categoría')
    plt.ylabel('Frecuencia')

    # Mostrar el gráfico
    plt.show()

    # Retornar las frecuencias calculadas
    return frec

#------- ANÁLISIS UNIVARIABLE DE VARIABLES NUMÉRICAS (COMPARATIVO DE 3 DATAFRAMES)-----------

def basic_stat_comparison(dfs, columna, show_outliers, bins):
    """
    Calcula estadísticas descriptivas básicas para una columna de tres DataFrames
    y genera los boxplots y histogramas para comparar los datos de las tres fuentes.

    Parámetros:
    dfs (list): Lista de tres DataFrames.
    columna (str): El nombre de la columna de interés, que debe estar presente en todos los DataFrames.
    show_outliers (bool): Si es True, se mostrarán los valores atípicos en los boxplots.
    bins (int): El número de bins (intervalos) a usar en el histograma para dividir los datos.

    No devuelve ningún valor. La función imprime las estadísticas descriptivas y muestra los gráficos comparativos.
    
    Estadísticas calculadas:
    - Moda
    - Media
    - Mediana
    - Varianza
    - Desviación estándar
    - Rango (diferencia entre el valor máximo y mínimo)
    - Rango intercuartil (diferencia entre los cuartiles 75% y 25%)
    - Asimetría (skewness)
    - Curtosis (kurtosis)

    Los gráficos generados son:
    - Boxplot: Muestra la distribución de los datos con énfasis en los valores atípicos.
    - Histograma: Muestra la distribución de frecuencia de los datos en intervalos (bins).
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns 

    # Verificamos que la columna exista en todos los DataFrames
    for i, df in enumerate(dfs):
        if columna not in df.columns:
            raise ValueError(f"La columna '{columna}' no se encuentra en el DataFrame {i+1}")

    # Configuramos la figura con subgráficas para compararlas
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()  # Aplanamos la matriz de ejes para fácil acceso

    # Definimos una paleta de colores para los tres DataFrames
    colors = ['skyblue', 'lightgreen', 'salmon']

    # Inicializamos las variables para cada columna
    modas = []
    medias = []
    medianas = []
    varianzas = []
    desests = []
    rangos = []
    rango_is = []
    skewness = []
    kurtosis = []

    # Iteramos sobre los tres DataFrames
    for i, df in enumerate(dfs):
        # Cálculos estadísticos
        moda = df[columna].mode().iloc[0]
        media = df[columna].mean()
        mediana = df[columna].median()
        varianza = df[columna].var()
        desest = df[columna].std()
        rango = df[columna].max() - df[columna].min()
        rango_i = df[columna].quantile(0.75) - df[columna].quantile(0.25)
        skew = df[columna].skew()
        kurt = df[columna].kurt()

        # Guardamos los resultados
        modas.append(moda)
        medias.append(media)
        medianas.append(mediana)
        varianzas.append(varianza)
        desests.append(desest)
        rangos.append(rango)
        rango_is.append(rango_i)
        skewness.append(skew)
        kurtosis.append(kurt)

    # Imprimimos los resultados
    print(f"ANÁLISIS DE VARIABLE: {columna}")
    for i, df in enumerate(dfs):
        print(f"  DataFrame {i+1} - media: {medias[i]} | mediana: {medianas[i]} | moda: {modas[i]} | varianza: {varianzas[i]} | desviación estándar: {desests[i]} | rango: {rangos[i]} | rango intercuartil: {rango_is[i]} | Asimetría: {skewness[i]} | Curtosis: {kurtosis[i]}")

    # Graficamos los Boxplots
    for i, df in enumerate(dfs):
        sns.boxplot(x=df[columna], ax=axes[i], color=colors[i], showfliers=show_outliers)
        axes[i].set_title(f'Boxplot de {columna} - DF {i+1}')
        axes[i].set_xlabel(columna)

    # Graficamos los Histogramas (un histograma común para todos los DataFrames)
    for i, df in enumerate(dfs):
        sns.histplot(df[columna], bins=bins, kde=True, color=colors[i], edgecolor='black', ax=axes[i+3])
        axes[i+3].set_title(f'Histograma de {columna} - DF {i+1}')
        axes[i+3].set_xlabel(columna)
        axes[i+3].set_ylabel('Frecuencia')

    # Ajustamos el layout para que no se superpongan
    plt.tight_layout()
    plt.show()


#------- ANÁLISIS UNIVARIABLE DE VARIABLES CATEGÓRICAS (COMPARATIVO DE 3 DATAFRAMES)-----------


def categorical_stat_comparison(dfs, columna):
    """
    Calcula las frecuencias de las categorías en una columna de tres DataFrames
    y genera un gráfico de barras para comparar la distribución de las categorías.

    Parámetros:
    dfs (list): Lista de tres DataFrames.
    columna (str): El nombre de la columna de interés, que debe estar presente en todos los DataFrames.

    No devuelve ningún valor. La función imprime las frecuencias y muestra los gráficos comparativos.
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns 
    # Verificamos que la columna exista en todos los DataFrames
    for i, df in enumerate(dfs):
        if columna not in df.columns:
            raise ValueError(f"La columna '{columna}' no se encuentra en el DataFrame {i+1}")

    # Configuramos la figura con subgráficas para compararlas
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Definimos una paleta de colores para los tres DataFrames
    colors = ['skyblue', 'lightgreen', 'salmon']

    # Iteramos sobre los tres DataFrames para calcular las frecuencias y graficar
    for i, df in enumerate(dfs):
        # Calculamos las frecuencias de las categorías en la columna
        freq = df[columna].value_counts()

        # Imprimimos las frecuencias
        print(f"FRECUENCIAS DE LA COLUMNA '{columna}' EN EL DataFrame {i+1}:")
        print(freq)
        print()

        # Graficamos el gráfico de barras para las frecuencias
        sns.barplot(x=freq.index, y=freq.values, ax=axes[i], color=colors[i])
        axes[i].set_title(f'Frecuencias de {columna} - DF {i+1}')
        axes[i].set_xlabel(columna)
        axes[i].set_ylabel('Frecuencia')

    # Ajustamos el layout para que no se superpongan
    plt.tight_layout()
    plt.show()


#-----------------------MAPA DE CORRELACIÓN LINEAL (PEARSON) ENTRE VARIABLES NUMÉRICAS
def corr_map_pearson(df, v_min, v_max):
    """
    Esta función genera un mapa de calor (heatmap) para visualizar la matriz de correlación de Pearson entre las variables 
    numéricas de un DataFrame. El gráfico muestra la fuerza y dirección de la relación lineal entre cada par de variables.

    Parámetros:
    df (DataFrame): 
        Un DataFrame de pandas que contiene las variables numéricas entre las que se calculará la correlación de Pearson.
    v_min (float): 
        El valor mínimo para la escala de colores en el heatmap. Este parámetro controla el rango inferior de la visualización de la correlación.
    v_max (float): 
        El valor máximo para la escala de colores en el heatmap. Este parámetro controla el rango superior de la visualización de la correlación.

    Retorna:
    None. La función genera un gráfico de tipo heatmap que visualiza las correlaciones entre las variables numéricas del DataFrame.

    Descripción del flujo:
    1. La función comienza calculando la **matriz de correlación de Pearson** entre las variables numéricas del DataFrame utilizando el método `.corr()` de pandas.
    2. Luego, se configura el gráfico con un tamaño adecuado utilizando `matplotlib`.
    3. Se utiliza **Seaborn** para dibujar el heatmap, donde cada celda del mapa muestra la correlación entre las variables correspondientes.
    4. Finalmente, se añade un título al gráfico y se visualiza utilizando `matplotlib`.

    """
    
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns 

    # Cálculo de la matriz de correlación de Pearson entre las variables numéricas del DataFrame
    correlation_matrix = df.corr()

    # Configuración de la figura de matplotlib con un tamaño adecuado
    plt.figure(figsize=(18, 15))

    # Dibujar el primer mapa de calor (heatmap) para la matriz de correlación completa
    sns.heatmap(correlation_matrix, annot=True, cmap="cividis", vmin=v_min, vmax=v_max)

    # Añadir un título al gráfico
    plt.title("Pearson Correlation Heatmap for Selected Numerical Variables")
    plt.show()



    #------------------MAPA DE CORRELACIÓN LINEAL/MONÓTONA (SPEARMAN) ENTRE VARIABLES NUMÉRICAS
def corr_map_spearman(df, v_min, v_max):
    """
    Esta función genera un mapa de calor (heatmap) que muestra la matriz de correlación de Spearman
    entre las variables numéricas de un DataFrame. Utiliza seaborn para visualizar la correlación 
    entre las columnas numéricas, mostrando valores numéricos de la correlación y ajustando los 
    valores del mapa de calor dentro de un rango especificado.

    Parámetros:
    - df (pandas.DataFrame): El DataFrame que contiene las variables numéricas para las cuales 
                              se calculará la correlación de Spearman.
    - v_min (float): El valor mínimo del rango de color para la visualización del heatmap.
    - v_max (float): El valor máximo del rango de color para la visualización del heatmap.

    Retorna:
    - None: La función no retorna un valor, pero muestra un heatmap de la correlación de Spearman.

    Funciona de la siguiente manera:
    1. Calcula la matriz de correlación de Spearman entre las variables numéricas del DataFrame.
    2. Utiliza seaborn para generar un mapa de calor con la matriz de correlación.
    3. Ajusta los límites de color del heatmap según los valores `v_min` y `v_max` proporcionados.
    4. Muestra el gráfico con matplotlib.
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns 

    # Calculando la matriz de correlación de Spearman
    spearman_corr_matrix = df.corr(method='spearman')

    # Configurando el tamaño de la figura del gráfico
    plt.figure(figsize=(18, 15))

    # Dibujando el heatmap con los valores de correlación y los límites de color ajustados
    sns.heatmap(spearman_corr_matrix, annot=True, cmap="copper", vmin=v_min, vmax=v_max)

    # Añadiendo título al gráfico
    plt.title("Spearman Correlation Heatmap for Selected Numerical Variables")
    
    # Mostrando el heatmap
    plt.show()


    #------------------GRÁFICAS DE  CORRELACIÓN ENTRE VARIABLES NUMÉRICAS TEST VS CONTROL

def pair_plots(df, var, vars, hue_var):
    """
    Esta función genera gráficos de pares (pairplots) para un conjunto de variables de un DataFrame,
    con la posibilidad de diferenciación por una variable categórica (hue).

    Parámetros:
    df (DataFrame): Un DataFrame de pandas que contiene los datos a visualizar.
    var (str): El nombre de la variable principal (en el eje X) para las gráficas de pares.
    vars (list of str): Una lista de nombres de columnas (strings) que se compararán con `var`.
    hue_var (str): El nombre de la variable categórica para diferenciar los puntos en los gráficos.
    
    Retorna:
    None. La función genera y muestra un gráfico para cada par de variables.

    Descripción del flujo:
    1. La función recorre cada elemento de `vars`, que contiene una lista de variables a comparar con `var`.
    2. Para cada par de variables, se genera un gráfico de pares utilizando `seaborn.pairplot`.
    3. En cada gráfico, se dibuja una línea de regresión, y se utiliza `hue_var` para colorear los puntos de acuerdo a la variable categórica.
    4. Se muestra el gráfico con `plt.show()`.
    """

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns 

    # Iterar a través de la lista de variables
    for vars_n in vars:
        # Seleccionar las columnas necesarias para el gráfico
        selected_columns = [var, vars_n, hue_var]
        
        # Crear el gráfico de pares con la regresión lineal
        g = sns.pairplot(df[selected_columns], hue=hue_var, kind='reg', 
                        plot_kws={'line_kws': {'color': 'red', 'linewidth': 2}})
        
        # Ajustar el espacio para que el título no se solape
        g.fig.subplots_adjust(top=0.95)  # Ajuste del espacio superior
        
        # Establecer el título del gráfico (fuera del área del gráfico para evitar solapamientos)
        g.fig.suptitle(f"Gráfica de correlación entre {var} y {vars_n}", fontsize=16)
        
        # Mostrar el gráfico
        plt.show()


# ------------------- FUNCIÓN PARA REALIZAR ESTANDARIZACIÓN Y PRUEBA DE KOLMOGOROV

def kolm_smir_test(df):
    """
    Esta función realiza el test de Kolmogorov-Smirnov para evaluar la normalidad de las distribuciones
    de las variables en un DataFrame, utilizando las columnas del DataFrame para realizar las pruebas de normalidad.

    Para cada columna del DataFrame, la función calcula el estadístico de Kolmogorov-Smirnov y su valor p
    para comprobar si la distribución de la variable es significativamente diferente de una distribución normal.

    Parámetros:
    df (DataFrame): Un DataFrame de pandas con las variables a evaluar. Cada columna se considera una variable
                    y se realiza una prueba de normalidad sobre ella.

    Retorna:
    None. La función imprime el resultado del test de normalidad para cada variable y genera un gráfico de probabilidad
          para cada una, visualizando si se ajusta a una distribución normal.

    Descripción del flujo:
    1. La función recorre cada columna del DataFrame.
    2. Para cada columna, calcula una versión estandarizada de los datos (media 0 y desviación estándar 1).
    3. Realiza el test de Kolmogorov-Smirnov usando la función `kstest` de SciPy, comparando la distribución
       estandarizada de los datos con una distribución normal estándar.
    4. Si el valor p del test es menor que 0.05, se considera que la distribución es significativamente diferente de la normal.
    5. Se muestra un gráfico de probabilidad para cada variable utilizando `probplot` de SciPy, lo que ayuda a visualizar
       la normalidad de los datos.
    """
    
    from scipy import stats
    import matplotlib.pyplot as plt

    # Iterar a través de las columnas del DataFrame
    for columna in df.columns:
        # Estandarizar los datos (media 0 y desviación estándar 1)
        standardized_z = (df[columna] - df[columna].mean()) / df[columna].std()
        
        # Realizar el test de Kolmogorov-Smirnov para comprobar la normalidad
        ks_test_statistic, ks_p_value = stats.kstest(standardized_z, 'norm')
        
        # Imprimir los resultados del test
        if ks_p_value < 0.05:
            print(f'The test results indicate that the distribution of {columna} : P value = {ks_p_value}')

        # Crear y mostrar el gráfico de probabilidad (Q-Q plot)
        stats.probplot(df[columna], plot=plt)
        plt.show()



def max_min_norm(df):
    """
    Aplica una transformación de normalización Min-Max a cada columna numérica de un DataFrame.
    
    La normalización Min-Max escala cada valor dentro de una columna para que esté en el rango [0, 1],
    usando la fórmula:
        X_norm = (X - X_min) / (X_max - X_min)
    
    Args:
        df (pd.DataFrame): El DataFrame de entrada que contiene las columnas numéricas que se desean normalizar.

    Returns:
        pd.DataFrame: Un nuevo DataFrame con las mismas columnas que el original, pero con los valores normalizados.
                      Cada columna tendrá un sufijo "_norm" para indicar que ha sido transformada.
    """
    
    from sklearn.preprocessing import MinMaxScaler
    import pandas as pd

    # Crear un nuevo DataFrame para almacenar las columnas normalizadas
    df_norm = pd.DataFrame()
    
    # Iterar a través de cada columna en el DataFrame original
    for columna in df:
        # Inicializando el escalador MinMaxScaler
        scaler = MinMaxScaler()  # Usamos el MinMaxScaler para escalar los valores

        # Realiza la transformación de Min-Max y guarda el resultado en el nuevo DataFrame
        df_norm[columna] = pd.DataFrame(scaler.fit_transform(df[[columna]]), columns=[columna + "_norm"])
        # La columna original es transformada y almacenada con el sufijo "_norm"
    
    # Retorna el DataFrame con las columnas normalizadas
    return df_norm

#----------------------- PRUEBA DE HIPÓTESIS CON T STUDENT -----------------------------------
def t_test(df1, var1, df2, var2, hipo):
    """
    Realiza una prueba t de dos muestras independientes para comparar las medias de dos grupos
    en base a las columnas especificadas de dos DataFrames. La prueba t se realiza utilizando
    el test de Welch, que no asume varianzas iguales entre los dos grupos.

    Parámetros:
    df1 : pandas.DataFrame
        El primer DataFrame que contiene la variable de interés (var1).
    var1 : str
        El nombre de la columna en df1 que se desea analizar.
    df2 : pandas.DataFrame
        El segundo DataFrame que contiene la variable de interés (var2).
    var2 : str
        El nombre de la columna en df2 que se desea analizar.
    hipo : str
        Tipo de hipótesis alternativa. Debe ser uno de los siguientes:
        - 'two-sided' : prueba de dos colas (las medias de ambos grupos son diferentes).
        - 'less' : prueba de una cola, en la que se prueba si la media de df1[var1] es menor que la de df2[var2].
        - 'greater' : prueba de una cola, en la que se prueba si la media de df1[var1] es mayor que la de df2[var2].

    Salida:
    None
        Imprime el valor p y un mensaje indicando si hay suficiente evidencia para rechazar la hipótesis nula.
    """
    
    # Importa el módulo de scipy.stats para realizar la prueba t
    import scipy.stats as st
    
    # Realiza la prueba t de dos muestras independientes con el test de Welch (equal_var=False)
    # La prueba se realiza para comparar las medias de las variables var1 y var2 de los DataFrames df1 y df2.
    t_statistic, p_value = st.ttest_ind(df1[var1], df2[var2], equal_var=False, alternative=hipo)
    
    # Si el p-valor es mayor que 0.05, no se rechaza la hipótesis nula
    if p_value > 0.05:
        print(f'P value = {p_value} \nNo hay evidencia para rechazar la hipótesis nula')
    else:
        # Si el p-valor es menor o igual que 0.05, se rechaza la hipótesis nula
        print(f'P value = {p_value} \nHay evidencia para rechazar la hipótesis nula')



#--------------------------------------GRAFICOS---------------------------------------------

def create_box_plot_with_columns(df, title="Box Plot Comparison"):
    """
    Crea un gráfico de caja (box plot) para comparar tiempos específicos.

    Args:
        df (pd.DataFrame): DataFrame que contiene los datos.
        title (str): Título del gráfico.

    Returns:
        plotly.graph_objects.Figure: Figura del gráfico de caja.
    """
    import plotly.express as px
    # Seleccionar las columnas relevantes para el análisis
    time_columns = ["confirm_time", "start_time", "step_1_time", "step_2_time", "step_3_time"]
    
    # Informar sobre las columnas seleccionadas
    print(f"Las columnas seleccionadas para el análisis son: {time_columns}")
    
    # Reorganizar los datos en formato largo para el gráfico
    df_melted = df[time_columns].melt(var_name="variable", value_name="value")
    
    # Crear el gráfico de caja con Plotly Express
    fig = px.box(df_melted, 
                 x='variable', 
                 y='value',
                 title=title,
                 points='outliers',
                 color='variable')

    # Personalizar el diseño del gráfico
    fig.update_layout(
        showlegend=False,
        yaxis_title='Values',
        xaxis_title='Variables',
        template='plotly_white'
    )
    
    return fig



def perform_time_statistical_tests(df_merged_final_test, df_merged_final_control):
    """
    Realiza pruebas T para comparar las columnas de tiempo entre los grupos Test y Control,
    genera gráficos de caja y explica las hipótesis.

    Args:
        df_merged_final_test (pd.DataFrame): DataFrame con los datos del grupo Test.
        df_merged_final_control (pd.DataFrame): DataFrame con los datos del grupo Control.
    """
    import pandas as pd
    from scipy.stats import ttest_ind
    import plotly.express as px
    # 1. Comparación para columnas de tiempo (`_time`).
    time_columns = ['confirm_time', 'start_time', 'step_1_time', 'step_2_time', 'step_3_time']
    
    print("Comparación de columnas de tiempo (T-test):\n")
    for col in time_columns:
        test_time = df_merged_final_test[col]
        control_time = df_merged_final_control[col]
        
        # Realizar la prueba T de varianzas desiguales
        t_stat, p_value = ttest_ind(test_time, control_time, equal_var=False)
        
        # Explicación de la hipótesis
        print(f"\nPrueba T para {col}:")
        print(f"  Hipótesis nula (H0): No hay diferencia significativa entre Test y Control en {col}.")
        print(f"  Hipótesis alternativa (H1): Hay una diferencia significativa entre Test y Control en {col}.")
        print(f"  Estadístico T: {t_stat:.2f}, Valor P: {p_value:.4f}")
        
        if p_value < 0.05:
            print("  -> Diferencia significativa entre Test y Control.\n")
        else:
            print("  -> No hay diferencia significativa.\n")
        
        # Crear gráfico de caja
        time_data = pd.DataFrame({
            'Tiempo (Segundos)': pd.concat([test_time, control_time], axis=0),
            'Grupo': ['Test'] * len(test_time) + ['Control'] * len(control_time)
        })
        
        fig = px.box(
            time_data,
            x='Grupo',
            y='Tiempo (Segundos)',
            title=f'Distribución de {col}',
            color='Grupo',
            color_discrete_sequence=["#636EFA", "#EF553B"],  # Colores personalizados
            points='all'  # Mostrar todos los puntos, incluidos los outliers
        )
        fig.update_layout(
            xaxis_title='Grupo',
            yaxis_title='Segundos',
            boxmode='group'  # Agrupar las cajas por grupo
        )
        fig.show()




def perform_count_statistical_tests(df_merged_final_test, df_merged_final_control):
    """
    Realiza pruebas Chi-cuadrado para comparar las columnas de conteo entre los grupos Test y Control,
    genera gráficos de barras y explica las hipótesis.

    Args:
        df_merged_final_test (pd.DataFrame): DataFrame con los datos del grupo Test.
        df_merged_final_control (pd.DataFrame): DataFrame con los datos del grupo Control.
    """

    import pandas as pd
    from scipy.stats import chi2_contingency
    import plotly.express as px
    # 1. Comparación para columnas de conteo (`_count`).
    count_columns = ['confirm_count', 'start_count', 'step_1_count', 'step_2_count', 'step_3_count']
    
    print("Comparación de columnas de conteo (Chi-cuadrado):\n")
    for col in count_columns:
        test_count = df_merged_final_test[col].value_counts()
        control_count = df_merged_final_control[col].value_counts()
        
        # Crear tabla de contingencia
        contingency_table = pd.DataFrame({
            'Test': test_count,
            'Control': control_count
        }).fillna(0)  # Rellenar NaN con 0 para evitar errores
        
        # Realizar prueba Chi-cuadrado
        chi2, p_value, _, _ = chi2_contingency(contingency_table)
        
        # Explicación de la hipótesis
        print(f"\nPrueba Chi-cuadrado para {col}:")
        print(f"  Hipótesis nula (H0): No hay diferencia significativa en la distribución de {col} entre Test y Control.")
        print(f"  Hipótesis alternativa (H1): Hay una diferencia significativa en la distribución de {col} entre Test y Control.")
        print(f"  Chi2: {chi2:.2f}, Valor P: {p_value:.4f}")
        
        if p_value < 0.05:
            print("  -> Diferencia significativa entre Test y Control.\n")
        else:
            print("  -> No hay diferencia significativa.\n")
        
        # Crear gráfico de barras
        counts = pd.DataFrame({
            'Grupo': ['Test'] * len(test_count) + ['Control'] * len(control_count),
            'Valor': list(test_count.index) + list(control_count.index),
            'Conteo': list(test_count.values) + list(control_count.values)
        })
        
        fig = px.bar(
            counts,
            x='Valor',
            y='Conteo',
            color='Grupo',
            barmode='group',
            title=f'Frecuencia de {col} por Grupo',
            color_discrete_sequence=["#636EFA", "#EF553B"]  # Colores personalizados
        )
        fig.update_layout(
            xaxis_title='Valor',
            yaxis_title='Número de ocurrencias',
            legend_title='Grupo'
        )
        fig.show()




def create_boxplot_comparison(df, column, title="Test vs Control Group Comparison"):
    """
    Crea un gráfico de caja (boxplot) para comparar los valores de una columna entre los grupos 'Test' y 'Control'.

    Args:
        df (pd.DataFrame): DataFrame que contiene los datos.
        column (str): El nombre de la columna que se desea comparar entre los grupos.
        title (str): Título del gráfico (opcional).
    
    Returns:
        plotly.graph_objects.Figure: El gráfico de caja generado.
    """

    import plotly.express as px
    # Crear el gráfico de caja
    fig = px.box(
        df,
        x='Variation',  # Columna para separar los grupos
        y=column,  # Columna para comparar
        title=title,
        points='outliers'  # Mostrar los outliers
    )
    
    # Personalizar el diseño del gráfico
    fig.update_layout(
        yaxis_title='Values',
        xaxis_title='Group',
        template='plotly_white'
    )
    
    # Mostrar el gráfico
    fig.show()

    return fig
#---------------------LLAMADO A CSV DE DATA FRAME FINAL---------------------------------------------------------------
def llama_datos():
    import pandas as pd
    df_final_completo=pd.read_csv("Data/cleaned/finalcompleto.csv")
    df_test=pd.read_csv("Data/cleaned/test.csv")
    df_tasas=pd.read_csv("Data/cleaned/tasas.csv")
    df_control=pd.read_csv("Data/cleaned/control.csv")

    df_test_num = df_test[['confirm_count', 'start_count', 'step_1_count', 'confirm_time', 'start_time', 
                       'step_1_time', 'step_2_time', 'step_3_time', 'step_2_count', 'step_3_count', 
                        'clnt_tenure_yr', 'clnt_tenure_mnth', 'clnt_age', 'num_accts', 
                       'bal', 'calls_6_mnth', 'logons_6_mnth']]

    # Selección de columnas categóricas
    df_test_categ = df_test[['visit_id', 'client_id', 'visitor_id', 'gendr']]

    df_control_num = df_control[['confirm_count', 'start_count', 'step_1_count', 'confirm_time', 'start_time', 
                       'step_1_time', 'step_2_time', 'step_3_time', 'step_2_count', 'step_3_count', 
                       'clnt_tenure_yr', 'clnt_tenure_mnth', 'clnt_age', 'num_accts', 
                       'bal', 'calls_6_mnth', 'logons_6_mnth']]

    # Selección de columnas categóricas
    df_control_categ = df_control[['visit_id', 'client_id', 'visitor_id', 'gendr']]
    
    return df_final_completo, df_test_num, df_test_categ, df_control_num, df_control_categ, df_tasas

