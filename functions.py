#------------------------FUNCIONES DE ANÁLISIS UNIVARIABLE-----------------------------------------------

#------- ANÁLISIS UNIVARIABLE DE FUNCIONES NUMÉRICAS

def basic_stat(df, columna, show_outliers, bins):
    import pandas as pd
    import seaborn as sb
    import matplotlib.pyplot as plt  # Importamos específicamente pyplot

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
    sb.boxplot(x=df[columna], color='skyblue', showfliers=show_outliers)

    # Agregamos título y etiquetas a los ejes del boxplot
    plt.title(f'Boxplot de {columna}')
    plt.xlabel(columna)  # Usamos el nombre de la columna directamente

    # Mostramos el boxplot
    plt.show()
    # Creamos el histograma con una curva de densidad KDE utilizando Seaborn
    sb.histplot(df[columna], bins=bins, kde=True, color='skyblue', edgecolor='black')

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
    import pandas as pd
    import seaborn as sb
    import matplotlib.pyplot as plt  # Importamos específicamente pyplot
    
    # Calcular las frecuencias de cada categoría en la columna
    frec = df[columna].value_counts().reset_index()
    
    # Crear un gráfico de barras (barplot) con Seaborn
    sb.countplot(x=df[columna], data=df)

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
    import pandas as pd
    import seaborn as sb
    import matplotlib.pyplot as plt

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
        sb.boxplot(x=df[columna], ax=axes[i], color=colors[i], showfliers=show_outliers)
        axes[i].set_title(f'Boxplot de {columna} - DF {i+1}')
        axes[i].set_xlabel(columna)

    # Graficamos los Histogramas (un histograma común para todos los DataFrames)
    for i, df in enumerate(dfs):
        sb.histplot(df[columna], bins=bins, kde=True, color=colors[i], edgecolor='black', ax=axes[i+3])
        axes[i+3].set_title(f'Histograma de {columna} - DF {i+1}')
        axes[i+3].set_xlabel(columna)
        axes[i+3].set_ylabel('Frecuencia')

    # Ajustamos el layout para que no se superpongan
    plt.tight_layout()
    plt.show()


#------- ANÁLISIS UNIVARIABLE DE VARIABLES CATEGÓRICAS (COMPARATIVO DE 3 DATAFRAMES)-----------
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt

def categorical_stat_comparison(dfs, columna):
    """
    Calcula las frecuencias de las categorías en una columna de tres DataFrames
    y genera un gráfico de barras para comparar la distribución de las categorías.

    Parámetros:
    dfs (list): Lista de tres DataFrames.
    columna (str): El nombre de la columna de interés, que debe estar presente en todos los DataFrames.

    No devuelve ningún valor. La función imprime las frecuencias y muestra los gráficos comparativos.
    """

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
        sb.barplot(x=freq.index, y=freq.values, ax=axes[i], color=colors[i])
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
    
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt

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
def corr_map_spearman(df,v_min,v_max):
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt

    spearman_corr_matrix = df.corr(method='spearman')

    # Setting up the matplotlib figure with an appropriate size
    plt.figure(figsize=(18, 15))

    # Drawing the heatmap for the numerical columns
    sns.heatmap(spearman_corr_matrix, annot=True, cmap="copper", vmin=v_min, vmax=v_max)

    plt.title("Spearman Correlation Heatmap for Selected Numerical Variables")
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

    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt

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
        standardized_saleprice = (df[columna] - df[columna].mean()) / df[columna].std()
        
        # Realizar el test de Kolmogorov-Smirnov para comprobar la normalidad
        ks_test_statistic, ks_p_value = stats.kstest(standardized_saleprice, 'norm')
        
        # Imprimir los resultados del test
        if ks_p_value < 0.05:
            print(f'The test results indicate that the distribution of {columna} is significantly different from a normal distribution.')
        else:
            print(f'The test results indicate that the distribution of {columna} is not significantly different from a normal distribution.')
        
        # Crear y mostrar el gráfico de probabilidad (Q-Q plot)
        stats.probplot(df[columna], plot=plt)
        plt.show()
