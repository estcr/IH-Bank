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