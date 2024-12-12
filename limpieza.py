import pandas as pd

def process_visitas_data(df_parts):
    """
    Procesa los datos de visitas desde múltiples DataFrames y devuelve el DataFrame final consolidado.
    
    Args:
        df_parts (list): Lista de DataFrames a concatenar antes de procesar.

    Returns:
        pd.DataFrame: DataFrame procesado con conteos, tiempos promedio por paso, y tasa de éxito.
    """
    # Paso 0: Concatenar los DataFrames de entrada
    df_visitas = pd.concat(df_parts, axis=0, ignore_index=True)

    # Paso 1: Convertir la columna 'date_time' a datetime
    df_visitas['date_time'] = pd.to_datetime(df_visitas['date_time'], format="%Y-%m-%d %H:%M:%S")
    
    # Paso 2: Ordenar por 'visitor_id' y 'date_time'
    df_visitas = df_visitas.sort_values(by=['visitor_id', 'date_time'])
    
    # Paso 3: Calcular el tiempo entre pasos
    df_visitas['time_in_step'] = (
        df_visitas.groupby('visit_id')['date_time'].shift(-1) - df_visitas['date_time']
    ).dt.total_seconds()
    
    # Paso 4: Filtrar los registros con 'time_in_step' <= 2 ya que los consideramos error
    df_visitas = df_visitas[df_visitas['time_in_step'] > 2]
    
    # Paso 5: Contar cuántas veces ocurre cada paso para cada 'visitor_id'
    count_steps = df_visitas.groupby(['client_id', 'visitor_id', 'visit_id', 'process_step']).size().reset_index(name='count')
    
    # Paso 6: Obtener el tiempo promedio en cada paso para cada 'visitor_id'
    time_steps = df_visitas.groupby(
        ['client_id', 'visitor_id', 'visit_id', 'process_step']
    )['time_in_step'].mean().reset_index(name='avg_time_in_step')
    
    # Paso 7: Combinar conteos y tiempos para obtener el DataFrame final
    df_visitas_final = pd.merge(count_steps, time_steps, on=['client_id', 'visitor_id', 'visit_id', 'process_step'], how='left')


    return df_visitas_final

def date_time_visit_id(df_parts):
    """
    Procesa los datos de visitas desde múltiples DataFrames y devuelve el DataFrame final consolidado.
    
    Args:
        df_parts (list): Lista de DataFrames a concatenar antes de procesar.

    Returns:
        pd.DataFrame: DataFrame procesado con conteos, tiempos promedio por paso, y tasa de éxito.
    """
    import pandas as pd
    # Paso 0: Concatenar los DataFrames de entrada
    df_visitas = pd.concat(df_parts, axis=0, ignore_index=True)

    # Paso 1: Convertir la columna 'date_time' a datetime
    df_visitas['date_time'] = pd.to_datetime(df_visitas['date_time'], format="%Y-%m-%d %H:%M:%S")
    
    # Paso 2: Ordenar por 'visitor_id' y 'date_time'
    df_visitas = df_visitas.sort_values(by=['visitor_id', 'date_time'])

    # Paso 3:
    df_date = df_visitas.groupby('visit_id').agg(min_date=('date_time', 'min')).reset_index()

    # Paso 4: Extraer solo la hora de min_date y renombrar la columna
    df_date['date'] = df_date['min_date'].dt.date

    # Paso 5: Eliminar la columna original
    df_date = df_date.drop(columns=['min_date'])
    return df_date


import pandas as pd

def create_df_tasas(df):
    """
    Toma un DataFrame con columnas de conteo para distintos `process_step`
    y devuelve otro DataFrame con columnas normalizadas en un orden específico,
    además de agregar las columnas `Variation` y `failure_rate`.

    Args:
        df (pd.DataFrame): DataFrame original con columnas como 'confirm_count', 'start_count', etc.

    Returns:
        pd.DataFrame: Nuevo DataFrame con las columnas normalizadas, `Variation` y `failure_rate`.
    """
    # Crear un DataFrame vacío para almacenar las columnas normalizadas
    tasas_df = pd.DataFrame()

    # Copiar identificadores clave del DataFrame original al nuevo
    for col in ['client_id', 'visitor_id', 'visit_id', 'date_time', 'Variation']:
        if col in df.columns:
            tasas_df[col] = df[col]

    # Orden de los pasos para las columnas normalizadas
    steps = ['start', 'step_1', 'step_2', 'step_3', 'confirm']

    # Crear las columnas normalizadas para cada paso
    for step in steps:
        count_column = f'{step}_count'
        if count_column in df.columns:
            # Normalización (dividir 1 / número, con verificación para evitar división por 0)
            tasas_df[f'{step}_rate'] = df[count_column].apply(lambda x: 1 / x if x != 0 else 0)
        else:
            # Si no existe la columna de conteo, rellenar con 0
            tasas_df[f'{step}_rate'] = 0

    # Calcular la columna failure_rate (1 si confirm_count es 0, de lo contrario 0)
    if 'confirm_count' in df.columns:
        tasas_df['failure_rate'] = df['confirm_count'].apply(lambda x: 1 if x == 0 else 0)
    else:
        # Si no existe confirm_count, se rellena con 0
        tasas_df['failure_rate'] = 0

    return tasas_df




def create_pivot_summary(df_visitas_final):
    """
    Crea un resumen pivoteado del DataFrame final con conteos, tiempos y datos adicionales.
    
    Args:
        df_visitas_final (pd.DataFrame): DataFrame procesado con conteos y tiempos por paso.

    Returns:
        pd.DataFrame: DataFrame con los datos pivotados y los campos `visit_id`, `client_id` y `visitor_id`.
    """
    # Paso 1: Agrupar por cliente, visitante, visita y paso del proceso
    df_visitas_grouped = df_visitas_final.groupby(['client_id', 'visitor_id', 'visit_id', 'process_step'], as_index=False).agg(
        count=('count', 'sum'),
        avg_time_in_step=('avg_time_in_step', 'mean')
    )

    # Paso 2: Crear la tabla pivote para los conteos
    pivot_count = df_visitas_grouped.pivot_table(
        index='visit_id', 
        columns='process_step', 
        values='count', 
        aggfunc='sum', 
        fill_value=0
    )
    pivot_count.columns = [f'{col}_count' for col in pivot_count.columns]  # Renombrar columnas

    # Paso 3: Crear la tabla pivote para los tiempos
    pivot_time = df_visitas_grouped.pivot_table(
        index='visit_id', 
        columns='process_step', 
        values='avg_time_in_step', 
        aggfunc='mean', 
        fill_value=0
    )
    pivot_time.columns = [f'{col}_time' for col in pivot_time.columns]  # Renombrar columnas

    # Paso 4: Combinar ambas tablas
    df_pivot_final = pd.concat([pivot_count, pivot_time], axis=1)

    # Paso 5: Agregar columnas adicionales al DataFrame final
    df_pivot_final = df_pivot_final.reset_index()
    df_pivot_final = df_pivot_final.merge(
        df_visitas_final[['client_id', 'visitor_id', 'visit_id']].drop_duplicates(), 
        on='visit_id', 
        how='left'
    )

    return df_pivot_final

def merge_tables(df_pivot_final, df_final_demo, df_experiment_clients):
    """
    Realiza la unión de los DataFrames proporcionados en un único DataFrame final.
    
    Args:
        df_pivot_final (pd.DataFrame): DataFrame con los datos pivotados y procesados.
        df_final_demo (pd.DataFrame): DataFrame con información demográfica.
        df_experiment_clients (pd.DataFrame): DataFrame con información de experimentos.
        
    Returns:
        pd.DataFrame: DataFrame resultante con todas las tablas unidas y filtrado de valores nulos.
    """
    # Realizar el primer merge entre df_pivot_final y df_final_demo usando 'client_id'
    df_merged_1 = pd.merge(df_pivot_final, df_final_demo, on='client_id', how='left')
    
    # Realizar el segundo merge entre el resultado anterior y df_experiment_clients usando 'client_id'
    df_merged_final = pd.merge(df_merged_1, df_experiment_clients, on='client_id', how='left')
    
    # Filtrar filas con menos de 5 columnas con valores nulos
    df_merged_final = df_merged_final[df_merged_final.isna().sum(axis=1) < 5]
    
    return df_merged_final



def clean_outliers_and_nans(df):
    """
    Filtra las filas del DataFrame eliminando duplicados, outliers según los valores máximos predefinidos 
    para cada columna, y elimina las filas con más de 5 valores nulos.

    Args:
        df (pd.DataFrame): DataFrame con los datos originales.

    Returns:
        pd.DataFrame: DataFrame limpio, con los duplicados y outliers eliminados y las filas con muchos valores nulos eliminadas.
    """
    # Paso 0: Eliminar duplicados
    print("Eliminando duplicados...")
    df = df.drop_duplicates(subset=["visit_id"], keep='first')  # Puedes ajustar 'subset' si necesitas columnas específicas.
    print(f"Filas restantes después de eliminar duplicados: {len(df)}")
    
    # Paso 1: Definir los umbrales internos para cada columna
    thresholds = {
        "confirm_time": 4500,
        "start_time": 4000,
        "step_1_time": 3500,
        "step_2_time": 1900,
        "step_3_time": 4000
    }

    # Paso 2: Filtrar el DataFrame según los umbrales predefinidos
    print(f"Aplicando filtros basados en los siguientes umbrales: {thresholds}")
    for col, max_value in thresholds.items():
        df = df[df[col] <= max_value]
    print(f"Filas restantes después de eliminar outliers: {len(df)}")

    # Paso 3: Eliminar filas con más de 5 valores nulos
    print("Eliminando filas con más de 5 valores nulos...")
    df = df[df.isnull().sum(axis=1) <= 5]
    print(f"Filas restantes después de eliminar filas con muchos valores nulos: {len(df)}")

    # Reiniciar el índice para el DataFrame resultante
    df_cleaned = df.reset_index(drop=True)
    
    # Mostrar las primeras filas del DataFrame limpio
    print("Datos filtrados y limpios generados.")
    print(df_cleaned.head())  # Esto es solo para mostrar las primeras filas como ejemplo

    return df_cleaned


def split_by_variation(df, column_name="Variation"):
    """
    Divide el DataFrame en dos: uno para 'Test' y otro para 'Control', 
    y elimina la columna especificada (por defecto 'Variation').

    Args:
        df (pd.DataFrame): DataFrame con los datos originales.
        column_name (str): Nombre de la columna que contiene la variación ('Test' o 'Control').

    Returns:
        tuple: Dos DataFrames, uno para 'Test' y otro para 'Control'.
    """
    # Filtrar filas para 'Test'
    df_test = df[df[column_name] == "Test"].drop(columns=[column_name])
    
    # Filtrar filas para 'Control'
    df_control = df[df[column_name] == "Control"].drop(columns=[column_name])
    
    return df_test, df_control


def save_all_csv(df_merged_final, df_date, df_merged_final_control, df_merged_final_test, df_merged_final_errores, df_tasas):
    """
    Guarda los DataFrames proporcionados en archivos CSV en la carpeta 'Data/cleaned'.
    
    Args:
        df_merged_final (pd.DataFrame): DataFrame principal que se guardará como 'finalcompleto.csv'.
        df_merged_final_control (pd.DataFrame): DataFrame con los datos del grupo control.
        df_merged_final_test (pd.DataFrame): DataFrame con los datos del grupo test.
        df_merged_final_errores (pd.DataFrame): DataFrame con los errores a guardar.
        df_merged_final_tasas (pd.DataFrame): DataFrame con las tasas a guardar.
    """
    # Guardar el DataFrame df_merged_final
    df_merged_final.to_csv(r'Data\cleaned\finalcompleto.csv', index=False)
    print("Archivo 'finalcompleto.csv' guardado.")

    # Guardar el DataFrame df_date
    df_date.to_csv(r'Data\cleaned\date.csv', index=False)
    print("Archivo 'date.csv' guardado.")

    # Guardar el DataFrame df_merged_final_control
    df_merged_final_control.to_csv(r'Data\cleaned\control.csv', index=False)
    print("Archivo 'control.csv' guardado.")
    
    # Guardar el DataFrame df_merged_final_test
    df_merged_final_test.to_csv(r'Data\cleaned\test.csv', index=False)
    print("Archivo 'test.csv' guardado.")
    
    # Guardar el DataFrame df_merged_final_errores
    df_merged_final_errores.to_csv(r'Data\cleaned\error.csv', index=False)
    print("Archivo 'error.csv' guardado.")

    # Guardar el DataFrame df_merged_final_rate
    df_tasas.to_csv(r'Data\cleaned\tasas.csv', index=False)
    print("Archivo 'tasas.csv' guardado.")


def execute_all(df_parts, df_final_demo, df_experiment_clients):
    """
    Ejecuta todas las funciones de limpieza, procesamiento y generación de archivos CSV.

    Args:
        df_parts (list): Lista de DataFrames a concatenar y procesar.
        df_final_demo (pd.DataFrame): DataFrame con información demográfica.
        df_experiment_clients (pd.DataFrame): DataFrame con información de experimentos.

    Returns:
        None: Genera y guarda los CSV en la carpeta indicada.
    """
    print("Iniciando procesamiento de datos...")

    # Proceso principal
    df_visitas_final = process_visitas_data(df_parts)
    print("Datos de visitas procesados.")

    # Se borran filas ya que las mismas solo tenian el client id y ningun dato mas
    df_final_demo = df_final_demo[~(df_final_demo.drop(columns=['client_id']).isna().all(axis=1))]

    df_date = date_time_visit_id(df_parts)
    print("Datos agrupados por visit_id procesados.")

    df_pivot_final = create_pivot_summary(df_visitas_final)
    print("Resumen pivoteado creado.")

    df_merged_final = merge_tables(df_pivot_final, df_final_demo, df_experiment_clients)
    print("Tablas unidas correctamente.")

    df_cleaned = clean_outliers_and_nans(df_merged_final)
    print("Datos limpios de outliers y valores nulos.")

    df_tasas = create_df_tasas(df_cleaned)
    print("Tasas calculadas correctamente.")

    df_merged_final_test, df_merged_final_control = split_by_variation(df_cleaned)
    print("Datos divididos en Test y Control.")

    # Guardar los archivos CSV
    save_all_csv(df_cleaned, df_date, df_merged_final_control, df_merged_final_test, df_merged_final, df_tasas)
    print("Todos los archivos CSV han sido guardados exitosamente.")


