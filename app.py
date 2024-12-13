"""
Este documento con tiene el código para la construcción del frontend del proyecto "A/B TEST"
Se utiliza streamlit.

"""
import streamlit as st
st.set_page_config(layout="wide")
import functions as f
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
# Personalización de estilos
st.markdown("""
    <style>
        /* Cambiar color de fondo de la barra lateral */
        .css-1d391kg {background-color: #5c2d91;} /* Morado para la barra lateral */
        
        /* Cambiar el color de los botones */
        .css-1emrehy.edgvbvh3 {
            background-color: #5c2d91;
            color: white;
        }
        
        /* Cambiar color de texto y otros elementos destacados */
        .css-1v0mbdj {
            color: #5c2d91;
        }
        
        /* Personalización de los encabezados */
        h1, h2, h3, h4, h5, h6 {
            color: #5c2d91; /* Color morado para todos los encabezados */
        }
    </style>
""", unsafe_allow_html=True)

# Lista visible en la barra lateral como botones
menu = [
    "Prueba piloto", 
    "Análisis de la aplicación de la prueba", 
    "Resultados de la prueba", 
    "Conclusiones y recomendaciones",
    "Power BI DEMO"
]

# Verificar si la selección ya está almacenada en session_state
if "seleccion" not in st.session_state:
    st.session_state.seleccion = "Prueba piloto"  # Valor por defecto

# Mostrar la lista de secciones en la barra lateral
st.sidebar.header("Selecciona una sección")
for item in menu:
    if st.sidebar.button(item):  # Usamos botones para cada opción
        st.session_state.seleccion = item  # Guardar la selección en session_state

# Recuperar la selección desde session_state
seleccion = st.session_state.seleccion

# Sección 1: Prueba piloto
if seleccion == "Prueba piloto":
    # Título principal de la aplicación
    st.title("IH Bank inicia pruebas piloto de la nueva web para inversiones.")
    st.image("https://www.hostinger.es/tutoriales/wp-content/uploads/sites/7/2019/08/precio-sitio-web.webp", caption="", use_column_width=True)
    st.header("Prueba Piloto")
    st.write("""
        En el marco de nuestro esfuerzo continuo por mejorar la experiencia digital de nuestros clientes, realizamos una prueba piloto para evaluar la eficacia de nuestro nuevo sitio web en el proceso de apertura de una inversión. El objetivo fue comparar la facilidad con la que nuestros clientes completan los cinco pasos del proceso (Start, Step 1, Step 2, Step 3 y Confirm) en dos versiones de la plataforma: el sitio web antiguo y el nuevo.

        Durante la prueba, dividimos a nuestros clientes en dos grupos. Un grupo utilizó el sitio antiguo y el otro probó el nuevo sitio. A través de esta prueba, buscamos determinar cuál de los dos sitios permite a los usuarios completar el proceso de inversión de manera más rápida y sencilla.
    """)
    st.subheader("Proceso general en el sitio web")
    st.image(r"Data/pasos.jpg", caption="", use_column_width=True)
    st.subheader("Innovaciones en la nueva web")
    textop1='''
        Resumen de las características:
        - Diseño visual limpio y moderno que reduce la complejidad visual.
        - Interacciones intuitivas con ayuda en cada paso.
        - Barra de progreso visible para guiar al usuario en el proceso de inversión.
        - Asistencia y soporte accesible mediante chat en vivo y asistentes automáticos.
        - Optimización móvil para que la experiencia sea consistente en cualquier dispositivo.
        - Transparencia y seguridad en cada paso del proceso.
        - Confirmación clara antes de finalizar la inversión.
        - Estas nuevas características están pensadas para ofrecer una experiencia de usuario sencilla, segura y eficiente, manteniendo el proceso de inversión accesible para todos los clientes.        
        '''
    st.write(textop1)
    st.image("https://images.ctfassets.net/wowgx05xsdrr/1q82ODPjZi6PohUPYicZp7/637c8f2125eb4eebd0f356d1bbfde224/ecommerce-investments-article-header.jpg?fm=webp&w=3840&q=75", caption="", use_column_width=True)

# Sección 2: Análisis de la aplicación de la prueba
elif seleccion == "Análisis de la aplicación de la prueba":
    st.title("¿A quiénes se aplicó la prueba piloto?")
    st.image("https://etf.dws.com/globalassets/_-knowledge/rebrand2023/3.7964_rebrand_visual_ai_bigdata_1889x480.jpg?width=1903&height=520&v=1459092475", caption="", use_column_width=True)
    st.header("A/B Test")
    st.write("""
        En esta sección, presentaremos un análisis detallado de las características de las muestras de clientes que participaron en la prueba piloto de nuestro nuevo sitio web. Las gráficas a continuación ofrecen una visión clara de los perfiles de los participantes en ambos grupos de prueba: aquellos que utilizaron el sitio web antiguo y los que probaron el nuevo diseño.

    Estos datos son fundamentales para comprender mejor el contexto de la prueba y asegurar que los resultados obtenidos sean representativos y válidos. Las gráficas incluyen información sobre variables clave como la edad, el tiempo como nuestros clientes y la frecuencia de uso de nuestros servicios en línea, lo que nos permite evaluar el impacto del nuevo sitio web en diferentes tipos de usuarios.
    """)
    df_final_completo, df_test_num, df_test_categ, df_control_num, df_control_categ, df_tasas=f.llama_datos() #llama a los datos cvs para graficar
    
    #Figura 1:
    st.subheader("Edad de los clientes")
    st.write("Registro general (AZUL)  /  Clientes en web nueva (VERDE)  /  Clientes en web antigua (ROSA)")
    fig_1=f.basic_stat_comparison([df_final_completo, df_test_num, df_control_num],'clnt_age', show_outliers=True, bins=20)
    st.pyplot(fig_1,use_container_width=True)
    st.write("En esta imagen se aprecia la similitud en los clientes seleccionados")
        
    #Figura 2:
    st.subheader("Número de inicios de sesión en el último semestre")
    st.write("Registro general (AZUL)  /  Clientes en web nueva (VERDE)  /  Clientes en web antigua (ROSA)")
    fig_2=f.basic_stat_comparison([df_final_completo, df_test_num, df_control_num],'logons_6_mnth', show_outliers=True, bins=7)
    st.pyplot(fig_2,use_container_width=True)
    st.write("En esta imagen se aprecia la similitud en los clientes seleccionados")

    #Figura 3:
    st.subheader("Años como clientes del banco")
    st.write("Registro general (AZUL)  /  Clientes en web nueva (VERDE)  /  Clientes en web antigua (ROSA)")
    fig_3=f.basic_stat_comparison([df_final_completo, df_test_num, df_control_num],'clnt_tenure_yr', show_outliers=True, bins=20)    
    st.pyplot(fig_3,use_container_width=True)
    st.write("En esta imagen se aprecia la similitud en los clientes seleccionados")
 
    
    st.header("¿La prueba piloto se aplicó correctamente?")
    st.write("Para comprobar que las características de los clientes que utilizaron el nuevo sitio web son similares a las de los clientes que utilizaron el sitio antiguo, se aplicaron pruebas de hipótesis. Algunos resultados se muestran a continuación:")
    
    st.write("💡 La edad media en las personas que probaron ambas páginas web es igual, con un 95% de confianza.")
    st.write("💡 La antigüedad media como clientes en las personas que probaron ambas páginas web es igual, con un 95% de confianza. ")
    st.write("💡 Con un 95% de confianza, el promedio de veces que los clientes iniciaron sesión duranta el último semestre en la página del banco es igual.")
    st.write("")
    st.subheader("Con pruebas como esta, se determina una correcta selección de los clientes en los que se aplicó la prueba piloto")

    st.write("Función generada para afectuar las pruebas de hipótesis")
    codigo2= """
    #----------------------- PRUEBA DE HIPÓTESIS CON T STUDENT -----------------------------------
    def t_test(df1, var1, df2, var2, hipo):

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


            """
    st.code(codigo2, language='python')
    st.subheader("🔎🔎Conoce a detalle el análisis efectuado y el código utilizado en este trabajo🔎🔎")
    st.markdown("💡--> Visita el [repositorio](https://github.com/gerardoJI/P1_Commodities_price) en GitHub.")
    st.image("https://static.vecteezy.com/system/resources/previews/039/342/550/non_2x/bank-service-employees-and-clients-financial-consultation-vector.jpg", caption="", use_column_width=True)



# Sección 3: Resultados de la prueba
elif seleccion == "Resultados de la prueba":
    st.title("Resultados de la prueba")
    st.image("https://pub.doubleverify.com/blog/content/images/2023/10/DVPS_BLOG_ABTEST_Inventory_Quality.png", caption="", use_column_width=True)
    st.title("¿Qué sitio web obtuvo mejores resultados?")
    st.write("Para interpretar los resultados de la prueba, se aplicaron un conjunto de técnicas a los datos. Una de ellas fue la búsqueda de correlación entre las variables. El objetivo fue detectar aquellas variables que se indrementaban o disminuían simultáneamente. ")
    
    df_final_completo, df_test_num, df_test_categ, df_control_num, df_control_categ, df_tasas=f.llama_datos() #llama a los datos cvs para graficar
    
    st.subheader("Análisis multivariable")
    st.write("Ejemplo de matriz de correlación (Pearson), para observar una posible correlación lineal entre los datos analizados. ")
    #Figura 4:
    fig_4= f.corr_map_pearson(df_test_num,-0.3,0.9)
    st.pyplot(fig_4,use_container_width=True)
    
    st.write("Ejemplo de matriz de correlación (Spearman), para observar una posible correlación monótona entre los datos analizados. ")
    #Figura 5:
    fig_5= f.corr_map_spearman(df_test_num,-0.6,0.9)
    st.pyplot(fig_5,use_container_width=True)

    st.write("Tras detectar una posible correlación entre dos variables, se observa su comportamiento a la par:")
    #Figura 5:
    fig_6= f.pair_plots(df_final_completo,'clnt_age',['step_2_time'],'Variation')
    st.pyplot(fig_6,use_container_width=True)
    st.write("En el gráfico anterior, se observa el la correlación entre la edad de los clientes y el tiempo que tardan en el paso 2, para ambos sitios web. Se interpreta que los clientes demoran más tiempo en este paso cuando usan el sitio web nuevo, que el antiguo.")

    st.header("Dataframe KPI's")
    df_final_completo, df_test_num, df_test_categ, df_control_num, df_control_categ, df_tasas=f.llama_datos() #llama a los datos cvs para graficar
    st.dataframe(df_tasas)

    st.write("Con el dataframe de KPI's, fue posible graficar el comportamiento general de nuestros clientes en ambos sitios web.")
    st.image(r"../Data/gra.jpg",  caption="", use_column_width=True)

    
    st.write("Para validar cuál de los sitios web ofrece mejores resultados, se aplicaron pruebas de hipótesis. Sus resultados se muestran a continuación:")
    
    st.subheader("Con un 95% de confianza se puede señalar que: ")
    st.write("💡 Los clientes no pasan del Step 1 con mayor frecuencia en el nuevo sitio web, con repespecto al antiguo.")
    st.write("💡 Los clientes no completan el Step 2 con mayor frecuencia en el nuevo sitio web, que en el antiguo.")
    st.write("💡 Los clientes no completan el Step 3 con mayor frecuencia en el nuevo sitio web, que en el antiguo.")
    st.write("💡 Los clientes terminan el proceso con mayor frecuencia en el nuevo sitio web, que en el antiguo.")
    st.write("")
    st.subheader("Con esta información, se determina que el nuevo sitio web tiene un mayor éxito que al antiguo al momento de cerrar el proceso de inversiones, pero registra una mayor cantidad de intentos para finalizar cada paso. Además, toma más tiempo a los usuarios completar los pasos, aproximadamente 7 segundos más en cada uno.  ")
    st.write("")
    st.write("")
    st.subheader("🔎🔎Conoce a detalle el análisis efectuado y el código utilizado en este trabajo🔎🔎")
    st.markdown("💡--> Visita el [repositorio](https://github.com/gerardoJI/P1_Commodities_price) en GitHub.")

# Sección 4: Conclusiones y recomendaciones
elif seleccion == "Conclusiones y recomendaciones":
    st.title("Conclusiones y recomendaciones")
    st.image("https://src.n-ix.com/uploads/2024/07/01/2e512188-d4cd-4ab3-a5f0-3210b3e03644.webp", caption="", use_column_width=True)
    st.subheader("Nuestras áreas de oportunidad")
    
    # Step 1: Verificación de Identidad
    st.subheader("🚩 1. Optimizar el Step 1: Verificación de Identidad")
    st.write("""
    **Problema:** Los clientes no pasan del Step 1 (Verificación de identidad) con mayor frecuencia en el nuevo sitio web, con respecto al antiguo.

    **Recomendaciones:**

    - **Simplificar la verificación de identidad:** Si el proceso de verificación de identidad es largo o requiere múltiples documentos, los usuarios pueden sentirse frustrados. Revisa si puedes reducir el número de pasos o documentos requeridos. Por ejemplo, permitir la carga de documentos con escaneo automático o utilizar opciones como la verificación facial o autenticación a través de aplicaciones de identidad digital puede acelerar este paso.
    - **Indicar claramente el propósito y la importancia de la verificación:** Los usuarios deben entender por qué es necesario este paso y cómo garantiza la seguridad de sus inversiones. Proporciona un mensaje claro de seguridad y protección de su información.
    - **Proporcionar asistencia inmediata:** Ofrece opciones de soporte en tiempo real, como un chat en vivo o una pregunta frecuente (FAQ) específica, para resolver dudas durante este paso crucial.
    - **Indicar el tiempo estimado:** Los usuarios deben saber cuánto tiempo tomará completar la verificación. Si es un proceso largo, un indicador de progreso visual que muestre el avance puede reducir la incertidumbre y ayudar a que los usuarios se sientan más cómodos.
    """)

    # Step 2: Selección del Tipo de Inversión
    st.subheader("🚩 2. Facilitar el Step 2: Selección del Tipo de Inversión")
    st.write("""
    **Problema:** Los clientes no completan el Step 2 (Selección del tipo de inversión) con mayor frecuencia en el nuevo sitio web.

    **Recomendaciones:**

    - **Personalizar las opciones de inversión:** Si las opciones de inversión no están adaptadas al perfil del usuario, puede ser confuso. Utiliza los datos proporcionados en la verificación de identidad para personalizar las opciones, ofreciendo solo aquellas que se ajusten a sus objetivos y nivel de riesgo. Esto puede ayudar a reducir el número de decisiones que el cliente debe tomar.
    - **Claridad en la presentación de opciones:** Presenta las opciones de inversión de manera visual y comparable, utilizando gráficos simples o tablas que resuman las características clave de cada opción (riesgos, beneficios, plazos, etc.).
    - **Asistencia contextual:** Ofrece explicaciones breves y claras junto a cada opción de inversión, como herramientas interactivas que expliquen los beneficios o riesgos asociados. Además, podrías permitir que el usuario simule el rendimiento de diferentes opciones antes de elegir.
    - **Indicadores de avance:** Si este paso tiene múltiples subopciones o categorías, utiliza un indicador de progreso para que los usuarios puedan ver cuánto falta para completar el paso, reduciendo el sentimiento de sobrecarga.
    """)

    # Step 3: Depósito de Fondos
    st.subheader("🚩 Optimizar el Step 3: Depósito de Fondos")
    st.write("""
    **Problema:** Los clientes no completan el Step 3 (Depósito de fondos) con mayor frecuencia en el nuevo sitio web.

    **Recomendaciones:**

    - **Métodos de pago claros y fáciles de usar:** Revisa si los métodos de depósito disponibles son accesibles y fáciles de usar. Asegúrate de ofrecer opciones populares y fáciles de entender, como transferencias bancarias instantáneas, tarjetas de débito/crédito, y carteras digitales (por ejemplo, PayPal, Apple Pay, Google Pay). Presenta estos métodos de manera destacada para que los usuarios elijan rápidamente.
    - **Reduce fricciones en el proceso de depósito:** El proceso de depósito debe ser sencillo y rápido. Si se requiere mucha información, como detalles bancarios o autenticación extra, simplifica estos pasos y proporciona instrucciones claras. Además, asegúrate de que los tiempos de carga sean mínimos.
    - **Garantizar transparencia en las comisiones:** Si hay tarifas o comisiones asociadas al depósito, hazlas visibles antes de que el usuario proceda con la acción. La falta de claridad sobre estos aspectos puede generar desconfianza y abandono.
    - **Ofrecer incentivos o promociones:** Para incentivar a los clientes a completar el depósito, podrías ofrecer promociones, como bonificaciones por el primer depósito, o recompensas adicionales si completan este paso rápidamente.
    """)

    # Step 4: Confirmación de la Inversión
    st.subheader("🚩 4. Fortalecer el Step 4: Confirmación de la Inversión")
    st.write("""
    **Problema:** No hay problemas aquí, ya que los clientes terminan el proceso con mayor frecuencia en el nuevo sitio que en el antiguo.

    **Recomendaciones:**

    - **Refuerzo positivo:** Asegúrate de que la confirmación sea clara y motivadora. Un mensaje como "¡Tu inversión está completa!" o "Felicidades, ahora eres un inversionista" genera una sensación de logro y refuerza la experiencia positiva.
    - **Resúmenes visuales:** En esta etapa, proporciona un resumen visual de la inversión (tipo de inversión, monto invertido, tiempo de inversión, etc.) para que los clientes puedan revisar rápidamente todos los detalles antes de hacer clic en "Confirmar".
    - **Confirmación inmediata y seguimiento:** Una vez que el usuario confirme la inversión, envíales un correo electrónico de confirmación y asegúrate de que puedan acceder a una página de agradecimiento con un resumen completo de su inversión y próximos pasos. Además, ofrece la opción de seguir invirtiendo o acceder a otros servicios en el sitio.
    """)

# Sección 5: DEMO de power bi 
elif seleccion == "Power BI DEMO":      
     st.title("Power BI")
     st.video(r"../Data/video_4k.mp4")

# Agregar los datos del autor y enlaces de redes sociales en la parte inferior de la barra lateral
st.sidebar.markdown("---")  # Línea separadora
st.sidebar.markdown("**Autores:**")

st.sidebar.markdown("📌 [Esteban Cristos LinkedIn](https://www.linkedin.com/in/esteban-daniel-cristos-muzzupappa-37b72635/)")
st.sidebar.markdown("📌 [Gerardo Jimenez LinkedIn](https://www.linkedin.com/in/gerardo-jimenez-islas/)")