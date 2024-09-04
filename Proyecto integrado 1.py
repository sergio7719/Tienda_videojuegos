#!/usr/bin/env python
# coding: utf-8

# Hola &#x1F600;
# 
# Soy **Hesus Garcia**  como "Jesús" pero con H. Sé que puede ser confuso al principio, pero una vez que lo recuerdes, ¡nunca lo olvidarás! &#x1F31D;	. Como revisor de código de Triple-Ten, estoy emocionado de examinar tus proyectos y ayudarte a mejorar tus habilidades en programación. si has cometido algún error, no te preocupes, pues ¡estoy aquí para ayudarte a corregirlo y hacer que tu código brille! &#x1F31F;. Si encuentro algún detalle en tu código, te lo señalaré para que lo corrijas, ya que mi objetivo es ayudarte a prepararte para un ambiente de trabajo real, donde el líder de tu equipo actuaría de la misma manera. Si no puedes solucionar el problema, te proporcionaré más información en la próxima oportunidad. Cuando encuentres un comentario,  **por favor, no los muevas, no los modifiques ni los borres**. 
# 
# Revisaré cuidadosamente todas las implementaciones que has realizado para cumplir con los requisitos y te proporcionaré mis comentarios de la siguiente manera:
# 
# 
# <div class="alert alert-block alert-success">
# <b>Comentario del revisor</b> <a class=“tocSkip”></a>
# Si todo está perfecto.
# </div>
# 
# <div class="alert alert-block alert-warning">
# <b>Comentario del revisor</b> <a class=“tocSkip”></a>
# Si tu código está bien pero se puede mejorar o hay algún detalle que le hace falta.
# </div>
# 
# <div class="alert alert-block alert-danger">
# <b>Comentario del revisor</b> <a class=“tocSkip”></a>
# Si de pronto hace falta algo o existe algún problema con tu código o conclusiones.
# </div>
# 
# Puedes responderme de esta forma:
# <div class="alert alert-block alert-info">
# <b>Respuesta del estudiante</b> <a class=“tocSkip”></a>
# </div>
# 
# </br>
# 
# **¡Empecemos!**  &#x1F680;

# In[2]:


# importar librerías
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats as st


# In[3]:


#Leer el Dataframe
df_game = pd.read_csv('/datasets/games.csv')
df_game.info()


# In[4]:


#Se cambian las columnas a minusculas
df_game.columns = df_game.columns.str.lower()
display(df_game)


# <div class="alert alert-block alert-success">
# <b>Comentario del revisor</b> <a class=“tocSkip”></a>
# Buena organización inicial y correcta importación de librerías esenciales. El tratamiento inicial del dataframe establece una base sólida para el análisis. 
# </div>
# 

# In[5]:


#Validar en cuantos NAN aparecen en el df
df_game.isna().sum()


# In[6]:


#Se rellena la columna de year_of_release con la media del año para quitar los NAN
median_year = df_game['year_of_release'].median()
df_game['year_of_release'].fillna(median_year, inplace = True)


# In[7]:


#Se pone a tipo entero la fecha
df_game['year_of_release'] = df_game['year_of_release'].astype('int')
df_game['year_of_release']


# In[9]:


#Reemplazar los NAN de las columnas "name" , "genre" y "rating"
df_game['name'].fillna('Unknown', inplace = True)
df_game['genre'].fillna('Unknown', inplace = True)
df_game['rating'].fillna('Unknown', inplace = True)


# In[10]:


#Validamos en que filas de la columna "user_score" estan los tbd
df_game[df_game['user_score'] == 'tbd']


# In[11]:


#Se reemplazan los tbd por none, y cambiamos el tipo de objetc a float 
df_game['user_score'] = df_game['user_score'].replace("tbd", np.nan).astype(float)
df_game['user_score']


# In[12]:


#Se reempalza los valores NAN  con la media  para la columna "critic_score"
mean_critical = df_game['critic_score'].mean()
df_game['critic_score'].fillna(mean_critical, inplace = True)


# In[13]:


#Se pone a tipo entero la la columan de "critic_score"
df_game['critic_score'] = df_game['critic_score'].astype('int')


# In[14]:


df_game['total_sales']= df_game['na_sales'] + df_game['eu_sales'] + df_game['jp_sales'] + df_game['other_sales']
df_game.head()


# Primero se analizo todo el df y vimos que existian varios valores ausentes, lo que se hizo fue reemplazar esos valores de cada columna segun su tipo.
# 
# 1.- En la columna "year_of_release" se relleno las fechas faltantes con la media del año y se paso de tipo floar a int. 
# 2.- Para las columnas de tipo object se relleno con "Unknown" para no tener datos vacios
# 3.- Para las columna  "user_score" se remplazara los "tbd" por Nan, ya que si los reemplazamos con 0 si afecta nuestra media.
# 4.- De los valores ausentes en la columna de "user_score", se puede deber a que los TBD son de los juegos que no han se han vendido mucho
# 5.- Y por ultimo se creo una columan llamada "total sales" en donde se plasma la venta general de cada juego en todas la regiones.
# 

# <div class="alert alert-block alert-success">
# <b>Comentario del revisor</b> <a class=“tocSkip”></a>
# Excelente atención a los valores faltantes y la conversión de tipos de datos. La estrategia de reemplazo por la mediana y 'Unknown' es adecuada para mantener la integridad del dataset. 
# </div>
# 

# ## Analisis de datos

# In[15]:


# Se agrupa la info entre el año lanzado y el numero dejuegos para validar en que año hubo mas lanzamientos
group_game = df_game.groupby('year_of_release')['name'].count()
group_game


# In[16]:


#Se valida que años fueron con menos lanzamientos 
for year, count in group_game.items():
    if count <10:
        print(f"Games for year {year} is not significant")


# In[17]:


#Se sacan las plataformas con mayor ventam en este caso es el top 10
platform_sales = df_game.groupby('platform')['total_sales'].sum()
top_platforms = platform_sales.nlargest(n= 20)
df_top_platforms = df_game[df_game['platform'].isin(top_platforms.index)]


# In[18]:


#Se saca la suma total ventas entre año y plataforma
grouped_data = df_top_platforms.groupby(['year_of_release', 'platform'])['total_sales'].sum()
sales_percentage = grouped_data.unstack(level=1).apply(lambda x: 100 * x / x.sum(), axis=1)


# In[19]:


# Print the results
print('Platforms with the highest total sales:')
print(top_platforms)


# In[20]:


#Tiempo en el que tarda en lanzar una nueva consola
disappeared_platforms = platform_sales[platform_sales == 0].index
platform_life = (df_game['year_of_release'].max() - df_game['year_of_release'].min()) / (len(disappeared_platforms) + len(df_game[df_game['platform'].isin(df_game['platform'].unique()[-5:])]))


# In[21]:


print('\nAverage platform life:', platform_life, 'years')


# 1.-Lo que se pudo llegar de conclusion en esta seccion es que el lider de venta a nivel general es el PS2 con un venta total de "1255.77", despues le sigue el xbox con un monto de "971.42" y en tercer lugar el PS3 con un total de "939.65".
# 
# 2.- Las plataformas que van en decadencia son las de los 1990 ya que no se renovaron en el mercado y por ende son las que ya no tienen tanto valor a diferencia de PS y Xbox
# 
# 3.- Y aproximadamente se tardan un año en sacar una nueva plataforma al mercado.

# <div class="alert alert-block alert-success">
# <b>Comentario del revisor</b> <a class=“tocSkip”></a>
# Muy bien por identificar los años con mayor lanzamiento y por realizar un análisis detallado de las ventas por plataforma. Este enfoque aporta insights valiosos sobre tendencias del mercado.
# </div>
# 

# ## Diagrama de Caja

# In[23]:


sns.boxplot(x='platform', y='total_sales', data=df_top_platforms)
plt.xlabel('Platform')
plt.ylabel('Global Sales')
plt.title('Boxplot of Global Sales by Platform (Top 10)')
plt.show()


# In[24]:


#Sacar las ventas promedio de cada plataforma
average_sales = df_game.groupby('platform')['total_sales'].mean()
for platform,average_sales in average_sales.items():
    print(f"Average sales for platform '{platform}': {average_sales}")


# NO hay mucha diferencia en las ventas en las primeras 5 consolas, en la sexta ya se puede notar un cambio en la ventas esto nos dice que posiblemente las consolas que no tiene ya un mayor margen de venta puede que sea porque no llego a lo esperado dicha consola o no fue de gran interes hacia los consumidores.
# 
# Que enfocandonos hacia el 2017, le apostaria mas a las consolas de PS, Xbox y PC.
# 
# 

# In[25]:


score_ps2 = df_game[df_game['platform'] == 'PS2']
# Se crea la grafica entre las reseñas y las ventas
plt.scatter(score_ps2['user_score'], score_ps2['total_sales'])
plt.xlabel('User Score')
plt.ylabel('Total Sales')
plt.title('Scatter Plot of User Score vs. Total Sales for PS2 Games')
plt.show()




# In[26]:


correlation = score_ps2['user_score'].corr(score_ps2['total_sales'])
print('Correlacion entre user_score y total_sales:', correlation)


# In[27]:


# Se crea la grafica entre el critic_score  y las ventas
plt.scatter(score_ps2['critic_score'], score_ps2['total_sales'])
plt.xlabel('Critic Score')
plt.ylabel('Total Sales')
plt.title('Scatter Plot of Critic Score vs. Total Sales for PS2 Games')
plt.show()


# In[28]:


correlation = score_ps2['critic_score'].corr(score_ps2['total_sales'])
print('Correlacion entre  critic_score y total_sales:', correlation)


# <div class="alert alert-block alert-success">
# <b>Comentario del revisor</b> <a class=“tocSkip”></a>
# El uso de diagramas de caja para visualizar las ventas globales por plataforma y el cálculo de correlaciones demuestra un análisis meticuloso. <b>Has calculado la correlación como se solicitó en el brief del proyecto</b>, proporcionando una comprensión clara de la relación entre puntuaciones y ventas.
# </div>
# 

# <div class="alert alert-block alert-warning">
# <b>Comentario del revisor</b> <a class=“tocSkip”></a>
# Para ambos gráficos te aconsejaría utilizar tamaños más adecuados. Utilizando fig_size() . Esto te permitirá que sean más legibles. Recuerda hacer estas correciones en los proyectos que cargues a tu portafolio.
# </div>
# 

# ## Las conclusiones que se pueden tomar en este punto es que si hay es importante la calificacion que den ya que esto determine si un juego o plataforma sera retentable y poder asi sacar proyecciones hacia el futuro y ver que juevo siempre estara en tendencia .

# In[29]:


# Se filtra todas la plafaormas quitando a PS2
gta_game = df_game[df_game['name'] == 'Grand Theft Auto: San Andreas']
# Compare total sales
print('Total sales comparison:')
print(f'GTA: San Andreas (PS2): {df_game[df_game["name"] == "Grand Theft Auto: San Andreas"]["total_sales"].values[0]}')
print(f'GTA: San Andreas (Other platforms): {gta_game["total_sales"].sum()}')


# ### Se puede ver que las ventas que tuvo Grand Theft Auto: San Andreas en otras consolas es del 23.85 y que solo el PS2 tuve 20.81, esto nos dice que  el juego con mayor ventas gracias al critic_score fue el GTA, esto haciendo al comparativa con PS2 ya que las demas consolas que fueron(XB,PC y X360) unidas superaron a PS2

# In[30]:


# Se agruapa para sacar la info del genero mas vendido de cada plataforma
genre_data = df_game.groupby(['genre', 'platform'])['total_sales'].sum()
genre_sales = genre_data.unstack(level=1).apply(lambda x: 100 * x / x.sum(), axis=1)

#Con el ciclo for se va itrando para saber el genero mas vendido
most_profitable_genres = {}
for platform, sales in genre_sales.items():
  most_profitable_genres[platform] = sales.sort_values(ascending=False).index[0]


sales_by_genre = df_game.groupby('genre')['total_sales'].sum().sort_values(ascending=True)
sales_by_genre.plot(kind='barh')
plt.title('Total Sales by Genre')
plt.xlabel('Total Sales')
plt.show()


# In[31]:


top_5_jp = df_game.groupby('platform')['jp_sales'].sum().nlargest(5)

# Se crea la grafica 
top_5_jp.plot(kind='bar')

plt.title('Top 5 Platforms by JP Sales')
plt.xlabel('Platform')
plt.ylabel('JP Sales')


plt.show()


# In[32]:


top_5_na = df_game.groupby('platform')['na_sales'].sum().nlargest(5)

# Se crea la grafica 
top_5_na.plot(kind='bar')

plt.title('Top 5 Platforms by NA Sales')
plt.xlabel('Platform')
plt.ylabel('NA Sales')

plt.show()


# In[33]:


top_5_eu = df_game.groupby('platform')['eu_sales'].sum().nlargest(5)

# Se crea la grafica 
top_5_eu.plot(kind='bar')


plt.title('Top 5 Platforms by EU Sales')
plt.xlabel('Platform')
plt.ylabel('EU Sales')

plt.show()


# In[34]:


#Se se crea el df con la info de genero y las ventas del pais
top5_gen_jp = df_game.groupby('genre')['jp_sales'].sum().nlargest(5)

x_labels = top5_gen_jp.index.to_list()
y_values = top5_gen_jp.values.tolist()


plt.figure(figsize=(6, 6))
plt.bar(x_labels, y_values)
plt.xlabel('Genre')
plt.ylabel('JP Sales (millions)')
plt.title('Top 5 Genres by JP Sales')
plt.xticks(rotation=45, ha='right')


plt.tight_layout()
plt.show()


# In[35]:


#Se se crea el df con la info de genero y las ventas del pais
top5_gen_na = df_game.groupby('genre')['na_sales'].sum().nlargest(5)


x_labels = top5_gen_na.index.to_list()
y_values = top5_gen_na.values.tolist()

plt.figure(figsize=(6, 6))
plt.bar(x_labels, y_values)
plt.xlabel('Genre')
plt.ylabel('NA Sales (millions)')
plt.title('Top 5 Genres by NA Sales')
plt.xticks(rotation=45, ha='right')


plt.tight_layout()
plt.show()


# In[36]:


#Se se crea el df con la info de genero y las ventas del pais
top5_gen_eu = df_game.groupby('genre')['eu_sales'].sum().nlargest(5)


x_labels = top5_gen_na.index.to_list()
y_values = top5_gen_na.values.tolist()


plt.figure(figsize=(6, 6))
plt.bar(x_labels, y_values)
plt.xlabel('Genre')
plt.ylabel('EU Sales (millions)')
plt.title('Top 5 Genres by EU Sales')
plt.xticks(rotation=45, ha='right')


plt.tight_layout()
plt.show()


# ### Se puede observar en en las 3 regiones si hay una diferencia de uso de consolas entre los consumidores, la unica paltaforma que se repite es  el PS, debido a que tiene una grna variedad de juegos
# 
# 2.- En la comparacion por genero, se puede apreciar que si hay preferencia por 4 generos en particular, estos cuatro generos que se repiten en las 3 regiones son (Action, Sport, Plataform y Misc) lo que nos indica que si alguna plataforma saca un juego con alguno de estos generos se podra obtener una buena venta en comparacion a otros generos que no hay mucha demanda.
# 
# 

# ### Prueba de hipotesis
# 

# In[46]:


#Hipótesis nula:**

# H0: μ_Xone = μ_PC

#**Hipótesis alternativa:**

#H1: μ_accion ≠ μ_sport

alpha = 0.05
#Se extrae el info de ada plataforma en este caso Xboxone y pc en relacion a el user_score
xbox_one_ratings = df_game[df_game['platform'] == 'XOne']['user_score']
pc_ratings = df_game[df_game['platform'] == 'PC']['user_score']



t_stat, p_value = st.ttest_ind(xbox_one_ratings, pc_ratings)

if p_value < alpha:
  print('Se Rechaze la hipótesis nula.\nExiste evidencia significativa de que las calificaciones promedio de los usuarios para Xbox One y PC son diferentes.')
else:
  print('No se rechaza la hipótesis nula.\nNo hay evidencia suficiente para concluir que las calificaciones promedio de los usuarios para Xbox One y PC sean diferentes..')



# In[45]:


action_ratings = df_game[df_game['genre'] == 'Action']['user_score']
sports_ratings = df_game[df_game['genre'] == 'Sports']['user_score']

t_stat, p_value = st.ttest_ind(action_ratings, sports_ratings)


if p_value < alpha:
  print('Se Rechaza la hipótesis nula.\n Existe evidencia significativa de que las calificaciones promedio de los usuarios para los géneros de acción y deportes son diferentes..')
else:
  print('No rechazar la hipótesis nula.\n No hay evidencia suficiente para concluir que las calificaciones promedio de los usuarios para los géneros de acción y deportes sean diferentes..')


# <div class="alert alert-block alert-warning">
# <b>Comentario del revisor</b> <a class=“tocSkip”></a>
# La ejecución de las pruebas de hipótesis está bien realizada, pero sería beneficioso profundizar en la interpretación de los resultados para entender mejor sus implicaciones en el contexto del análisis de videojuegos.
# </div>
# 

# ### Conclusiones
# 
# *Se puede observar que si afecta el rating que puedan dar tanto los usuarios como el critical_score
# *Que para este ejercicio PlayStation tuvo una mayo demanda en la venta de plataformas en comparacion con sus principales
#   compentidores en este caso (Xbox,PC y Wii).
# 
# *Que la mayoria de usuarios prefiere juegos de accion en relaccion hacia otras generos.
# 
# * Que los años 2007, 2008 y 2009 fueron años con el mayor numero de lanzmientos de videojuegos para las principales
# plataformas del mercado.
# 
# *Que dentro de unos años la disputa estara entre consolas vs PC.
# 
# 
# 

# <div class="alert alert-block alert-success">
# <b>Comentario del revisor</b> <a class=“tocSkip”></a>
# Has sintetizado eficazmente los hallazgos clave de tu análisis. La identificación de los años con mayor número de lanzamientos y la preferencia de género entre los usuarios son insights particularmente interesantes que resaltan el éxito de tu investigación.
# </div>
# 
