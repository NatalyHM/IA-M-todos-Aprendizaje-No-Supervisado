import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

import os
print("Directorio de trabajo actual:", os.getcwd())

# Cargar el dataset desde un archivo CSV
df = pd.read_csv('C:/Users/natal/Escritorio local/AprendizajeNoSupervisado/Dataset_Estaciones.csv')

# Crear la nueva columna 'Trafico' como suma de 'Entradas' y 'Salidas'
df['Trafico'] = df['Entradas'] + df['Salidas']

# Preprocesamiento de datos: usar 'Trafico' y 'Rutas_Disponibles'
X = df[['Trafico', 'Rutas_Disponibles']]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Aplicar el modelo K-Means
kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# Visualización de los resultados con etiquetas
plt.figure(figsize=(12, 8))
sns.scatterplot(
    x=df['Trafico'],
    y=df['Rutas_Disponibles'],
    hue=df['Cluster'],
    palette='viridis',
    s=100
)

# Agregar etiquetas con el nombre de la estación
for i in range(df.shape[0]):
    plt.text(df['Trafico'][i], df['Rutas_Disponibles'][i], df['Estacion'][i], 
             fontsize=8, alpha=0.7, color='black')

# Configuración del gráfico
plt.title('Agrupamiento de Estaciones por Tráfico y Rutas Disponibles')
plt.xlabel('Tráfico (Entradas + Salidas)')
plt.ylabel('Rutas Disponibles')
plt.legend(title='Cluster')
plt.grid()
plt.show()

# Mostrar el dataframe con los resultados
print(df[['Estacion', 'Troncal', 'Trafico', 'Rutas_Disponibles', 'Cluster']].head())


output_file = 'C:/Users/natal/Escritorio local/AprendizajeNoSupervisado/resultado_clustering_transmilenio.csv'
df.to_csv(output_file, index=False)
print(f"Resultados guardados en: {output_file}")
