import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D  # Importar para gráficos 3D

import os
print("Directorio de trabajo actual:", os.getcwd())

# Cargar el dataset desde un archivo CSV
df = pd.read_csv('C:/Users/natal/Escritorio local/AprendizajeNoSupervisado/Dataset_Estaciones.csv')

# Crear la nueva columna 'Trafico' como suma de 'Entradas' y 'Salidas'
df['Trafico'] = df['Entradas'] + df['Salidas']

# Preprocesamiento de datos: usar 'Entradas', 'Salidas' y 'Rutas_Disponibles'
X = df[['Entradas', 'Salidas', 'Rutas_Disponibles']]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Cambiar el número de clusters
n_clusters = 4  # Cambia este valor al número de clusters que desees

# Aplicar el modelo K-Means con el número de clusters deseado
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# Visualización 3D de los resultados
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Graficar los puntos en 3D con colores de acuerdo a los clusters
scatter = ax.scatter(df['Entradas'], df['Salidas'], df['Rutas_Disponibles'], 
                     c=df['Cluster'], cmap='viridis', s=100)

# Agregar etiquetas con el nombre de la estación
for i in range(df.shape[0]):
    ax.text(df['Entradas'][i], df['Salidas'][i], df['Rutas_Disponibles'][i], df['Estacion'][i], 
            fontsize=8, alpha=0.7, color='black')

# Configuración del gráfico
ax.set_title('Agrupamiento de Estaciones en 3D por Entradas, Salidas y Rutas Disponibles')
ax.set_xlabel('Entradas')
ax.set_ylabel('Salidas')
ax.set_zlabel('Rutas Disponibles')

# Agregar la leyenda de colores
plt.colorbar(scatter)  # Muestra la barra de color para los clusters
ax.legend(title='Cluster')  # Esta línea añade la leyenda

# Mostrar la gráfica
plt.show()

# Mostrar el dataframe con los resultados
print(df[['Estacion', 'Troncal', 'Entradas', 'Salidas', 'Rutas_Disponibles', 'Cluster']].head())

# Guardar los resultados en un archivo CSV
output_file = 'C:/Users/natal/Escritorio local/AprendizajeNoSupervisado/resultado_clustering_transmilenio3D.csv'
df.to_csv(output_file, index=False)
print(f"Resultados guardados en: {output_file}")

# Ver las estadísticas descriptivas por cada grupo
cluster_stats = df.groupby('Cluster')[['Entradas', 'Salidas']].describe()

# Mostrar el resumen de las estadísticas descriptivas
print("\nEstadísticas descriptivas por grupo (Cluster):")
print(cluster_stats)

# Ver las estadísticas descriptivas por cada grupo
cluster_stats2 = df.groupby('Cluster')[['Rutas_Disponibles']].describe()

# Mostrar el resumen de las estadísticas descriptivas
print("\nEstadísticas descriptivas por grupo (Cluster):")
print(cluster_stats2)

# Ver la cantidad de estaciones en cada grupo
cluster_counts = df['Cluster'].value_counts()
print("\nCantidad de estaciones por grupo (Cluster):")
print(cluster_counts)
