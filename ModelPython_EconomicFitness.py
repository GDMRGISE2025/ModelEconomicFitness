# -- coding: utf-8 --
"""
Created on Wed Dec  4 14:41:02 2024

@author: GENESIS DELGADO
"""

import numpy as np  # Biblioteca para manejar operaciones matemáticas y matrices.
import pandas as pd  # Biblioteca para trabajar con datos tabulares y archivos Excel.

# Cargar datos de la matriz de exportaciones desde un archivo Excel.
# 'header=None' indica que el archivo no tiene encabezados en las filas o columnas.
# '.values' convierte el DataFrame de pandas en una matriz de NumPy.
EXP = pd.read_excel().values #Insertar ruta de archivo de base de datos

# Inicialización de variables
SumPaises = 0  # Variable para almacenar la suma total de exportaciones por países.
SumTotal = 0   # Variable para la suma total de todas las exportaciones.
SumProd = 0    # Variable para almacenar la suma total de exportaciones por productos.

# Inicialización de vectores
Vc = np.zeros(EXP.shape[0])  # Vector para almacenar la suma de exportaciones por país.
Vp = np.zeros(EXP.shape[1])  # Vector para almacenar la suma de exportaciones por producto.

# Inicialización de matrices
VCR = np.zeros_like(EXP, dtype=float)  # Matriz para almacenar los valores de VCR.
Mcp = np.zeros_like(EXP, dtype=int)    # Matriz para almacenar los valores binarios de Mcp.

# Calcular suma por países
for i in range(EXP.shape[0]):  # Iterar sobre cada fila (país).
    Vc[i] = np.sum(EXP[i, :])  # Sumar todas las exportaciones del país i.
    SumPaises += Vc[i]         # Acumular la suma total por países.

# Calcular suma por productos
for j in range(EXP.shape[1]):  # Iterar sobre cada columna (producto).
    Vp[j] = np.sum(EXP[:, j])  # Sumar todas las exportaciones del producto j.
    SumProd += Vp[j]           # Acumular la suma total por productos.

# Calcular la suma total de exportaciones
SumTotal = np.sum(Vc)  # La suma total es igual a la suma de todas las exportaciones por países.

# Calcular la matriz VCR
for f in range(EXP.shape[0]):  # Iterar sobre cada fila (país).
    for c in range(EXP.shape[1]):  # Iterar sobre cada columna (producto).
        # Fórmula de VCR: (exportación del país y producto) / (promedio nacional del país)
        # dividido entre (promedio global del producto).
        VCR[f, c] = (EXP[f, c] / Vc[f]) / (Vp[c] / SumTotal)

# Calcular la matriz Mcp
# Si el valor en VCR es >= 1, entonces Mcp es 1; de lo contrario, es 0.
Mcp = (VCR >= 1).astype(int)

# Configuración para el cálculo de Fitness y Complejidad
N = 1000  # Número de iteraciones.
FcN = np.ones((Mcp.shape[0], N))  # Fitness para cada país, inicializado con unos.
QpN = np.ones((Mcp.shape[1], N))  # Complejidad para cada producto, inicializado con unos.
fcn = np.ones((Mcp.shape[0], N))  # Fitness normalizado.
qpn = np.ones((Mcp.shape[1], N))  # Complejidad normalizada.

# Iteraciones para calcular Fitness y Complejidad
for k in range(1, N):  # Empezar desde la segunda iteración.
    # Calcular el Fitness (FcN) para cada país.
    for i in range(Mcp.shape[0]):  # Iterar sobre cada país.
        FcN[i, k] = np.sum(Mcp[i, :] * qpn[:, k-1])  # Sumar contribuciones ponderadas por complejidad.

    # Normalizar el Fitness.
    fcn[:, k] = FcN[:, k] / (np.mean(FcN[:, k]))  # Normalización respecto al promedio.

    # Calcular la Complejidad (QpN) para cada producto.
    for j in range(Mcp.shape[1]):  # Iterar sobre cada producto.
        QpN[j, k] = 1 / np.sum(Mcp[:, j] * (1 / fcn[:, k-1]))  # Sumar contribuciones inversas de fitness.

    # Normalizar la Complejidad.
    qpn[:, k] = QpN[:, k] / (np.mean(QpN[:, k]))  # Normalización respecto al promedio.

# Verificar la convergencia del Fitness
A = []  # Lista para almacenar los valores convergentes de Fitness.
tol = 1e-6  # Tolerancia para la convergencia.
for i in range(fcn.shape[0]):  # Iterar sobre cada país.
    for j in range(1, fcn.shape[1]):  # Iterar sobre las iteraciones (a partir de la segunda).
        if abs(fcn[i, j] - fcn[i, j-1]) <= tol:  # Verificar si el cambio entre iteraciones es menor que la tolerancia.
            A.append((fcn[i, j], j))  # Almacenar el valor convergente y el número de iteración.
            break  # Pasar al siguiente país una vez encontrada la convergencia.

# Verificar la convergencia de la Complejidad
B = []  # Lista para almacenar los valores convergentes de Complejidad.
for i in range(qpn.shape[0]):  # Iterar sobre cada producto.
    for j in range(1, qpn.shape[1]):  # Iterar sobre las iteraciones (a partir de la segunda).
        if abs(qpn[i, j] - qpn[i, j-1]) <= tol:  # Verificar si el cambio entre iteraciones es menor que la tolerancia.
            B.append((qpn[i, j], j))  # Almacenar el valor convergente y el número de iteración.
            break  # Pasar al siguiente producto una vez encontrada la convergencia.

# Exportar resultados 
with pd.ExcelWriter('Resultados Economic Fitness.xlsx') as writer:
    # Escribir los resultados de Fitness en una hoja
    pd.DataFrame(A).to_excel(writer, sheet_name='Fitness Convergence', index=False, header=False)
    
    # Escribir los resultados de Complejidad en otra hoja
    pd.DataFrame(B).to_excel(writer, sheet_name='Complexity Convergence', index=False, header=False)
