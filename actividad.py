import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linprog

# Definimos las restricciones
# 4x1 + 6x2 <= 40  -> 4x1 + 6x2 - 40 <= 0
# 10x1 + 20x2 <= 200  -> 10x1 + 20x2 - 200 <= 0

# Coeficientes de las restricciones
A = [[4, 6], [10, 20]]  # Matriz de coeficientes
b = [40, 200]  # Límite de las restricciones

# Definimos la función objetivo (maximizar la cantidad de entrenamientos)
# Queremos maximizar 1*x1 + 1*x2 (es decir, el número de entrenamientos)
c = [-1, -1]  # Negativo porque linprog minimiza (así que se maximiza 1*x1 + 1*x2)

# Resolver el problema de programación lineal
resultado = linprog(c, A_ub=A, b_ub=b, method='highs')

# Extraemos la solución
x1_optimo = resultado.x[0]
x2_optimo = resultado.x[1]
max_entrenamientos = -resultado.fun  # Convertimos el valor máximo (lo calculamos como negativo)

# Graficamos las restricciones
x = np.linspace(0, 10, 400)
y_tiempo = (40 - 4*x) / 6  # Restricción de tiempo: 4x1 + 6x2 <= 40
y_almacenamiento = (200 - 10*x) / 20  # Restricción de almacenamiento: 10x1 + 20x2 <= 200

# Configuramos el gráfico
plt.figure(figsize=(8, 6))

# Graficar las restricciones
plt.plot(x, y_tiempo, label=r'$4x_1 + 6x_2 \leq 40$', color='blue')
plt.plot(x, y_almacenamiento, label=r'$10x_1 + 20x_2 \leq 200$', color='green')

# Rellenamos la región factible
plt.fill_between(x, np.minimum(y_tiempo, y_almacenamiento), where=(x >= 0), color='yellow', alpha=0.3)

# Graficamos la solución óptima
plt.scatter(x1_optimo, x2_optimo, color='red', zorder=5)
plt.text(x1_optimo + 0.2, x2_optimo, f'({x1_optimo:.2f}, {x2_optimo:.2f})', color='red')

# Etiquetas y título
plt.xlim(0, 10)
plt.ylim(0, 10)
plt.xlabel('Entrenamientos Modelo 1')
plt.ylabel('Entrenamientos Modelo 2')
plt.title('Optimización de Recursos para Entrenamiento de Modelos')
plt.axhline(0, color='black',linewidth=1)
plt.axvline(0, color='black',linewidth=1)

# Leyenda
plt.legend()

# Mostrar el gráfico
plt.grid(True)
plt.show()

# Imprimir la solución óptima
print(f"Solución óptima: Entrenamientos Modelo 1: {x1_optimo:.2f}, Entrenamientos Modelo 2: {x2_optimo:.2f}")
print(f"Máximo uso de recursos (entrenamientos totales): {max_entrenamientos:.2f}")
