# Título Programa: RED ADALINE.
# Fecha: 05-diciembre-2022
# Autor: Renteria Arriaga Josue
# Descripcion: El codigo fue recuperado por las presentaciones del Dr. Juan Humberto Sossa Azuela.

import numpy as np
import matplotlib.pyplot as plt
import time
#from func import *

# Función de costo -- MSE
def CostFunction(t, y):
	cost = np.sum(0.5 * (t-y) ** 2)
	return cost

# Lineal
def lineal(a):
	s = a
	return s

# DELTA Rule
def ReglaDelta(x, t, w_i, alpha):
	err_vector = []
	weights = []

	# Producto punto
	a = np.dot(x, w_i)

	# Función lineal
	y = lineal(a)

	# print('y: {}',format(y))
	err = CostFunction(t, y)
	err_vector.append(err)
	epoch = 0

	# while(sum(y != t)):
	epochs = 1600

	for i in range(1500):
		epoch += 1
		
		for i in range(np.shape(x)[0]):

			# Producto punto
			a = np.dot(x[i], w_i)

			# Función lineal
			y[i] = lineal(a)

			x_p = np.reshape(x[i], (len(w_i), 1))
			w_n = w_i + alpha * (t[i] - y[i]) * x_p
			w_i = w_n

		err = 0.25 * CostFunction(t, y)
		err_vector.append(err)
		weights.append(w_i)

	return w_i, weights, err_vector

# Gráfica del error
def ErrGraph(err_vector):
	plt.figure(0)
	plt.plot(err_vector, linewidth = 2)
	plt.xlabel('Épocas')
	plt.ylabel('Magnitud de Error')
	plt.title('Gráfica de Error: REGLA DELTA')
	plt.scatter(len(err_vector) - 1, 0, color = 'r', s = 200, marker = 'o', alpha = 0.4)
	plt.show()

# Funcion para concatenar el Bias.
def ConBias(x):
	bias = -np.ones((np.shape(x)[0], 1))
	x = np.concatenate([x, bias], axis = 1)
	return x

# Funcion para mostrar los pesos.
def MostrarPesos(w_f):
	print('\nPesos Finales: ')
	for i in range(1):
		res = str(w_f)
		print (res)

# Funcion que muestra los resultados esperados.
def SalidaT(x, w_f, t):
	a = np.dot(x, w_f)
	y = lineal(a)
	print('\nREGLA DELTA')
	print('Meta: Predicho:')
	for i in range(len(y)):
		res = str(t[i]) + '--------' + str(y[i])
		print(res)

# Valores de entradas y sus respectivas salidas (Para entrenar la Red).
x = np.array([[2, 0, 0], [4, 0, 0], [3, 1, 0], [4, 2, 1], [6, 0, 0], [5, 1, 0], [6, 2, 1], [4, 2, 0], [5, 3, 1], [6, 4, 2], [6, 6, 3]])
t = np.array([[1], [2], [2], [2], [3], [3], [3], [3], [3], [3], [3]])
alpha = 0.01

# Agregamos a la Matriz de las x nuestro Bias con valor de -1. 
x = ConBias(x)

# Planteamos unos Pesos para entrenar (Deben ser Bajos).
w_i = np.array([[0.5], [0.2], [0.3], [0.1]])

# Aplicamos la funcion de la Regla del Percertron.
w_f, weights, err_vector = ReglaDelta(x, t, w_i, alpha)

# Impresion de nuestros pesos Finales Finales y nuestras salidas.
MostrarPesos(w_f)
SalidaT(x, w_f, t)

# Valores de entrada para ver sus salidas y sus respectivas salidas (No son necesarias las Salidas).
x = np.array([[6, 6, 2], [8, 8, 4], [8, 8, 3], [10, 8, 4], [12, 12, 6], [10, 4, 1]])
t = np.array([[4], [4], [5], [5], [6], [6]])

# Agregamos a la Matriz de las x nuestro Bias con valor de -1. 
x = ConBias(x)

# Resultados de nuestras salidas.
SalidaT(x, w_f, t)

# Graficamos el Error
ErrGraph(err_vector)