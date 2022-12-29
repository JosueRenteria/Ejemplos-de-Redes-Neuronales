# Título Programa: RELU.
# Fecha: 05-diciembre-2022
# Autor: Renteria Arriaga Josue
# Descripcion: El codigo fue recuperado por las presentaciones del Dr. Juan Humberto Sossa Azuela.

import numpy as np
import matplotlib.pyplot as plt

# Funcion max para la derivada.
def max(a): 
	return np.maximum(a, 0)

# Funcion para la derivada de la ReLu.
def deriv_relu(a): 
	return a > 0

# Funcion de perdida o de Costo.
def MSE(y, t):
		return np.sum(0.5 * (t-y) ** 2)

# Funcion que grafica la figura (Frontera de Desicion). 
def DecBoundaries(w_f, x, t):
	plt.figure(1)
	plt.xlim([-1.0, 2.0])
	plt.ylim([-1.0, 2.0])
	patterns = np.unique(t)
	for patt in patterns:
		pos = np.where(patt == t)[0]
		if patt == -1:
			plt.scatter(x[pos, 0], x[pos, 1], color = 'g', s = 200, marker ='o', alpha = 0.4)
		else:
			plt.scatter(x[pos, 0], x[pos, 1], color = 'b', s = 200, marker ='x', alpha = 0.4)

	x1 = np.linspace(-1, 2)
	x2 = w_f[2] / w_f[1] - (x1 * w_f[0]) / w_f[1]
	plt.plot(x1, x2, 'red', linewidth = 2)
	plt.title('FRONTERAS DE DECISIÓN: REGLA DELTA')
	plt.show()

# Clase donde se inicializa la ReLu.
class Relu():
	# Constructor de los patrones de entrada.
	def __init__(self, xs):
		bias = -np.ones((np.shape(xs)[0], 1))
		self.xs = np.concatenate([xs, bias], axis = 1)
		self.n_patrones = np.shape(self.xs)[1]
		self.n_muestras = np.shape(self.xs)[0]
		self.ws = np.random.rand(self.n_patrones, 1)
		print(f'Pesos iniciales: \n {self.ws}\n')

	# FAuncion del proceso de la regla Delta.
	def regla_delta(self, t, alpha, epocas):
		a = np.dot(self.xs, self.ws)
		y = max(a)
		e_actual = MSE(y, t)
		e_anterior = 0

		for epoca in range(epocas):
			for i in range(self.n_muestras):
				a = np.dot(self.xs[i], self.ws)
				y[i] = max(a)
				x_p = np.reshape(self.xs[i], (self.n_patrones, 1))
				w_n = self.ws - alpha * (y[i] - t[i]) * deriv_relu(y[i]) * x_p
				self.ws = w_n
			e_actual = MSE(y, t)/self.n_muestras
		print(f'Epocas: {epocas}')

	# Funcion que muestra los pesos.
	def mostrar_pesos(self):
		np.set_printoptions(formatter={'float_kind':'{:f}'.format})
		print(self.ws)

	# Funcion que devuelve los pesos.
	def pesos(self):
		return self.ws

	# Funcion que devuelve las x.
	def equis(self):
		return self.xs

# Datos para el problema de AND,
x = np.array([[0,0], [0,1], [1,0], [1,1]])
t = np.array([[-1], [-1], [-1], [1]])

# Mandamos a llamar a los procesos de ReLu.
modelo_relu = Relu(x)
alpha = 0.02

modelo_relu.regla_delta(t, alpha, 20)
modelo_relu.mostrar_pesos()

# Sacamos los xs y ws.
ws = modelo_relu.pesos() 
xs = modelo_relu.equis()

# Graficamos la frontera de Desicion. 
DecBoundaries(ws, xs, t)

# Datos para el problema de OR.
x = np.array([[0,0], [0,1], [1,0], [1,1]])
t = np.array([[-1], [1], [1], [1]])

# Mandamos a llamar a los procesos de ReLu.
alpha = 0.1
modelo_relu = Relu(x)

modelo_relu.regla_delta(t, alpha, 10)
modelo_relu.mostrar_pesos()

# Sacamos los xs y ws.
ws = modelo_relu.pesos() 
xs = modelo_relu.equis()

# Graficamos la frontera de Desicion. 
DecBoundaries(ws, xs, t)