# Título Programa: Red de Hopfield.
# Fecha: 05-diciembre-2022 Red de Hopfield
# Autor: Renteria Arriaga Josue

import numpy as np
import random

# Carga de todos los numeros y anexados a una lista.
lista = []

# Matriz del Numero 1
E1_numero1= [[-1], [-1], [1], [-1], [-1], [-1], [1], [1], [-1], [-1], [-1], [-1], [1], [-1], [-1], [-1], [-1], [1], [-1], [-1], [-1], [-1], [1], [-1], [-1], [-1], [-1], [1], [-1], [-1], [-1], [1], [1], [1], [-1]]
E1_numero1_contrario= [[-1, -1, 1, -1, -1, -1, 1, 1, -1, -1, -1, -1, 1, -1, -1, -1, -1, 1, -1, -1, -1, -1, 1, -1, -1, -1, -1, 1, -1, -1, -1, 1, 1, 1, -1]]
lista.append(E1_numero1)
lista.append(E1_numero1_contrario)

# Matriz del Numero 2
E1_numero2= [[1], [1], [1], [1], [-1], [-1], [-1], [-1], [-1], [1], [-1], [-1], [-1], [-1], [1], [-1], [1], [1], [1], [-1], [1], [-1], [-1], [-1], [-1], [1], [-1], [-1], [-1], [-1], [-1], [1], [1], [1], [1]]
E1_numero2_contrario= [[1, 1, 1, 1, -1, -1, -1, -1, -1, 1, -1, -1, -1, -1, 1, -1, 1, 1, 1, -1, 1, -1, -1, -1, -1, 1, -1, -1, -1, -1, -1, 1, 1, 1, 1]]
        
lista.append(E1_numero2)
lista.append(E1_numero2_contrario)

# Matriz del Numero 3
E1_numero3= [[1], [1], [1], [1], [-1], [-1], [-1], [-1], [-1], [1], [-1], [-1], [-1], [-1], [1], [-1], [1], [1], [1], [1], [-1], [-1], [-1], [-1], [1], [-1], [-1], [-1], [-1], [1], [1], [1], [1], [1], [-1]]
E1_numero3_contrario = [[1, 1, 1, 1, -1, -1, -1, -1, -1, 1, -1, -1, -1, -1, 1, -1, 1, 1, 1, 1, -1, -1, -1, -1, 1, -1, -1, -1, -1, 1, 1, 1, 1, 1, -1]]
lista.append(E1_numero3)
lista.append(E1_numero3_contrario)

# Matriz del Numero 4
E1_numero4= [[1], [-1], [-1], [-1], [1], [1], [-1], [-1], [-1], [1], [1], [-1], [-1], [-1], [1], [1], [1], [1], [1], [1], [-1], [-1], [-1], [-1], [1], [-1], [-1], [-1], [-1], [1], [-1], [-1], [-1], [-1], [-1]]
E1_numero4_contrario = [[1, -1, -1, -1, 1,1, -1, -1, -1, 1, 1, -1, -1, -1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, 1, -1, -1, -1, -1, 1, -1, -1, -1, -1, -1]]
lista.append(E1_numero4)
lista.append(E1_numero4_contrario)

# Matriz del Numero 5
E1_numero5= [[1], [1], [1], [1], [1], [1], [-1], [-1], [-1], [-1], [1], [-1], [-1], [-1], [-1], [1], [1], [1], [1], [-1], [-1], [-1], [-1], [-1], [1], [-1], [-1], [-1], [-1], [1], [1], [1], [1], [1], [-1]]
E1_numero5_contrario = [[1, 1, 1, 1, 1, 1, -1, -1, -1, -1, 1, -1, -1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1, -1, 1, -1, -1, -1, -1, 1, 1, 1, 1, 1, -1]]
lista.append(E1_numero5)
lista.append(E1_numero5_contrario)

# Funcion que asocia las matrices
def asociar_matriz(numero_matriz, numero_matriz_contrario):
    # Hacemos el producto de la matriz 1x35 y 35x1
    xk_xk = np.matmul(numero_matriz, numero_matriz_contrario)
    # Creamos la Matriz identidad.
    matriz_identidad = np.identity(35)
    # Restamos la primer matriz por la matriz identidad.
    return (xk_xk - matriz_identidad)

# Funcion que nos da la Matriz de Hopfield 
def matriz_H(lista):
    memoria = np.zeros((35, 35))
    for i in range((len(lista)-1)):
        if i % 2 == 0:
            memoria = memoria + asociar_matriz(lista[i], lista[i+1])
    return memoria

# Funcion para la fase de Recuperacion.
def fase_recuperacion(memoria, matriz):
    xi = np.matmul(memoria, matriz)
    xi_transformado = []
    for i in range(len(xi)):
        numero = xi[i]
        if numero > 0: xi_transformado.append([1])
        elif numero < 0: xi_transformado.append([-1])
        else: xi_transformado.append([numero])
    return xi_transformado

# Funcion que muestra el numero
def mostrar_matriz(matriz):
    x = 0
    lista = []
    for i in range(len(matriz)):
        x += 1
        if x == 5:
            if matriz[i] == [-1]:
                lista.append(0)
            else:
                lista.append(1)
            print(lista)
            lista = []
            x = 0 
        else:
            if matriz[i] == [-1]:
                lista.append(0)
            else:
                lista.append(1)

# Funcion que agrega ruido.
def ruido(numero_ruido, matriz):
    for i in range(numero_ruido):
        numero = random.randint(0, (len(matriz)-1))
        if matriz[numero] == [1]: matriz[numero] = [-1]
        elif matriz[numero] == [-1]: matriz[numero] = [1]
        else: matriz[numero] = [0]
    return matriz

# Mostramos la Matriz
matriz_memoria = matriz_H(lista)
xi_transformado = fase_recuperacion(matriz_memoria, E1_numero1)
print("Matriz de Hopfield\n")
print(matriz_memoria)
print("\n")

# Fase de Recuperacion del 1 al 5.
print("\tFASE DE RECUPERACION\n")
print("Resultado para el 1")
xi_transformado = fase_recuperacion(matriz_memoria, E1_numero1)
mostrar_matriz(xi_transformado)
print("\n")
print("Resultado para el 2")
xi_transformado = fase_recuperacion(matriz_memoria, E1_numero2)
mostrar_matriz(xi_transformado)
print("\n")
print("Resultado para el 3")
xi_transformado = fase_recuperacion(matriz_memoria, E1_numero3)
mostrar_matriz(xi_transformado)
print("\n")
print("Resultado para el 4")
xi_transformado = fase_recuperacion(matriz_memoria, E1_numero4)
mostrar_matriz(xi_transformado)
print("\n")
print("Resultado para el 5")
xi_transformado = fase_recuperacion(matriz_memoria, E1_numero5)
mostrar_matriz(xi_transformado)
print("\n")

# Fase de Recuperacion del 1 con ruido.
print("\tAGREGAMOS RUIDO AL 1\n")
E1_numero1_ruido = ruido(1, E1_numero1)
print("Ruido agregado para el 1. se agrego no = 1")
mostrar_matriz(E1_numero1_ruido)
print("Resultado para el 1")
xi_transformado = fase_recuperacion(matriz_memoria, E1_numero1_ruido)
mostrar_matriz(xi_transformado)
print("\n")
E1_numero1_ruido = ruido(2, E1_numero1)
print("Ruido agregado para el 1. se agrego no = 2")
mostrar_matriz(E1_numero1_ruido)
print("Resultado para el 1")
xi_transformado = fase_recuperacion(matriz_memoria, E1_numero1_ruido)
mostrar_matriz(xi_transformado)
print("\n")
E1_numero1_ruido = ruido(5, E1_numero1)
print("Ruido agregado para el 1. se agrego no = 5")
mostrar_matriz(E1_numero1_ruido)
print("Resultado para el 1")
xi_transformado = fase_recuperacion(matriz_memoria, E1_numero1_ruido)
mostrar_matriz(xi_transformado)
print("\n")

# Fase de Recuperacion del 3 con ruido.
print("\tAGREGAMOS RUIDO AL 3\n")
E1_numero3_ruido = ruido(1, E1_numero3)
print("Ruido agregado para el 3. se agrego no = 1")
mostrar_matriz(E1_numero3_ruido)
print("Resultado para el 3")
xi_transformado = fase_recuperacion(matriz_memoria, E1_numero3_ruido)
mostrar_matriz(xi_transformado)
print("\n")
E1_numero3_ruido = ruido(2, E1_numero3)
print("Ruido agregado para el 3. se agrego no = 2")
mostrar_matriz(E1_numero3_ruido)
print("Resultado para el 3")
xi_transformado = fase_recuperacion(matriz_memoria, E1_numero3_ruido)
mostrar_matriz(xi_transformado)
print("\n")
E1_numero3_ruido = ruido(5, E1_numero3)
print("Ruido agregado para el 3. se agrego no = 5")
mostrar_matriz(E1_numero3_ruido)
print("Resultado para el 1")
xi_transformado = fase_recuperacion(matriz_memoria, E1_numero3_ruido)
mostrar_matriz(xi_transformado)
print("\n")