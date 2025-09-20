from exercises.redes_neuronales_artificiales.single_perceptron import InputData, Perceptron
import random
# Adjust the import path to include the parent directory for py_utils
import sys
import os
from logging import DEBUG, INFO, WARNING, ERROR
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from py_utils.logger import set_logging, plog

# Llamada a la funcion para configurar el logging
set_logging(log_file='redes_neuronales_artificiales.log')

i1 = InputData(x=0.1)
i2 = InputData(x=0.2)
i3 = InputData(x=0.3)
 
b1 = random.random()

p1 = Perceptron(inputs=[i1, i2], b=b1)

plog(f"perceptron 1:\n   - inputs: {p1.inputs}\n   - bias: {p1.b}\n   - forward: {p1.z}\n   - output: {p1.a}\n")

# Ejercicio 1: Red de una neurona.
#
# TODO: Crea una red neuronal de un solo perceptrón y una sola entrada con los objetos previamente creados.
#       Asigna el valor de la salida a "single_layer_perceptron"
#
# NOTE: Toma como referencia el diagrama conceptual de la red.
#    
#   x ──► [ Neurona ] ──► a
#              ↑
#            (w, b)
#
single_layer_perceptron = single_layer_perceptron = Perceptron(inputs=[i1], b=b1).a


plog(f"Salida de la red de una sola capa: {single_layer_perceptron}", level=ERROR if single_layer_perceptron is None else DEBUG, eol=True)

# Ejercicio 2: Red de dos neuronas y una entrada.
#
# TODO: Crea una red neuronal de dos perceptrones y una sola entrada con los objetos previamente creados.
#       Asigna el valor de la salida a "two_layer_network"
#
# NOTE: Toma como referencia el diagrama conceptual de la red.
#
#   x1 ──► [ Neurona 1 ] ──► a1 ──► [ Neurona 2 ] ──► a2
#              ↑                          ↑
#            (w1, b1)                   (w2, b2)
#
# Ejercicio 2: Red de dos neuronas y una sola entrada
# Primera neurona
two_layer_network = Perceptron(inputs=[InputData(x=Perceptron(inputs=[i1], b=b1).a)], 
    b=b1
).a

plog(f"Salida de la red de dos capas: {two_layer_network}", level=ERROR if two_layer_network is None else DEBUG, eol=True)

# Ejercicio 3: Red de dos neuronas y una entrada.
#
# TODO: Crea una red neuronal de dos perceptrones y dos entradas con los objetos previamente creados.
#       Asigna el valor de la salida a "small_network"
#
# NOTE: Toma como referencia el diagrama conceptual de la red.
#
# Entrada         Capa Oculta 1        Salida
#   x1 ─────────► [ Neurona 1 ]  
#       ↘      ↗                ↘
#         ↘  ↗                    ↘
#           x                        ► [ a2 ] 
#         ↗  ↘                    ↗    
#       ↗      ↘                ↗         
#   x2 ─────────► [ Neurona 2 ]
#

# Ejercicio 3: Red de dos neuronas y dos entradas
# Capa oculta: dos neuronas reciben i1 e i2
small_network =Perceptron(
    inputs=[
        InputData(x=Perceptron(inputs=[i1, i2], b=b1).a),
        InputData(x=Perceptron(inputs=[i1, i2], b=b1).a)
    ],
    b=b1
).a
#erferferferf
plog(f"Salida de la red de pequeña: {small_network}", level=ERROR if small_network is None else DEBUG, eol=True)
##fefwes