import matplotlib.pyplot as plt
import numpy as np
import math


# Activation function 1 => sigmoid
def sigmoid(x): # Takes an array
    a = []
    for i in x:
        a.append(1/(1+math.exp(-i)))
    return a


# 2 => reLU
def relu(x):
    a = [ ]
    for i in x:
        if i < 0:
            a.append(0)
        else:
            a.append(i)
    return a


# 3 => leaky reLU
def leaky_relu(x):
    leak_rate = 10
    a = [ ]
    for i in x:
        if i < 0:
            a.append(i/leak_rate)
        else:
            a.append(i)
    return a


# 4 => Hyperbolic tangent
def tanh(x, derivative = False):
    if derivative:
        return 1 - x**2
    return np.tanh(x)


# 5 => Swish
def swish(x):
    return sigmoid(x)*x


# Arranging the data and implying the functions
data = np.arange(-3., 3., 0.1)
sig = sigmoid(data)
relu = relu(data)
lr = leaky_relu(data)
tanh = tanh(data)
swish = swish(data)

# Plotting
line1, = plt.plot(data, sig, label="Sigmoid")
line2, = plt.plot(data, relu, label="reLU")
line3, = plt.plot(data, lr, label="Leaky_reLU")
line4, = plt.plot(data, tanh, label="Hyperbolic_Tangent")
line5, = plt.plot(data, swish, label="Swish")

plt.legend(handles=[line1, line2, line3, line4, line5])
plt.axhline(y=0, color="k")
plt.axvline(x=0, color="k")
plt.show()
