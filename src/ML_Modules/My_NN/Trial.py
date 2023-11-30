import numpy as np
import NeuralNetwork
from NeuralNetwork import *
import matplotlib.pyplot as plt


def my_func(x, y):
    return x ** 2 + y ** 2

input_size = 2
output_size = 1
layer_size_list = [15, 15]

my_nn = NeuralNetwork(input_size, output_size, layer_size_list)

n = 10000
x1 = np.linspace(-10, 10, num = n)
x2 = np.linspace(-10, 10, num = n)
np.random.shuffle(x1)
np.random.shuffle(x2)

X = np.vstack((x1, x2))
Y = my_func(x1, x2)

# Evaluate the predictions and loss pre-training
Y_pretrain = my_nn.NN_eval(X)
L_pretrain = my_nn.NN_Loss(X, Y)
print(f'{Y_pretrain=}')
print(f'{L_pretrain=}')

alpha = 0.01
my_nn.NN_train(X, Y, alpha)

Y_posttrain = my_nn.NN_eval(X)
L_posttrain = my_nn.NN_Loss(X, Y)
print(f'{Y_posttrain=}')
print(f'{L_posttrain=}')

quit()


def my_func(x, y):
    return x **2 + y ** 2

input_size = 2
output_size = 1
layer_size_list = [2, 2]

my_nn = NeuralNetwork.NeuralNetwork(input_size, output_size, layer_size_list)

n = 100
x1 = np.linspace(-10, 10, n)
x2 = np.linspace(-10, 10, n)
np.random.shuffle(x1)
np.random.shuffle(x2)

X = np.vstack((x1, x2)).T
Y = my_func(x1, x2)

my_nn.NN_train(X, Y)

Y_pred = np.zeros(n)
for i in range(n):
    x_i = X[i, :]
    Y_pred[i] = my_nn.NN_eval(x_i)[0]


my_Y = np.vstack((Y, Y_pred)).T
print(my_Y)