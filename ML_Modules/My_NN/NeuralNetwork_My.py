import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, output_size, layer_size_list):
        self.input_size = input_size
        self.output_size = output_size
        self.layers = []
        layer_size_list = [input_size] + layer_size_list + [output_size]
        for i in range(1, len(layer_size_list)):
            layer_size = layer_size_list[i]
            prev_layer_size = layer_size_list[i - 1]
            self.layers.append(Layer(layer_size, prev_layer_size))
    
    def NN_eval(self, X):
        vec = X.copy()
        for layer in self.layers:
            vec = ReLU(((layer.W @ vec).T + layer.b).T)
        return vec.flatten()
    
    def NN_Loss(self, X, Y):
        Y_hat = self.NN_eval(X)
        err = Y_hat - Y
        Loss = (1 / 2) * err @ err
        return Loss
    
    def NN_train(self, X, Y, alpha):
        for k in range(1):
            for i in range(Y.shape[0]):
                x_i = X[:, i]
                y_i = Y[i]
                self.forward_pass(x_i)
                self.backward_pass(x_i, y_i)
                self.update_NN(alpha)
                Loss = self.NN_Loss(X, Y)
            print(f'Iteration: {k} Loss: {Loss}')
            # self.forward_pass(X)
            # self.backward_pass(X, Y)
            # self.update_NN(alpha)
            # Loss = self.NN_Loss(X, Y)
            # print(f'Iteration: {k} Loss: {Loss}')
    
    def forward_pass(self, X):
        vec = X.copy()
        for layer in self.layers:
            # vec = ReLU(((layer.W @ vec).T + layer.b).T)
            layer.z_eval(vec)
            layer.a_eval()
            vec = layer.a
    
    def backward_pass(self, X, Y):
        # Start with output layer
        layer = self.layers[-1]
        prev_layer = self.layers[-2]
        dJ_dz = - (Y - layer.z)
        dJ_dW = np.outer(dJ_dz, prev_layer.a)
        dJ_db = dJ_dz
        layer.dJdW = dJ_dW
        layer.dJdb = dJ_db

        for i in range(-2, -len(self.layers), -1):
            layer = self.layers[i]
            prev_layer = self.layers[i - 1]
            next_layer = self.layers[i + 1]
            dJ_da = (next_layer.W).T @ dJ_dz
            dJ_dz = dJ_da * dReLU(layer.z)
            dJ_dW = np.outer(dJ_dz, prev_layer.a)
            dJ_db = dJ_dz
            layer.dJdW = dJ_dW
            layer.dJdb = dJ_db

        # Update first layer
        layer = self.layers[-len(self.layers)]
        next_layer = self.layers[-len(self.layers) + 1]
        dJ_da = (next_layer.W).T @ dJ_dz
        dJ_dz = dJ_da * dReLU(layer.z)
        dJ_dW = np.outer(dJ_dz, ReLU(X))
        dJ_db = dJ_dz
        layer.dJdW = dJ_dW
        layer.dJdb = dJ_db
    
    def update_NN(self, alpha):
        for layer in self.layers:
            a = layer.W.copy()
            layer.W -= alpha * layer.dJdW
            layer.b -= alpha * layer.dJdb


class Layer:
    def __init__(self, layer_size, prev_layer_size):
        self.size = layer_size
        mean = 0
        std = 1
        self.W = mean + std * np.random.randn(layer_size, prev_layer_size)
        self.b = mean + std * np.zeros(layer_size)
        self.z = mean + std * np.zeros(layer_size)
        self.a = mean + std * np.zeros(layer_size)
        self.dJdW = mean + std * np.random.randn(layer_size, prev_layer_size)
        self.dJdb = mean + std * np.random.randn(layer_size)
    
    def z_eval(self, X):
        self.z = self.W @ X + self.b
    
    def a_eval(self):
        self.a = ReLU(self.z)


def ReLU(X):
    return X * (X > 0)

def dReLU(X):
    return 1.0 * (X > 0)
