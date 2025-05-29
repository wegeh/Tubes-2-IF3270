from activation import activation_functions, softmax
import numpy as np

class DenseLayer:
    def __init__(self, weights, bias, activation='relu'):
        self.weights = weights   
        self.bias = bias      
        self.activation = activation_functions[activation]

    def forward(self, x):
        out = np.dot(x, self.weights) + self.bias
        if self.activation == softmax:
            return self.activation(out)
        else:
            return self.activation(out)
