from typing import Optional
import numpy as np
from activation import softmax, tanh

class DenseLayer:
    def __init__(self, weights: np.ndarray, bias: np.ndarray, activation: Optional[str] = None):
        self.weights = weights
        self.bias = bias
        self.activation = activation

    def forward(self, X: np.ndarray) -> np.ndarray:
        output = np.dot(X, self.weights) + self.bias
        if self.activation == 'softmax':
            return softmax(output)
        elif self.activation == 'tanh':
            return tanh(output)
        return output