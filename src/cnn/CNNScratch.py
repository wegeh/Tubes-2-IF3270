from ConvolutionalLayer import ConvolutionalLayer
from PoolingLayer import PoolingLayer
from FlattenLayer import FlattenLayer
from DenseLayer import DenseLayer
import numpy as np
from tensorflow.keras.layers import Conv2D, Dense, MaxPooling2D, AveragePooling2D, Flatten


class CNNScratch:
    def __init__(self, keras_model):
        layers = keras_model.layers
        self.layers = []
        for l in layers:
            if isinstance(l, Conv2D):
                W, b = l.get_weights()
                activation_name = l.activation.__name__ if hasattr(l, 'activation') else None
                self.layers.append(ConvolutionalLayer(W, b, activation=activation_name))
            elif isinstance(l, MaxPooling2D):
                self.layers.append(PoolingLayer(size=2, type='max'))
            elif isinstance(l, AveragePooling2D):
                self.layers.append(PoolingLayer(size=2, type='avg'))
            elif isinstance(l, Flatten):
                self.layers.append(FlattenLayer())
            elif isinstance(l, Dense):
                W, b = l.get_weights()
                activation_name = l.activation.__name__ if hasattr(l, 'activation') else None
                if activation_name == 'softmax':
                    act = 'softmax'
                elif activation_name == 'relu':
                    act = 'relu'
                elif activation_name == 'sigmoid':
                    act = 'sigmoid'
                elif activation_name == 'tanh':
                    act = 'tanh'
                elif activation_name == 'leaky_relu':
                    act = 'leaky_relu'
                else:
                    act = None
                self.layers.append(DenseLayer(W, b, activation=act))

    def forward(self, x, batch_size=32):
        results = []
        for i in range(0, x.shape[0], batch_size):
            batch = x[i:i+batch_size]
            out = batch
            for layer in self.layers:
                out = layer.forward(out)
            results.append(out)
        return np.vstack(results) 
