import numpy as np
from activation import activation_functions

class ConvolutionalLayer:
    def __init__(self, weights, bias, stride=1, padding='same', activation='relu'):
        self.weights = weights  
        self.bias = bias        
        self.stride = stride
        self.padding = padding
        self.activation = activation_functions[activation]

    def forward(self, x):
        batch_size, h_in, w_in, c_in = x.shape
        f_h, f_w, _, c_out = self.weights.shape
        if self.padding == 'same':
            pad_h = (f_h - 1) // 2
            pad_w = (f_w - 1) // 2
            x_padded = np.pad(x, ((0,0),(pad_h,pad_h),(pad_w,pad_w),(0,0)), mode='constant')
        else:
            x_padded = x
        h_out = (h_in + 2*pad_h - f_h)//self.stride + 1
        w_out = (w_in + 2*pad_w - f_w)//self.stride + 1
        out = np.zeros((batch_size, h_out, w_out, c_out))
        for b in range(batch_size):
            for i in range(h_out):
                for j in range(w_out):
                    for f in range(c_out):
                        h_start = i*self.stride
                        w_start = j*self.stride
                        window = x_padded[b, h_start:h_start+f_h, w_start:w_start+f_w, :]
                        out[b, i, j, f] = np.sum(window * self.weights[:,:,:,f]) + self.bias[f]
        return self.activation(out)
