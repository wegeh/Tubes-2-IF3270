import numpy as np

class PoolingLayer:
    def __init__(self, size=2, type='max'):
        self.size = size
        self.type = type

    def forward(self, x):
        batch, h_in, w_in, c = x.shape
        h_out = h_in // self.size
        w_out = w_in // self.size
        out = np.zeros((batch, h_out, w_out, c))
        for b in range(batch):
            for i in range(h_out):
                for j in range(w_out):
                    window = x[b, i*self.size:(i+1)*self.size, j*self.size:(j+1)*self.size, :]
                    if self.type == 'max':
                        out[b,i,j,:] = np.max(window, axis=(0,1))
                    else:
                        out[b,i,j,:] = np.mean(window, axis=(0,1))
        return out
