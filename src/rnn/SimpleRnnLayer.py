import numpy as np
from activation import tanh

class SimpleRNNLayer:
    def __init__(self, W_xh: np.ndarray, W_hh: np.ndarray, b_h: np.ndarray, rnn_units: int):
        self.W_xh = W_xh
        self.W_hh = W_hh
        self.b_h = b_h
        self.rnn_units = rnn_units

    def forward(self, X: np.ndarray, return_sequences: bool = False) -> np.ndarray:
        batch_size, seq_len, _ = X.shape
        h = np.zeros((batch_size, self.rnn_units))
        outputs = []

        for t in range(seq_len):
            x_t = X[:, t, :]
            h = tanh(np.dot(x_t, self.W_xh) + np.dot(h, self.W_hh) + self.b_h)
            if return_sequences:
                outputs.append(h.copy())
        
        if return_sequences:
            return np.stack(outputs, axis=1)
        else:
            return h 