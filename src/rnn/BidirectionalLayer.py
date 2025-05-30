import numpy as np
from SimpleRnnLayer import SimpleRNNLayer

class BidirectionalLayer:
    def __init__(self, forward_layer: SimpleRNNLayer, backward_layer: SimpleRNNLayer):
        self.forward_layer = forward_layer
        self.backward_layer = backward_layer

    def forward(self, X: np.ndarray, return_sequences: bool = False) -> np.ndarray:

        out_fw = self.forward_layer.forward(X, return_sequences=return_sequences)

        X_reversed = X[:, ::-1, :]
        out_bw = self.backward_layer.forward(X_reversed, return_sequences=return_sequences)

        if return_sequences:
            out_bw_reversed = out_bw[:, ::-1, :]
            return np.concatenate([out_fw, out_bw_reversed], axis=-1)
        else:
            return np.concatenate([out_fw, out_bw], axis=-1)