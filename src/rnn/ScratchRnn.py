from SimpleRnnLayer import SimpleRNNLayer
from BidirectionalLayer import BidirectionalLayer
from DropoutLayer import DropoutLayer
import numpy as np
from EmbeddingLayer import EmbeddingLayer
from typing import List, Any
class ScratchRNNModel:
    def __init__(self, layers: List[Any]):
        self.layers = layers
        rec_types = (SimpleRNNLayer, BidirectionalLayer)
        rec_indices = [i for i, layer in enumerate(self.layers) if isinstance(layer, rec_types)]
        if not rec_indices:
            raise ValueError("Tidak ada layer rekuren (SimpleRNNLayer/BidirectionalLayer) yang ditemukan.")
        self._last_recurrent_idx = max(rec_indices)

    def forward(self, X: np.ndarray, training: bool = False) -> np.ndarray:
        output = X
        rec_types = (SimpleRNNLayer, BidirectionalLayer)

        for i, layer in enumerate(self.layers):
            if isinstance(layer, rec_types):
                return_seq = (i < self._last_recurrent_idx)
                output = layer.forward(output, return_sequences=return_seq)
            elif isinstance(layer, DropoutLayer):
                output = layer.forward(output, training=training)
            else: 
                output = layer.forward(output)
        return output

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X, training=False)