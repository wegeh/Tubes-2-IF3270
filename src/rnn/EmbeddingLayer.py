import numpy as np


class EmbeddingLayer:
    def __init__(self, embedding_weights: np.ndarray):
        self.embedding_weights = embedding_weights
        self.vocab_size, self.embedding_dim = embedding_weights.shape

    def forward(self, token_indices: np.ndarray) -> np.ndarray:
        return self.embedding_weights[token_indices]