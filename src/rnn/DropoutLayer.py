import numpy as np


class DropoutLayer:
    def __init__(self, rate: float):
        self.rate = rate

    def forward(self, X: np.ndarray, training: bool = False) -> np.ndarray:
        if not training or self.rate == 0.0:
            return X
        
        keep_prob = 1.0 - self.rate
        mask = np.random.binomial(1, keep_prob, size=X.shape)
        return X * mask / keep_prob