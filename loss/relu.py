import numpy as np


class ReLU:
    def __init__(self):
        self._forward_cash

    def forward(self, x: np.ndarray):
        self._forward_cash = (x > 0).astype(int)
        return x * self._forward_cash

    def backward(self, gradient: np.ndarray):
        return gradient * self._forward_cash


act = ReLU()
x = np.random.rand(3, 4)
act.forward(x)
gradient = np.random.rand(3, 4)
act.backward(x)
