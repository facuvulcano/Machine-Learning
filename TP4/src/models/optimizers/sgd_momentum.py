from .optimizer import Optimizer
import numpy as np

class SGDMomentum(Optimizer):
    def __init__(self, learning_rate=0.01, momentum=0.9):
        self.lr = learning_rate
        self.momentum = momentum
        self.velocity = {}

    def update(self, params, grads):
        for idx, (param, grad) in enumerate(zip(params, grads)):
            if idx not in self.velocity:
                self.velocity[idx] = np.zeros_like(grad)
            self.velocity[idx] = self.momentum * self.velocity[idx] - self.lr * grad
            param.data += self.velocity[idx]