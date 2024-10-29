from .optimizer import Optimizer
import numpy as np

class Adam(Optimizer):
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = {}
        self.v = {}
        self.t = 0

    def update(self, params, grads):
        self.t += 1
        for idx, (param, grad) in enumerate(zip(params, grads)):
            if idx not in self.m:
                self.m[idx] = np.zeros_like(grad)
                self.v[idx] = np.zeros_like(grad)

            # Actualizar estimaciones de momentos
            self.m[idx] = self.beta1 * self.m[idx] + (1 - self.beta1) * grad
            self.v[idx] = self.beta2 * self.v[idx] + (1 - self.beta2) * (grad ** 2)

            # Corregir el sesgo
            m_corr = self.m[idx] / (1 - self.beta1 ** self.t)
            v_corr = self.v[idx] / (1 - self.beta2 ** self.t)

            # Actualizar parametros
            param.data -= self.lr * m_corr / (np.sqrt(v_corr) + self.epsilon)
            