from .optimizer import Optimizer

class SGD(Optimizer):
    def __init__(self, learning_rate=0.01):
        self.lr = learning_rate
    
    def update(self, params, grads):
        for param, grad in zip(params, grads):
            param.data -= self.lr * grad
            