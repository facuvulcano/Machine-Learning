from .optimizer import Optimizer

class MiniBatchSGD(Optimizer):
    def __init__(self, learning_rate=0.01, batch_size=32):
        self.lr = learning_rate
        self.batch_size = batch_size

    def update(self, params, grads):
        for param, grad in zip(params, grads):
            param.data -= self.lr * grad
