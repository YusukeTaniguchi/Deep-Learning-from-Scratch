import numpy as np

class AdaGrad:
    def __init__(self, lr = 0.01):
        self.lr = lr
        self.h = None

    def update(self, params, grads):
        if self.h is None:
            self.h = {}
            for key, value in params.items():
                self[key] = np.zeros_like(value)

        for key in params.keys():
            self.h[key] += grads[key] ** 2
            self.params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + 1e-7)