import numpy as np
from softmax import softmax
from gradient_descent import gradient_descent
from cross_entropy_error import cross_entropy_error

class simpleNet:

    def __init__(self):
        self.W = np.random.randn(2, 3)

    def predict(self, x):
        A = np.dot(x, self.W)
        return A

    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y, t)

        return loss


#試験用コード
net = simpleNet()
print(net.W)

x = np.array([0.6, 0.9])
p = net.predict(x)
print(p)

t = np.array([0, 0, 1])
print(net.loss(x, t))


