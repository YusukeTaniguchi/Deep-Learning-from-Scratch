import numpy as np
from numerical_diff import numerical_diff
from sigmoid import sigmoid
from softmax import softmax
from cross_entropy_error import cross_entropy_error

class TwoLayerNet:

    def __init__(self, input_size, hidden_size, output_size, weight_init_std = 0.01):
        self.params = {}
        self.params["W1"] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params["b1"] = np.zeros(hidden_size)
        self.params["W2"] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params["b2"] = np.zeros(output_size)

    def predict(self, x):

        A1 = np.dot(x, self.params["W1"]) + self.params["b1"]
        Z1 = sigmoid(A1)
        A2 = np.dot(Z1, self.params["W2"]) + self.params["b2"]
        y = softmax(A2)

        return y

    def loss(self, x, t):
        y = self.predict(x)

        return cross_entropy_error(y, t)

    def numerical_gradient(self, x, t):
        f = lambda W: self.loss(x, t)

        grads = {}

        grads["W1"] = numerical_diff(f, self.params["W1"])
        grads["b1"] = numerical_diff(f, self.params["b1"])
        grads["W2"] = numerical_diff(f, self.params["W2"])
        grads["b2"] = numerical_diff(f, self.params["b2"])

        return grads

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis = 1)
        t = np.argmax(t, axis = 1)

        return np.sum(y == t) / float(y.shape[0])
