from mnist import load_mnist
import pickle
from sigmoid import sigmoid
import numpy as np
from softmax import softmax


def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)

    return x_test, t_test

def init_network():
    with open("sample_weight.pkl", "rb") as f:
        network = pickle.load(f)

    return network;

def predict(network, x):
    W1, W2, W3 = network["W1"], network["W2"], network["W3"]
    b1, b2, b3 = network["b1"], network["b2"], network["b3"]

    A1 = np.dot(x, W1) + b1
    Z1 = sigmoid(A1)
    A2 = np.dot(Z1, W2) + b2
    Z2 = sigmoid(A2)
    A3 = np.dot(Z2, W3) + b3
    y = softmax(A3)

    return y

x, t = get_data()
network = init_network()

batch_size = 100
accuracy_count = 0
for index in range(0, len(x), batch_size):
    y = predict(network, x[index:(index + batch_size)])
    p = np.argmax(y, axis = 1)
    accuracy_count += np.sum(p == t[index:(index + batch_size)])

print(accuracy_count, len(x), accuracy_count / len(x))