import numpy as np

def numerical_diff(f, x):
    h = 1e-4

    grad = np.zeros_like(x)

    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        index = it.multi_index

        temp = x[index]
        x[index] = temp + h
        fx1 = f(x)

        x[index] = temp - h
        fx2 = f(x)

        grad[index] = (fx1 - fx2)/ (2 * h)

        x[index] = temp

        it.iternext()

    return grad