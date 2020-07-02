import numpy as np

# def softmax(x):
#     max_x = np.max(x)
#     exp_x = np.exp(x - max_x)
#     return exp_x / sum(exp_x)

def softmax(x):
    x = x - np.max(x, axis=-1, keepdims=True)
    return np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True)