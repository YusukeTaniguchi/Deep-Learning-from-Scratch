import numpy as np

def cross_entropy_error(y, t):
    if y.ndim == 1:
        y = y.reshape(1, y.size)
        t = t.reshape(1, t.size)

    #バッチの平均
    return -np.sum(t * np.log(y + 1e-7))/y.shape[0]