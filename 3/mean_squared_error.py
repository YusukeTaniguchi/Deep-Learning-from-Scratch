import numpy as np

def mean_squared_error(y, t):
    error = (y - t) ** 2
    return np.sum(error) * 0.5