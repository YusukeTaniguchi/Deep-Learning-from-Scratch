import numpy as np
from numerical_gradient import numerical_gradient as NG

def gradient_descent(f, init_x, lr = 0.01, step_num = 100):
    x = init_x

    for i in range(step_num):
        grad = NG(f, x)
        x -= lr * grad
        
