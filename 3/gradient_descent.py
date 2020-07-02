from numerical_diff import numerical_diff
import numpy as np

def gradient_descent(f, init_x, lr = 0.01, step_num = 100):
    x = init_x


    for index in range(step_num):
        grad = numerical_diff(f, x)

        x -= lr * grad

    return x

#動作確認用
# import numpy as np
# def function_2(x):
#     return x[0]**2 + x[1] ** 2
#
# init_x = np.array([-3.0, 4.0])
# print(gradient_descent(function_2, init_x, 0.1, 100))