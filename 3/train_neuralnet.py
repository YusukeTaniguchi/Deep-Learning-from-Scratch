import numpy as np
from two_layer_net import TwoLayerNet
from mnist import load_mnist

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

train_loss_list = []

iter_num = 10000
train_size = t_train.shape[0]
batch_size = 100
learning_late = 0.1

network = TwoLayerNet(input_size = 784, hidden_size=50, output_size=10)

for i in range(iter_num):
    index = np.random.choice(train_size, batch_size)

    x_batch = x_train[index]
    t_batch = t_train[index]

    grad = network.numerical_gradient(x_batch, t_batch)

    for key in ("W1", "b1", "W2", "b2"):
        network.params[key] -= learning_late * grad[key]

    loss = network.loss(x_batch, t_batch)

    train_loss_list.append(loss)
