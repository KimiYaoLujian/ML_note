#pytorch 从零实现线性回归

import torch
from time import time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
import time

num_input = 2
number = 1000

true_w = torch.tensor([2, 3], dtype=torch.float64)
true_b = 5
features = torch.tensor(np.random.normal(0, 1, (number, num_input)))
labels = torch.mm(features, true_w.view(2, 1)) + true_b
labels += torch.tensor(np.random.normal(0, 0.1, (number, 1)))
# print(labels)
# fig = plt.figure()
# ax = Axes3D(fig)


# print(features[:, 1].numpy())

# plt.scatter(features[:, 1].numpy(), labels.numpy(), 10)
# plt.show()

def data_iter(batch_size, features, labels):
    number_exmples = len(features)
    indices = list(range(number_exmples))
    random.shuffle(indices)
    for i in range(0, number_exmples, batch_size):
        m = torch.tensor(indices[i : min(i + batch_size, number_exmples)])
        yield features.index_select(0, m), labels.index_select(0, m)


w = torch.tensor(np.random.normal(0, 0.01, (num_input, 1)), dtype=torch.float64)
b = torch.tensor([0], dtype=torch.float64)

w.requires_grad_(requires_grad=True)
b.requires_grad_(requires_grad=True)

def net(X, w, b):
    return torch.mm(X, w) + b


def loss(y_hat, y):
    return (y_hat - y.view(y_hat.size())) ** 2 / 2


def sgd(params, lr, batch_size):
    for param in params:
        param.data -= lr * param.grad / batch_size


lr = 0.01
epochs = 100
batch_size = 10
loss_list = []
for epoch in range(epochs):
    for X, y in data_iter(batch_size, features, labels):
        start = time.time()
        l = loss(net(X, w, b), y).sum()
        l.backward()
        sgd([w, b], lr, batch_size)
        w.grad.data.zero_()
        b.grad.data.zero_()
    train_loss = loss(net(features, w, b), labels)
    loss_list.append(train_loss.mean().item())
    print('epoch %d, loss %f time %.9f' % (epoch, train_loss.mean().item(), time.time()-start))

epoch_list = range(epochs)

plt.plot(epoch_list, loss_list)
plt.show()







