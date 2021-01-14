---
title: Learn MNIST in PyTorch from Scratch to CNN
mathjax: true
date: 2021-01-11 19:33:08
tags: Deep-Learning
---

Today I spent nearly an afternoon to follow the tutorial on [pytorch.org](pytorch.org). So just recall what I have learnt here.

(all in PyTorch...)

## from Scratch

We first write our code without too many features of PyTorch so that we can gradually see what can be simplified when using PyTorch.

### Download MNIST Data

data download link: https://github.com/pytorch/tutorials/raw/master/_static/mnist.pkl.gz

After manually decompressing this file, we use `pickle` to read data.

```python
def read_data():
    path = Path('data/mnist/mnist.pkl')
    if path.exists():
        with open('data/mnist/mnist.pkl', 'rb') as f:
            (XTrain, YTrain), (XTest, YTest), _ = pickle.load(f, encoding='latin-1')
        return XTrain, YTrain, XTest, YTest
    else:
        raise Exception(FileNotFoundError)
```

It's worth mentioning that the second dimension of `XTrain` and `XTest` are 784, which is identical to 28 * 28.

Using `plt.imshow` and `plt.show` function, single data can be shown easily.

Here is the initial code implementing MNIST with few feature of PyTorch:

```python
import numpy as np
from pathlib import Path
import pickle
import matplotlib.pyplot as plt
import torch
import math


def read_data():
    path = Path('data/mnist/mnist.pkl')
    if path.exists():
        with open('data/mnist/mnist.pkl', 'rb') as f:
            (XTrain, YTrain), (XTest, YTest), _ = pickle.load(f, encoding='latin-1')
        return XTrain, YTrain, XTest, YTest
    else:
        raise Exception(FileNotFoundError)


def draw(X):
    print(X.shape)
    plt.imshow(X.reshape((28, 28)), cmap='gray')
    plt.show()


def log_softmax(x):
    return x - x.exp().sum(-1).log().unsqueeze(-1)


def forward(X):
    return log_softmax(X @ weights + bias)


def nll(batch_z, batch_y):
    return -batch_z[range(batch_y.shape[0]), batch_y].mean()


loss_func = nll


def accuracy(batch_z, batch_y):
    temp = torch.argmax(batch_z, dim=1)
    r = (temp == batch_y)
    return r.float().mean()


def get_batch_train_data(batch_size, iteration):
    start = batch_size * iteration
    end = start + iteration
    return XTrain[start:end], YTrain[start:end]


def get_batch_test_data(batch_size, iteration):
    start = batch_size * iteration
    end = start + iteration
    return XTest[start:end], YTest[start:end]


XTrain, YTrain, XTest, YTest = read_data()  # train: 50000, test: 10000
XTrain, YTrain, XTest, YTest = map(torch.tensor, (XTrain, YTrain, XTest, YTest))

weights = torch.randn(784, 10) / math.sqrt(784)
weights.requires_grad_()
bias = torch.zeros(10, requires_grad=True)


def train(max_epoch, max_iteration, batch_size, lr):
    print('training...')
    global weights, bias
    for epoch in range(max_epoch):
        for iteration in range(max_iteration):
            start = iteration * batch_size
            end = start + batch_size
            batch_x, batch_y = get_batch_train_data(batch_size, iteration)
            batch_z = forward(batch_x)
            loss = loss_func(batch_z, batch_y)

            loss.backward()
            with torch.no_grad():
                weights -= lr * weights.grad
                bias -= lr * bias.grad
                weights.grad.zero_()
                bias.grad.zero_()

    print('training done.')


def test():
    print('testing...')
    ZTest = forward(XTest)
    print('loss=%.4f, accuracy=%.4f' % (loss_func(ZTest, YTest), accuracy(ZTest, YTest)))
    print('testing done.')


def main():
    num_train = XTrain.shape[0]
    num_test = XTest.shape[0]
    # batch_x = XTrain[:batch_size]
    # batch_z = forward(batch_x)
    # print(batch_z[0], batch_z.shape)
    #
    # batch_y = YTrain[:batch_size]
    # print(loss_func(batch_z, batch_y))
    #
    # print(accuracy(batch_z, batch_y))

    batch_size = 64
    lr = 0.05
    max_epoch = 20
    max_iteration = math.ceil(num_train / batch_size)
    train(max_epoch, max_iteration, batch_size, lr)
    test()


if __name__ == '__main__':
    main()

```

Most of the details can be answered if you have learnt about the basic knowledge of neural network, and most of the procedures are very similar to [the tutorial I learn](github.com/microsoft/ai-edu).

Now the magic just begins.

## Where can be simplified using PyTorch feature?

### choosing from torch.nn.functional

In previous code, we must manually define a function `nll` for calculating loss, which can be replaced by `torch.nn.functional`.

This stuff contains lots of functions, so that we needn't implement each function we use, which is quite convenient.

## extending torch.nn.Module

we can define our whole neural network as a class, whose super class is `torch.nn.Module`. In this way, parameters can be stored inside this object, which is friendly for us to program.

## using layer objects from torch.nn

The model previous code uses is exactly a linear layer, which can be replaced by `torch.nn.Linear`, which contains parameters within it.

What's more, pooling layer, convolution layer are also available to use in `torch.nn`, which greatly reduces workflow.

```python
loss_func = F.cross_entropy
model = NeuralNet() # I am hanhan!

def train(max_epoch, max_iteration, batch_size, lr):
    print('training...')
    for epoch in range(max_epoch):
        for iteration in range(max_iteration):
            batch_x, batch_y = get_batch_train_data(batch_size, iteration)
            batch_z = model(batch_x)
            loss = loss_func(batch_z, batch_y)

            loss.backward()
            with torch.no_grad():
                for p in model.parameters():
                    p -= p.grad * lr
                model.zero_grad()

    print('training done.')


def test():
    print('testing...')
    ZTest = model(XTest)
    print('loss=%.4f, accuracy=%.4f' % (loss_func(ZTest, YTest), accuracy(ZTest, YTest)))
    print('testing done.')

```

## modifying parameters by torch.optim

`torch.optim` includes many methods of optimization, including most commonly-used SGD. With this tool, we needn't traverse all parameters and subtract its specific value from itself, but only write two lines of code:

Before:
```python
with torch.no_grad():
    for p in model.parameters():
        p -= p.grad * lr
    model.zero_grad()
```

After:
```python
with torch.no_grad():
    optimizer.step()
    optimizer.zero_grad()
```
Remember to zero grad after each epoch is done, otherwise the gradients will become way too large and get unexpected results.

btw, why I comment that I am hanhan? Because I made mistake on `model`. Here `model` must be an instance of `NeuralNet` rather than a alias, for the values of weights are random. Otherwise, your loss value will always get above 2...

### loading dataset and dataloader

How to import?

```python
from torch.utils.data import TensorDataset, DataLoader
```

How to declare?
```python
train_set = TensorDataset(XTrain, YTrain)
train_loader = DataLoader(train_set, batch_size=bs, shuffle=True)
valid_set = TensorDataset(XValid, YValid)
valid_loader = DataLoader(valid_set, batch_size=bs * 2, shuffle=False)
```

Where is the validation set? I just generate the validation set by extracting one tenth of data of training set. This trick is learnt from "microsoft/ai-edu".

Since we have things prepared, the whole training code is simple:

```python
def train():
    print('training...')
    for epoch in range(max_epoch):
        model.train() # written before training
        for batch_x, batch_y in train_loader: # traversal simplified
            batch_z = model(batch_x)
            loss = loss_func(batch_z, batch_y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        model.eval() # written before validating
        with torch.no_grad():
            valid_loss = sum(loss_func(model(batch_x), batch_y) for batch_x, batch_y in valid_loader) / num_valid
        print("epoch %d, validation loss=%.4f" % (epoch, valid_loss))

    print('training done.')
```

## Switch to CNN

CNN is widely used when data is images. Now let's try to solve MNIST with CNN, just to feel how powerful CNN is.

In fact, most of the code remain the same. The only area we need to modify is in the definition of class, replacing linear layer with more complex layers.

Here is the code:
```python
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader

import matplotlib.pyplot as plt
from pathlib import Path
import pickle


# class MNIST(nn.Module):
#     def __init__(self):
#         super(MNIST, self).__init__()
#         self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1)
#         self.conv2 = nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1)
#         self.conv3 = nn.Conv2d(16, 10, kernel_size=3, stride=2, padding=1)
#
#     def forward(self, batch_x):
#         batch_x = batch_x.view(-1, 1, 28, 28)
#         batch_x = F.relu(self.conv1(batch_x))
#         batch_x = F.relu(self.conv2(batch_x))
#         batch_x = F.relu(self.conv3(batch_x))
#         batch_x = F.avg_pool2d(batch_x, 4)
#         return batch_x.view(-1, batch_x.size(1))


def draw(X):
    print(X.shape)
    plt.imshow(X.reshape((28, 28)), cmap='gray')
    plt.show()

def read_data():
    path = Path('data/mnist/mnist.pkl')
    if path.exists():
        with open('data/mnist/mnist.pkl', 'rb') as f:
            (XTrain, YTrain), (XTest, YTest), _ = pickle.load(f, encoding='latin-1')
        return XTrain, YTrain, XTest, YTest
    else:
        raise Exception(FileNotFoundError)

def generate_validation_set(k=10):
    global num_train, XTrain, YTrain
    num_valid = num_train // k
    num_train -= num_valid
    XValid, YValid = XTrain[:num_valid], YTrain[:num_valid]
    XTrain, YTrain = XTrain[num_valid:], YTrain[num_valid:]
    return XValid, YValid, num_valid


XTrain, YTrain, XTest, YTest = read_data()  # train: 50000, test: 10000
num_train = XTrain.shape[0]
num_test = XTest.shape[0]
XValid, YValid, num_valid = generate_validation_set(k=10)
XTrain, YTrain, XValid, YValid, XTest, YTest = map(torch.tensor, (XTrain, YTrain, XValid, YValid, XTest, YTest))

def accuracy(batch_z, batch_y):
    temp = torch.argmax(batch_z, dim=1)
    r = (temp == batch_y)
    return r.float().mean()


class Lambda(nn.Module):
    def __init__(self, func):
        super(Lambda, self).__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)

# hyper-parameter
bs = 64
lr = 0.1
momentum = 0.9
max_epoch = 20
# essential stuff
loss_func = F.cross_entropy
# model = MNIST()
model = nn.Sequential(
    Lambda(lambda x: x.view(-1, 1, 28, 28)),
    nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
    nn.ReLU(),
    nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1),
    nn.ReLU(),
    nn.Conv2d(16, 10, kernel_size=3, stride=2, padding=1),
    nn.ReLU(),
    nn.AvgPool2d(4),
    Lambda(lambda x: x.view(x.size(0), -1)),
)
# NOTE: relu is different in these two forms!(F.relu vs nn.ReLU)
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
# datasets and dataloaders
train_set = TensorDataset(XTrain, YTrain)
train_loader = DataLoader(train_set, batch_size=bs, shuffle=True)
valid_set = TensorDataset(XValid, YValid)
valid_loader = DataLoader(valid_set, batch_size=bs * 2, shuffle=False)

def train():
    print('training...')
    for epoch in range(max_epoch):
        model.train()
        # training: using training set
        for batch_x, batch_y in train_loader:
            # forward
            batch_z = model(batch_x)
            # backward
            loss = loss_func(batch_z, batch_y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        model.eval()
        # inference: using validation set
        with torch.no_grad():
            valid_loss = sum(loss_func(model(batch_x), batch_y) for batch_x, batch_y in valid_loader) / num_valid
        print("epoch %d, validation loss=%.4f" % (epoch, valid_loss))

    print('training done.')


def test():
    print('testing...')
    ZTest = model(XTest)
    print('loss=%.4f, accuracy=%.4f' % (loss_func(ZTest, YTest), accuracy(ZTest, YTest)))
    print('testing done.')


train()
test()
```

## Result Comparision

### Linear
```
training...
epoch 0, validation loss=0.0032
epoch 1, validation loss=0.0028
epoch 2, validation loss=0.0026
epoch 3, validation loss=0.0025
epoch 4, validation loss=0.0024
epoch 5, validation loss=0.0024
epoch 6, validation loss=0.0023
epoch 7, validation loss=0.0023
epoch 8, validation loss=0.0023
epoch 9, validation loss=0.0022
epoch 10, validation loss=0.0022
epoch 11, validation loss=0.0022
epoch 12, validation loss=0.0022
epoch 13, validation loss=0.0022
epoch 14, validation loss=0.0022
epoch 15, validation loss=0.0022
epoch 16, validation loss=0.0022
epoch 17, validation loss=0.0021
epoch 18, validation loss=0.0022
epoch 19, validation loss=0.0021
training done.
testing...
loss=0.2707, accuracy=0.9251
testing done.
```

### CNN
```
training...
epoch 0, validation loss=0.0042
epoch 1, validation loss=0.0020
epoch 2, validation loss=0.0018
epoch 3, validation loss=0.0017
epoch 4, validation loss=0.0015
epoch 5, validation loss=0.0012
epoch 6, validation loss=0.0015
epoch 7, validation loss=0.0013
epoch 8, validation loss=0.0012
epoch 9, validation loss=0.0011
epoch 10, validation loss=0.0011
epoch 11, validation loss=0.0012
epoch 12, validation loss=0.0011
epoch 13, validation loss=0.0013
epoch 14, validation loss=0.0010
epoch 15, validation loss=0.0010
epoch 16, validation loss=0.0010
epoch 17, validation loss=0.0010
epoch 18, validation loss=0.0010
epoch 19, validation loss=0.0009
training done.
testing...
loss=0.1135, accuracy=0.9666
testing done.
```

## Conclusion

Life is short, I use PyTorch.

CNN, yyds!