---
title: Convolution Neural Network Learning Notes
mathjax: true
date: 2021-01-25 13:00:29
tags: Deep-Learning
---

## Definition of Convolution Neural Network

Definition in Discrete Mathematics:

$$h(x) = (f*g)(x) = \sum_{t=-\infty}^{\infty} f(t)g(x-t)$$

Two-dimensional Definition(I: image, K: kernal, cross-correlation):

$$h(i,j) = (I*K)(i,j) = \sum_m \sum_n I(i-m,j-n)K(m,n)$$

However, our convolution here does not reverse kernal, which means actually:

$$h(i,j) = (I*K)(i,j) = \sum_m \sum_n I(i+m,j+n)K(m,n)$$

Without reversed kernal, the operation is exactly the matrix dot multiplication.

## Relevant Concepts

A kernal is a square matrix responsible for extracting a feature from input. When using multiple kernal, we can extract multiple features from the same picture sample.

The size of kernal is commonly an odd number, and especially there exists 1*1 kernal.

The set of convolution kernals is called Filter. The number of kernal in a filter is usually euqal to that of input channels. For example, when processing RGB pictures, we usually use three kernals to calculate with corresponding channels, and these three kernals can be included in a filter.

Similarly with neural network learnt before, there is a bias corresponding with each filter, whose size is the same as the output size of the filter.

Several filters and their corresponding bias matrices consist of a WeightsBias.

Stride is a parameter of a convolution layer, which stands for the increment of coordination of width and height after each update is done. By default the stride is set 1. Obviously, the bigger the stride, the smaller the output size.

Padding is used when we want to control the output size. When padding is needed, we will add several layer of zeros on the edge of original matrix, thus incrementing the size. By default the padding is 0. On the contrary, the bigger the padding, the bigger the output size.

## Size Calculation

Actually we can calculate the width and height of output:

$$Width_{out} = \lfloor \frac{Width_{in} - Width_{K} + 2Padding}{Stride} \rfloor + 1$$

$$Height_{out} = \lfloor \frac{Height_{in} - Height_{K} + 2Padding}{Stride} \rfloor+ 1$$

## About PyTorch

When retrieving data from the dataloader previously loaded, the dimension of the input tensor is 4, respectively:

1. batch size: int, one part of hyper-parameter
2. input channels: int, the number of channels of data(gray-scale: 1, RGB: 3)
3. width: int, consistent with dataset
4. height: int, consistent with dataset

The number of first dimension remains unchanged during the whole forward process. However, input channels will be changed according to our design of convolution layers. Width and height can be calculated by applying the formulas above.

When `LayerChoice` and `InputChoice` are used in definition of model, we must guarantee each calculation is meaningful rather than size dismatched.

```python
class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.conv1 = mutables.LayerChoice(OrderedDict([
            ("conv3*3", nn.Conv2d(3, 8, 3, 1)),
            ("conv5*5", nn.Conv2d(3, 8, 5, 1))
        ]), key='conv1')
        self.mid_conv = mutables.LayerChoice([
            nn.Conv2d(8, 8, 3, 1, padding=1),
            nn.Conv2d(8, 8, 5, 1, padding=2)
        ], key='mid_conv')
        self.conv2 = nn.Conv2d(8, 16, 5, 1)
        self.pool = nn.MaxPool2d(2, 2)
        self.func1 = nn.Linear(16 * 5 * 5, 120)
        self.func2 = nn.Linear(120, 84)
        self.func3 = nn.Linear(84, 10)
        self.input_switch = mutables.InputChoice(n_candidates=2, n_chosen=1, key="skip_conv")

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        old_x = x
        zero_x = torch.zeros_like(old_x)
        skip_x = self.input_switch([zero_x, old_x])
        x = F.relu(self.mid_conv(x))
        x += skip_x
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.func1(x))
        x = F.relu(self.func2(x))
        x = self.func3(x)
        return x

```

In this example, dataset is CIFAR-10, where all samples are 32*32.

When`x = self.conv1(x)`, now the size may be 30 or 28. After 2*2 pooling, the size(width and height) may be 15 or 14.

Here we must make the size unchanged after `x = self.mid_conv(x)` since it is a layer allowed to be skipped. And we can see when kernal size is 3, padding is 1 and kernal size equals 5, padding euqals 2, width and height both remain unchanged.

After `x = self.conv2(x)`, the size shrinks to 10 or 11. After max-pooling, the size becomes 5 as expected.

