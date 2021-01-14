---
title: Convolution Neural Network Learning Notes
mathjax: true
date: 2021-01-10 16:02:21
tags: Deep-Learning
---

when describing a "weights-bias", we use four dimensions:
1. the number of filters(euqal to the number of features to output)
2. the number of kernals(equal to the number of input channels; RGB: 3, gray: 1)
3. the first dimension of the kernal
4. the second dimension of the kernal

a "weights-bias" consist of multiple filters plus a bias.

While using PyTorch, you can leave the first dimension -1, which means undecided, so that this number can be calculated with the help of the rest known dimension.

