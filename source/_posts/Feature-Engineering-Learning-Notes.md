---
title: Feature Engineering Learning Notes
mathjax: true
date: 2021-01-27 15:06:32
tags: Machine-Learning
---

## 定义

### 特征的定义

feature就是从数据中提取出来的有用的属性。

### 特征工程的定义

特征工程(Feature Engineering)是机器学习中的一个重要分支，指的是通过运用多种数据处理方法，将把原始数据转化成更好的特征的过程。

特征有优劣之分，更好的特征更适合机器学习，意味着能够训练出更好的结果。

## 特征处理

### 去除异常数据

特征清洗即在数据中去除异常数据。常见的去除异常数据方式可以基于简单统计方法借助$3\delta$原则来去除，也可以用KNN算法等内容来处理。

### 处理缺失数据

拿数据来训练自然需要各类数据数量较均匀，有缺失会对模型准确度造成影响。

至于如何处理缺失数据，有几个原则：

1. 该类数据缺失得太多了，干脆全部丢弃。
2. 缺失得不多的话，可以利用均值或中位数补充少量数据。
3. 利用其他的算法进行缺失数据的预测，做prediction然后补齐。

### 数据采样及均衡操作

做分类任务的话，正负样本要求数量较均衡，如果给定数据不均衡的话就需要数据采样操作。

数据采样的操作主要有两种：上采样和下采样。

#### 下采样

当正负两类数据规模都比较大时，可以适当对数据多的那一类进行欠采样。

#### 上采样

当正负两类规模都比较小时，应该对数据少的那一类做过采样操作，经常可以用到一个叫SMOTE的过采样算法来合成新样本，使得两类规模相当。

### 特征预处理

#### 数值数据

针对普通数值型数据，一般可以使用MinMax或者标准化来做无量纲化操作。

这里的所谓标准化方法，就是处理出均值和方差，每个数据就表示成跟均值差了多少个方差（带正负符号）。

两种方法分别可以在`sklearn.preprocessing`的`StandardScaler`和`MinMaxScaler`找到。

#### 分类数据

针对分类数据，经常需要转化成OneHot编码，这个操作可以在`pandas`或者`sklearn`里做到。

```python
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

data = pd.DataFrame({'age': [4, 6, 3, 3], 'pet': ['cat', 'dog', 'dog', 'fish']})
# method 1
pet_values = LabelEncoder.fit_transform(data.pet) # [0, 1, 1, 2]，即离散化
OneHotEncoder().fit_transform(pet_values.reshape(-1, 1)).toarray()
# method 2
pd.get_dummies(data,columns=['pet'])

```

#### 时间数据

时间数据最简便的是用`pandas`中的`DatetimeIndex`直接做。
