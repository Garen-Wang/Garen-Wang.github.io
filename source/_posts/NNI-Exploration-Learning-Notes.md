---
title: NNI Exploration Learning Notes
mathjax: true
date: 2021-01-23 23:19:35
tags: NNI
---

## 未完成任务

### Task 2.2

-[] HPO
-[] 在搜索空间中选择随机结构，并验证性能
-[] NAS

### Task 3.1

-[] 跑通NNI Feature Engineering Sample

### Task 3.2

#### Task 3.2.1

#### Task 3.2.2

### Task 4


## HPO

超参调优在NNI中比较好实现，只要有参数和模型的搜索空间，就可以利用NNI自带的tuner来做调参工作。

### Assessor

在数据量较大的情况下，一般一个trial普遍会比较久，NNI支持Assessor，实现在调优过程中类似“剪枝”的功能，提供了提前终止某些trial的策略以节省实验时间。

需要添加assessor时只需在`config.yml`中添加，这里以Curvefitting为例：

```yaml
assessor:
  builtinAssessorName: Curvefitting
  classArgs:
    epoch_num: 10
    threshold: 0.9

```

## NAS

### 搜索空间的编写

在做NAS的过程中，我们需要手动写出待搜索的模型的类，我们借助NNI中的mutables来实现模型搜索空间的构建。

```python
class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.conv1 = mutables.LayerChoice([
            nn.Conv2d(3, 8, 3, 1, padding=1),
            nn.Conv2d(3, 8, 5, 1, padding=2)
        ], key='conv1')
        self.mid_conv = mutables.LayerChoice([
            nn.Conv2d(8, 8, 3, 1, padding=1),
            nn.Conv2d(8, 8, 5, 1, padding=2)
        ], key='mid_conv')
        self.conv2 = nn.Conv2d(8, 16, 5, 1)
        self.pool = nn.MaxPool2d(2, 2)
        self.func1 = nn.Linear(16 * 5 * 5, 120)
        self.func2 = nn.Linear(120, 84)
        self.func3 = nn.Linear(84, 10)
        self.input_switch = mutables.InputChoice(n_candidates=1, key='skip')

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.mid_conv(x)
        skip_x = self.input_switch([x])
        x = self.conv2(x)
        if skip_x is not None:
            x = x + skip_x
        x = self.pool(F.relu(x))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.func1(x))
        x = F.relu(self.func2(x))
        x = self.func3(x)
        return x
```

`mutables.LayerChoice`实现了神经网络模型中一层的多选一，待选的神经网络层只需要在里面列出来即可。例如上面的代码，就实现了3\*3和5\*5两种二维卷积层的选择空间。

`mutables.InputChoice`实现了可跳过连接。在上述代码中，表示了mid_conv层是可跳过层。可跳过层的前后代码保持不变，在可跳过层则需要从可能连接加入到后一层的输出中。

### Classical NAS

### One-shot NAS

### DARTS


### ENAS

