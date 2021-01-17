---
title: glibc Heap Learning
mathjax: true
date: 2021-01-16 22:18:04
tags: pwn
---

学了好几天的堆，今晚把已经看过的堆的知识记录一下。

## 什么是堆

系统用堆(Heap)来动态管理内存，堆从低地址向高地址生长。

一直听到堆栈的说法，其实堆跟栈区别真的很大的好吧：比如栈从高地址向低地址生长，内存较为固定，地址一直是`0x7ffff...`开头的，不能跟堆混为一谈吧。

堆的实现就是时间与空间达到权衡(trade off)的生动案例。后面我们会体会到。

堆想要效率高，就应该提高单次分配和释放的速率，同时也要减少内存空间利用的碎片化。

glibc中堆的管理器是ptmalloc2。我们在pwn学堆的时候，就学习ptmalloc2的堆管理。

## 堆的两个C语言高级函数

在C++里面是`new`跟`delete`，而在C语言里面是`malloc`跟`free`。

### malloc

```
void* malloc(size_t n);
return a pointer to the newly-allocated chunk.
```

### free

```
void free(void* p);
release the chunk pointed by the pointer p
```

## 堆底层的常用概念

### arena

arena可以理解为一个区域内的内存集合，可以看作是一片连续的内存空间。

在多线程中，每个线程都有一个专属的arena，主线程的arena就叫`main_arena`，后续做题经常见到。

主线程的arena通过系统调用`sbrk`创建，通过`brk`进行伸缩，其他线程的arena通过`mmap`来创建。

`main_arena`其实是由一个`struct malloc_state`来组织的，这个结构体里面储存了多种类型的bin和top chunk等内容。

### chunk

chunk即是`malloc`和`free`操作时，内存块的基本单位。

#### free chunk的结构

一个空闲的chunk不是都是unused area，而是在chunk的头部储存了很多信息，具体是这么储存的：

- prev_size：储存上一个chunk的size
- size：储存当前free chunk的size
- fd：下一个free chunk
- bk：上一个free chunk
- unused area

另外，注意到x86-64平台下，chunk都是每8个字节对齐的，所以chunk的大小也一定是8个字节的倍数，所以上面用来表示size的8个字节，就可以保证二进制表示下最后必有3个0。

而这3个0的位置，就被设计来分别储存3个信息：

- N：NON_MAIN_ARENA，1表示不是main_arena的，0代表是main_arena的。
- M：IS_MMAPPED，1代表该chunk是`mmap`出来的，0则不是。
- P：PREV_INUSE，1代表前面的chunk正在被使用，0则代表前面的chunk是空闲的。

#### allocated chunk的结构

allocated chunk的结构跟free chunk大体相似，不过也有不同：

- prev_size、size、NMP这前两个字段都是跟free chunk一样的。
- 没有fd和bk，从第三个字段开始即可开始储存数据。

注意一下，prev_size到底什么时候有必要？当可以与前面的chunk合并时有必要存在。

什么时候allocated chunk可以省去prev_size这一个字段的空间？当前面的chunk也是allocated的。

所以，在设计之中，allocated chunk之间是可以把prev_size那8个字节也用来存入数据，这样能多出8个字节的存储空间。

### top chunk

top chunk就是一个arena里面最后的那块chunk，不管怎样都会存在，作为一个arena的结束，不输入任何一个bin。

top chunk可以通过系统调用`brk`来变长变短，也可以在`malloc`过程中被切出一块去用，但是一直会存在。

### bin

bin是用来管理**空闲的chunk**的一个数据结构，通过单向或双向链表来进行组织。

通过将不同类型的chunk放进不同的bin中进行管理，能够提高`malloc`过程找到合适的chunk的速率。

#### fast bin

fast bin维护小型的内存块，将这些小内存块用于系统频繁的小型内存申请调用。

fast bin只有1组，也就是只有一条单向链表来维护。

fast bin中的free chunk有这么几个特点：

1. 不与其他的free chunk合并
2. 使用singly linked list进行组织
3. 采用Last In First Out Policy
4. 申请小内存时，最先在fast bin中寻找
5. 当被free时，不会将P位置0（PREV_INUSE）

一般0x20到0x7f大小的chunk，在free后并且分类后，会被丢进fast bin进行维护。

#### small bins

small bins有62组链表，负责维护相对较小的chunk。

small bins的free chunk就跟fast bin不同了：

1. 相同大小的chunk就会被放在同一组small bin之中
2. 使用doubly linked list维护
3. First In First Out
4. 当被free时，会诚实地记录P位
5. 并且，有条件时，会主动地合并成一个更大的free chunk

大小从0x80到0x400的chunk最后会被丢到small bins去维护。（大小小于1M）

#### large bins

large bins共有63组。每一组large bin储存的不是特定大小的chunk，而是大小处在一定范围的chunk。

记录的方法与small bin几乎相同。一样是FIFO，一样是双向链表，一样会主动合并。

不过有一点特殊：large bin中的chunk是按照从大到小进行排序的。

大于0x400即1M的chunk就会被安排到large bin里面去。

#### unsorted bin

unsorted bin可以通俗想象成是chunk的“垃圾桶”，任何大于0x80的chunk都会被丢进unsorted bin里面去。（太小的直接丢进fast bin里面维护）

unsorted bin中的chunk没有大小规定，也没有大小顺序，一切都是待整理状态。

在里面的chunk会通过后续的“捡垃圾”（即chunk维护整理工作）进入到专属的chunk。

与fast bin一样，unsorted bin也只有一组。也只是一个暂存的缓冲区域，该挑合适的chunk，还是去规定的bin找，万不得已最后才来搜垃圾堆嘛。。。

### 小知识

有一个原则：任意两个物理相邻的空闲chunk不能排在一起。（不过fast bin还是得除外的）

## 堆的工作流程

### malloc的工作流程

### free的工作流程

## 堆有关的攻击手段

### UAF

### Heap Overflow

### Unlink

### Fastbin Attack

