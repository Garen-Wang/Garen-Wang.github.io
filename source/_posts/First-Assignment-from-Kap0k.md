---
title: First Assignment from Kap0k
mathjax: true
date: 2021-01-16 00:26:45
tags: Kap0k, pwn
---

## 手撕shellcode

最后的结果是：

```
\x31\xc0\x50\x68\x66\x69\x6c\x65\x68\x74\x65\x73\x74\x89\xe3\x50\x53\x31\xc9\xb1\x02\xb0\x05\xcd\x80\x89\xc3\x31\xc0\x50\x68\x6f\x72\x6c\x64\x68\x6f\x2c\x20\x77\x68\x68\x65\x6c\x6c\x89\xe1\x50\x51\x31\xd2\xb2\x0c\xb0\x04\xcd\x80\x31\xdb\x31\xc0\xb0\x01\xcd\x80
```

### 最初的思路

查了很久资料，最后才在google上找到有用的东西。（用i386编译出来的）

最简单的写法自然是这样：

```
section .data
    msg db "Hello, world!", 0xa
    len equ $ - msg
    filename db "sb"

section .text
global _start
_start:
    ;xor edx, edx
    mov ecx, 2
    mov ebx, filename
    mov eax, 5
    int 0x80
    
    mov ebx, eax
    mov ecx, msg
    mov edx, 12
    mov eax, 4
    int 0x80

    mov ebx, 0
    mov eax, 1
    int 0x80
```

这里所运用到的是linux kernel里面的syscall指令，通过`int 0x80`的软中断来执行底层函数。

我们用到的有`sys_open`和`sys_write`两个函数，他们的用法如下：

```c++
4. sys_write
Syntax: ssize_t sys_write(unsigned int fd, const char * buf, size_t count)

Source: fs/read_write.c

Action: write to a file descriptor

Details:



5. sys_open
Syntax: int sys_open(const char * filename, int flags, int mode)

Source: fs/open.c

Action: open and possibly create a file or device

Details:
```
`sys_open`的第二个参数`flags`中，`0`代表只读，`1`代表只写，`2`代表可读写。

这里试了一下，第三个参数可以不用去控制，默认留0没问题。

然后`sys_open`的返回值是一个文件描述数字，这个概念可以参考stdin是0，stdout是1，反正就是一个在`sys_write`调用的时候，第一个参数填的值。

然后就是照着规定填好寄存器，最后`int 0x80`调用一下就可以执行函数了。最后再`sys_exit`退出就可以了。

编译命令：

```
$ nasm -f elf helloworld.asm
$ ld -m elf_i386 -s -o shellcode helloworld.o
```

不过这样编译过后会发现机器码里面一大堆都是`\x00`，不符合要求；并且存在常量字符串，没法在shellcode中跳到里面的奇妙地址来读取字符串。

### 解决方案

#### 去除\x00

我们通过几个技巧来实现：

1. `mov eax, 0`转而通过`mov eax, eax`来实现。
2. `mov eax, 1`转而通过`mov al, 1`来实现。（前提是eax高位也没问题）

#### 在shellcode中注入常量字符串

我们没法把我们想要的字符串在被注入的程序中找到，所以还是得存在栈里面。

不过怎么存呢？通过push来存。

然后就有非常强的技巧：将字符串翻转后变成十六进制编码，每8位每8位的push进去，最后从栈顶开始的字符串就是我们想要的字符串。

但是又有问题：这样会不会又产生`\x00`？

其实有可能，所以我们无论如何，长度都补齐到4的整数倍。这样就可以保证没有`\x00`了。

最终我的shellcode输出至名字为`testfile`的文件中，输入内容为`hello, world`。

缺点是`testfile`必须要先存在然后才能写进去，这应该和我在`sys_open`的时候，`flags`的取值有关系。有时间的话再去探究这个参数到底该怎么取。

{% qnimg First-Assignment-From-Kap0k/objdump.png %}

最后通过一个在网上找到的命令，直接提取出了机器码，生成了shellcode：

```
$ objdump -d ./shellcode|grep '[0-9a-f]:'|grep -v 'file'|cut -f2 -d:|cut -f1-6 -d' '|tr -s ' '|tr '\t' ' '|sed 's/ $//g'|sed 's/ /\\x/g'|paste -d '' -s |sed 's/^/"/'|sed 's/$/"/g'
```

