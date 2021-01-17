---
title: CSAPP Bomb Lab Writeup
mathjax: true
date: 2021-01-13 23:40:47
tags: CSAPP
---

这是CSAPP的bomblab，对打pwn的新手补补基础还是非常有用的，尤其是各种汇编操作和IDA Pro里各种各样的奇妙语法，更是让我这个菜鸡大开眼界（还能这么坑……）

前五关非常的常规，我们通过汇编跟反汇编都看一下。

第六关我不行了，就通过反汇编的C代码走一走。

做了一个晚上加半个早上，终于搞定了，是我太菜……

## phase 1

### 汇编

```
0000000000400ee0 <phase_1>:
  400ee0:	48 83 ec 08          	sub    $0x8,%rsp
  400ee4:	be 00 24 40 00       	mov    $0x402400,%esi
  400ee9:	e8 4a 04 00 00       	callq  401338 <strings_not_equal>
  400eee:	85 c0                	test   %eax,%eax
  400ef0:	74 05                	je     400ef7 <phase_1+0x17>
  400ef2:	e8 43 05 00 00       	callq  40143a <explode_bomb>
  400ef7:	48 83 c4 08          	add    $0x8,%rsp
  400efb:	c3                   	retq   
```

其中0x402400这个地址很奇妙，我们用gdb跟进去看一看：

{% qnimg CSAPP-Bomb-Lab-Writeup/phase1.png %}

这里的`test`跟`je`两个汇编语句是连接在一起的，一般就像是这样用的：

```
test %rax, %rax
je 0x??????
```
`test`语句本质就是一个`and`，不过用`test`的话不会去改变%rax的值，而会直接放到下面来进行比较。

这两句汇编的意思就是%rax值等于0时就跳转，否则不跳转，执行下一条命令。

就是比较字符串相等就可以进入下一步了。

所以只需要保证输入的字符串是`"Border relations with Canada have never been better."`，就可以了。

### IDA

{% qnimg CSAPP-Bomb-Lab-Writeup/phase1(IDA).png %}

用IDA的话一眼看出来，就不用分析了。

## phase 2

### 汇编
```
0000000000400efc <phase_2>:
  400efc:	55                   	push   %rbp
  400efd:	53                   	push   %rbx
  400efe:	48 83 ec 28          	sub    $0x28,%rsp
  400f02:	48 89 e6             	mov    %rsp,%rsi
  400f05:	e8 52 05 00 00       	callq  40145c <read_six_numbers>
  400f0a:	83 3c 24 01          	cmpl   $0x1,(%rsp)
  400f0e:	74 20                	je     400f30 <phase_2+0x34>
  400f10:	e8 25 05 00 00       	callq  40143a <explode_bomb>
  400f15:	eb 19                	jmp    400f30 <phase_2+0x34>
  400f17:	8b 43 fc             	mov    -0x4(%rbx),%eax
  400f1a:	01 c0                	add    %eax,%eax
  400f1c:	39 03                	cmp    %eax,(%rbx)
  400f1e:	74 05                	je     400f25 <phase_2+0x29>
  400f20:	e8 15 05 00 00       	callq  40143a <explode_bomb>
  400f25:	48 83 c3 04          	add    $0x4,%rbx
  400f29:	48 39 eb             	cmp    %rbp,%rbx
  400f2c:	75 e9                	jne    400f17 <phase_2+0x1b>
  400f2e:	eb 0c                	jmp    400f3c <phase_2+0x40>
  400f30:	48 8d 5c 24 04       	lea    0x4(%rsp),%rbx
  400f35:	48 8d 6c 24 18       	lea    0x18(%rsp),%rbp
  400f3a:	eb db                	jmp    400f17 <phase_2+0x1b>
  400f3c:	48 83 c4 28          	add    $0x28,%rsp
  400f40:	5b                   	pop    %rbx
  400f41:	5d                   	pop    %rbp
  400f42:	c3                   	retq   
```
按照汇编来分析，stack frame的构造如下：

```
0x00(rsp)
0x04(rbp)
0x08      rbp
0x1c          [5]
0x10          [4]
0x14          [3]
0x18          [2]
0x1c          [1] <- rbx
0x20 rsp  rsi [0] <- rax
```

在从`rsp - 0x20`到`rsp - 0x08`遍历的过程中，rax永远在栈上比rbx的地址小个4，也就是一个`int`的位置。每次check之后依次往后移一位。

我们需要满足的是两倍的rax等于rbx，也就是我们输入的数列是成倍增长的。

还有一个条件：读入到`rsp - 0x20`，也就是第一个数字，必须是1。

所以最终的输入就是`1 2 4 8 16 32`。

### IDA

{% qnimg CSAPP-Bomb-Lab-Writeup/phase2(IDA).png %}

输入六个整数，需要符合里面的这个规则：
```c++
  do
  {
    result = (unsigned int)(2 * *((_DWORD *)v2 - 1));
    if ( *(_DWORD *)v2 != (_DWORD)result )
      explode_bomb();
    v2 += 4;
  }
  while(v2 != v5);
```
这里需要注意：在第三行的代码里，`v2`先被强制类型转换为`DWORD*`，然后再执行减1的操作。

因为`v2`的指针类型在减1之前已经确定，所以实际上`*((_DWORD *)v2 - 1)`就相当于`*(_DWORD *)(v2 - 4)`，也就是数组里面的上一个元素。

所以六个整数，只需要满足后一个是前一个的两倍，就可以了。

## phase 3

### IDA

非常简单，switch里面提供了8个配套选择，任选一个即可过关。

{% qnimg CSAPP-Bomb-Lab-Writeup/phase3(IDA).png %}
### 汇编

然而这个关卡的话看汇编会比较难看出来。这也是这一关的价值所在。

```
0000000000400f43 <phase_3>:
  400f43:	48 83 ec 18          	sub    $0x18,%rsp
  400f47:	48 8d 4c 24 0c       	lea    0xc(%rsp),%rcx
  400f4c:	48 8d 54 24 08       	lea    0x8(%rsp),%rdx
  400f51:	be cf 25 40 00       	mov    $0x4025cf,%esi
  400f56:	b8 00 00 00 00       	mov    $0x0,%eax
  400f5b:	e8 90 fc ff ff       	callq  400bf0 <__isoc99_sscanf@plt>
  400f60:	83 f8 01             	cmp    $0x1,%eax
  400f63:	7f 05                	jg     400f6a <phase_3+0x27>
  400f65:	e8 d0 04 00 00       	callq  40143a <explode_bomb>
  400f6a:	83 7c 24 08 07       	cmpl   $0x7,0x8(%rsp)
  400f6f:	77 3c                	ja     400fad <phase_3+0x6a>
  400f71:	8b 44 24 08          	mov    0x8(%rsp),%eax
  400f75:	ff 24 c5 70 24 40 00 	jmpq   *0x402470(,%rax,8)
  400f7c:	b8 cf 00 00 00       	mov    $0xcf,%eax
  400f81:	eb 3b                	jmp    400fbe <phase_3+0x7b>
  400f83:	b8 c3 02 00 00       	mov    $0x2c3,%eax
  400f88:	eb 34                	jmp    400fbe <phase_3+0x7b>
  400f8a:	b8 00 01 00 00       	mov    $0x100,%eax
  400f8f:	eb 2d                	jmp    400fbe <phase_3+0x7b>
  400f91:	b8 85 01 00 00       	mov    $0x185,%eax
  400f96:	eb 26                	jmp    400fbe <phase_3+0x7b>
  400f98:	b8 ce 00 00 00       	mov    $0xce,%eax
  400f9d:	eb 1f                	jmp    400fbe <phase_3+0x7b>
  400f9f:	b8 aa 02 00 00       	mov    $0x2aa,%eax
  400fa4:	eb 18                	jmp    400fbe <phase_3+0x7b>
  400fa6:	b8 47 01 00 00       	mov    $0x147,%eax
  400fab:	eb 11                	jmp    400fbe <phase_3+0x7b>
  400fad:	e8 88 04 00 00       	callq  40143a <explode_bomb>
  400fb2:	b8 00 00 00 00       	mov    $0x0,%eax
  400fb7:	eb 05                	jmp    400fbe <phase_3+0x7b>
  400fb9:	b8 37 01 00 00       	mov    $0x137,%eax
  400fbe:	3b 44 24 0c          	cmp    0xc(%rsp),%eax
  400fc2:	74 05                	je     400fc9 <phase_3+0x86>
  400fc4:	e8 71 04 00 00       	callq  40143a <explode_bomb>
  400fc9:	48 83 c4 18          	add    $0x18,%rsp
  400fcd:	c3                   	retq   
```
stack frame大概长这样：
```
0x00(rsp)
0x04
0x08
0x0c rcx [1]
0x10 rdx [0]
0x14
0x18 rsp
```
发现了第一个奇妙地址0x4025cf，我们也用gdb看看：

{% qnimg CSAPP-Bomb-Lab-Writeup/phase3_disass.png %}

害……

不过这里有另一个奇妙地址，其实这句话就是switch汇编实现的核心：

```
  400f75:	ff 24 c5 70 24 40 00 	jmpq   *0x402470(,%rax,8)
```

穿插复习下括号里两个数字和三个数字的表示法：

- (a, b) = a + b
- (a, b, c) = a + b * c

这种括号的表示方法不只在lea指令里面能用，在其他指令里也能见到。

再查一查0x402470这个地址的值，还有后面几个地址的值：

{% qnimg CSAPP-Bomb-Lab-Writeup/phase3_switch.png %}

```
(gdb) p/x *0x402470
$9 = 0x400f7c
(gdb) p/x *0x402478
$10 = 0x400fb9
(gdb) p/x *0x402480
$11 = 0x400f83
(gdb) p/x *0x402488
$12 = 0x400f8a
(gdb) p/x *0x402490
$13 = 0x400f91
(gdb) p/x *0x402498
$14 = 0x400f98
(gdb) p/x *0x4024a0
$15 = 0x400f9f
(gdb) p/x *0x4024a8
$16 = 0x400fa6
(gdb) p/x *0x4024b0
$17 = 0x7564616d
```

可以发现，从0x402470开始储存的是一个指针数组，因为是64位，所以地址自然是8个字节8个字节间隔的。

并且，这个数组里的指针指向的值，都是`phase_3`函数的mov指令，即对应了switch语句中的不同分支。

> 说句题外话，之所以switch中每个case的最后一般都得加一个`break`，就是因为在底层就是这样实现的。如果不加`break`，在每一句执行后就不会`jmp`出这个switch的判断，在这里就可能%eax被多次赋值。所以该加`break`还是得加的哦！

一一对应后，可以梳理出能够通过的8个输出：

```
0: 0xcf
1: 0x137
2: 0x2c3
3: 0x100
4: 0x185
5: 0xce
6: 0x2aa
7: 0x147
```

任选其一，就能通过第三关。

## phase 4

### IDA

{% qnimg CSAPP-Bomb-Lab-Writeup/phase4(IDA).png %}

这个部分我们需要保证第一个读入的整数`v3`小于等于14的同时，`func4(v3, 0, 14)`也等于0，第二个读入的整数`v4`也要等于0。

{% qnimg CSAPP-Bomb-Lab-Writeup/func4(IDA).png %}

而要使这个函数的返回值为0，只需要让`a1 = v3 = (14 - 0) / 2 + 0 = 7`。

### 汇编

然而汇编并不像IDA反汇编出来的这样清晰，这一关一眼看上去可能眼花，认真看就好了。

```
000000000040100c <phase_4>:
  40100c:	48 83 ec 18          	sub    $0x18,%rsp
  401010:	48 8d 4c 24 0c       	lea    0xc(%rsp),%rcx
  401015:	48 8d 54 24 08       	lea    0x8(%rsp),%rdx
  40101a:	be cf 25 40 00       	mov    $0x4025cf,%esi
  40101f:	b8 00 00 00 00       	mov    $0x0,%eax
  401024:	e8 c7 fb ff ff       	callq  400bf0 <__isoc99_sscanf@plt>
  401029:	83 f8 02             	cmp    $0x2,%eax
  40102c:	75 07                	jne    401035 <phase_4+0x29>
  40102e:	83 7c 24 08 0e       	cmpl   $0xe,0x8(%rsp)
  401033:	76 05                	jbe    40103a <phase_4+0x2e>
  401035:	e8 00 04 00 00       	callq  40143a <explode_bomb>
  40103a:	ba 0e 00 00 00       	mov    $0xe,%edx
  40103f:	be 00 00 00 00       	mov    $0x0,%esi
  401044:	8b 7c 24 08          	mov    0x8(%rsp),%edi
  401048:	e8 81 ff ff ff       	callq  400fce <func4>
  40104d:	85 c0                	test   %eax,%eax
  40104f:	75 07                	jne    401058 <phase_4+0x4c>
  401051:	83 7c 24 0c 00       	cmpl   $0x0,0xc(%rsp)
  401056:	74 05                	je     40105d <phase_4+0x51>
  401058:	e8 dd 03 00 00       	callq  40143a <explode_bomb>
  40105d:	48 83 c4 18          	add    $0x18,%rsp
  401061:	c3                   	retq   
```

栈布局是这样的：

```
0x00(rsp)
0x04
0x08
0x0c rcx [1]
0x10 rdx [0]
0x14
0x18 rsp
```

在这里需要满足的有：
- `0xe >= *(rsp + 0x8)`
- `0x0 == *(rsp + 0xc)`
- `func4(*(rsp + 0x8), 0, 0xe) == 0`

我们进入`func4`看看汇编：

```
0000000000400fce <func4>:
  400fce:	48 83 ec 08          	sub    $0x8,%rsp
  400fd2:	89 d0                	mov    %edx,%eax
  400fd4:	29 f0                	sub    %esi,%eax
  400fd6:	89 c1                	mov    %eax,%ecx
  400fd8:	c1 e9 1f             	shr    $0x1f,%ecx
  400fdb:	01 c8                	add    %ecx,%eax
  400fdd:	d1 f8                	sar    %eax
  400fdf:	8d 0c 30             	lea    (%rax,%rsi,1),%ecx
  400fe2:	39 f9                	cmp    %edi,%ecx
  400fe4:	7e 0c                	jle    400ff2 <func4+0x24>
  400fe6:	8d 51 ff             	lea    -0x1(%rcx),%edx
  400fe9:	e8 e0 ff ff ff       	callq  400fce <func4>
  400fee:	01 c0                	add    %eax,%eax
  400ff0:	eb 15                	jmp    401007 <func4+0x39>
  400ff2:	b8 00 00 00 00       	mov    $0x0,%eax
  400ff7:	39 f9                	cmp    %edi,%ecx
  400ff9:	7d 0c                	jge    401007 <func4+0x39>
  400ffb:	8d 71 01             	lea    0x1(%rcx),%esi
  400ffe:	e8 cb ff ff ff       	callq  400fce <func4>
  401003:	8d 44 00 01          	lea    0x1(%rax,%rax,1),%eax
  401007:	48 83 c4 08          	add    $0x8,%rsp
  40100b:	c3                   	retq   
```

没有什么栈的布局，就是些寄存器之间的计算，我们一个一个模拟一下：

（初始化：rdi = ?, rsi = 0, rdx = 0xe）

1. eax = edx,  eax = 0xe
2. eax -= esi, eax = 0xe
3. ecx = eax,  ecx = 0xe
4. ecx >>= 0x1f, ecx >>= 31, ecx = 0（注意是逻辑右移）
5. eax += ecx, eax = 0xe
6. eax >>= 1, eax = 0x7（注意是算术右移，且只有一个参数时默认右移1位）
7. ecx = rax + rsi * 1 = 0x7 + 0 = 0x7

然后我们分析下后面跳转的流程：

- 如果%edi <= %ecx，就会跳转到0x400ff2去。
- 跳转完再来一个cmp，如果%edi >= %ecx，就可以调到0x401007结束函数了。

所以只需要%ecx和%edi一样大就可以了，所以rdi直接等于7就可以了。

所以我们直接输入7跟0就可以了。

所以最后复习下这些奇妙的汇编指令，以免我又忘了：

- `imul src, dest` 乘法
- `sal  src, dest` 算术左移
- `sar  src, dest` 算术右移
- `shl  src, dest` 逻辑左移
- `shr  src, dest` 逻辑右移

## phase 5

### 汇编

```
0000000000401062 <phase_5>:
  401062:	53                   	push   %rbx
  401063:	48 83 ec 20          	sub    $0x20,%rsp
  401067:	48 89 fb             	mov    %rdi,%rbx
  40106a:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
  401071:	00 00 
  401073:	48 89 44 24 18       	mov    %rax,0x18(%rsp)
  401078:	31 c0                	xor    %eax,%eax
  40107a:	e8 9c 02 00 00       	callq  40131b <string_length>
  40107f:	83 f8 06             	cmp    $0x6,%eax
  401082:	74 4e                	je     4010d2 <phase_5+0x70>
  401084:	e8 b1 03 00 00       	callq  40143a <explode_bomb>
  401089:	eb 47                	jmp    4010d2 <phase_5+0x70>
  40108b:	0f b6 0c 03          	movzbl (%rbx,%rax,1),%ecx
  40108f:	88 0c 24             	mov    %cl,(%rsp)
  401092:	48 8b 14 24          	mov    (%rsp),%rdx
  401096:	83 e2 0f             	and    $0xf,%edx
  401099:	0f b6 92 b0 24 40 00 	movzbl 0x4024b0(%rdx),%edx
  4010a0:	88 54 04 10          	mov    %dl,0x10(%rsp,%rax,1)
  4010a4:	48 83 c0 01          	add    $0x1,%rax
  4010a8:	48 83 f8 06          	cmp    $0x6,%rax
  4010ac:	75 dd                	jne    40108b <phase_5+0x29>
  4010ae:	c6 44 24 16 00       	movb   $0x0,0x16(%rsp)
  4010b3:	be 5e 24 40 00       	mov    $0x40245e,%esi
  4010b8:	48 8d 7c 24 10       	lea    0x10(%rsp),%rdi
  4010bd:	e8 76 02 00 00       	callq  401338 <strings_not_equal>
  4010c2:	85 c0                	test   %eax,%eax
  4010c4:	74 13                	je     4010d9 <phase_5+0x77>
  4010c6:	e8 6f 03 00 00       	callq  40143a <explode_bomb>
  4010cb:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
  4010d0:	eb 07                	jmp    4010d9 <phase_5+0x77>
  4010d2:	b8 00 00 00 00       	mov    $0x0,%eax
  4010d7:	eb b2                	jmp    40108b <phase_5+0x29>
  4010d9:	48 8b 44 24 18       	mov    0x18(%rsp),%rax
  4010de:	64 48 33 04 25 28 00 	xor    %fs:0x28,%rax
  4010e5:	00 00 
  4010e7:	74 05                	je     4010ee <phase_5+0x8c>
  4010e9:	e8 42 fa ff ff       	callq  400b30 <__stack_chk_fail@plt>
  4010ee:	48 83 c4 20          	add    $0x20,%rsp
  4010f2:	5b                   	pop    %rbx
  4010f3:	c3                   	retq   
```

这个函数的stack frame是这样的：

```
phase 5
0x00 (rsp)
0x08 canary
0x10 rdi
0x18
0x20 rsp
```
同样有奇妙地址，我们查一查：

{% qnimg CSAPP-Bomb-Lab-Writeup/phase5_str.png %}

这个字符串打印出来之所以这样，是因为它最后一位不是`\x00`，所以就连续着把紧连着的下一个字符串也输出出来了。

最开始在call出`string_length`之前的这部分是用来初始化canary的。不用管。

字符串长度必须为6，才能跳转，不然会踩雷。

接下来从0x40108b开始，就是一个6次的循环，rax充当循环的counter，很容易看出来。

如果我们过完这个循环，最终要满足的是这个条件：
```
  4010ae:	c6 44 24 16 00       	movb   $0x0,0x16(%rsp)
  4010b3:	be 5e 24 40 00       	mov    $0x40245e,%esi
  4010b8:	48 8d 7c 24 10       	lea    0x10(%rsp),%rdi
  4010bd:	e8 76 02 00 00       	callq  401338 <strings_not_equal>
  4010c2:	85 c0                	test   %eax,%eax
  4010c4:	74 13                	je     4010d9 <phase_5+0x77>
```

所以我们要做的，就是在跑完上面这次循环之后，让`rsp + 0x10`开始的字符串跟`flyers`一毛一样。

这段代码粘下来集中看一看：

```
  40108b:	0f b6 0c 03          	movzbl (%rbx,%rax,1),%ecx
  40108f:	88 0c 24             	mov    %cl,(%rsp)
  401092:	48 8b 14 24          	mov    (%rsp),%rdx
  401096:	83 e2 0f             	and    $0xf,%edx
  401099:	0f b6 92 b0 24 40 00 	movzbl 0x4024b0(%rdx),%edx
  4010a0:	88 54 04 10          	mov    %dl,0x10(%rsp,%rax,1)
  4010a4:	48 83 c0 01          	add    $0x1,%rax
  4010a8:	48 83 f8 06          	cmp    $0x6,%rax
  4010ac:	75 dd                	jne    40108b <phase_5+0x29>
```

开始模拟：

（初始化rbx指向的是最开始的rdi，也就是字符串的开始）

```
ecx = str[i]
*rsp = cl (lower 4 digits of str[i])
rdx = *rsp = cl (lower 4 digits of str[i])
edx &= 0xf
edx = array3449[cl]
*(rsp + rax + 0x10) = dl (lower 4 digits of array3449[cl])
```

最后的这个`(rsp + rax + 0x10)`看上去不认识，但是参照下上面的栈结构，其实表示的就是字符串的第i位。

所以我们只需要去注意输入的6个字符中，每个字符的低4位在`array3449`中索引出来的值，这些值就会一个一个的，填到以`rsp + 0x10`为开始的字符串中。

手动数一数下标，就可以发现，要对应弄出`flyers`，我们依次需要下标是`9 15 14 5 6 7`。

所以我们只需要翻翻ASCII表，找到低4位是这些的字符，拼到一起就可以了。

我最终的答案是`ionefg`。答案不唯一。

### IDA

{% qnimg CSAPP-Bomb-Lab-Writeup/phase5(IDA).png %}

主要是这句代码太具有迷惑性：

```c
v3[i] = array_3449[*(_BYTE *)(a1 + i) & 0xF];
```

正确的解读是：

```c
v3[i] = array_3449[a1[i] & 0xF];
```

在C里面，一个char所占据的大小恰好就是一个byte，所以`_BYTE`可以直接看成`char`。

这里我之所以迷糊，是因为IDA Pro反汇编说`a1`的类型是`int64`，然而事实上`a1`就是个字符串。

## phase 6

最后一关，太复杂了！那我们就不分析汇编，直接上手看IDA Pro弄出来的代码。

其实弄出来的代码也不好看懂，一不小心也很容易晕！这里重新做一下记录。

### IDA

反汇编出来的代码长这样，非常长，变量非常多。

```c++
__int64 __fastcall phase_6(__int64 a1)
{
  int *v1; // r13
  signed int v2; // er12
  signed int v3; // ebx
  char *v4; // rax
  unsigned __int64 v5; // rsi
  _QWORD *v6; // rdx
  signed int v7; // eax
  int v8; // ecx
  __int64 v9; // rbx
  char *v10; // rax
  __int64 i; // rcx
  __int64 v12; // rdx
  signed int v13; // ebp
  __int64 result; // rax
  int v15[6]; // [rsp+0h] [rbp-78h]
  char v16; // [rsp+18h] [rbp-60h]
  __int64 v17; // [rsp+20h] [rbp-58h]
  char v18; // [rsp+28h] [rbp-50h]
  char v19; // [rsp+50h] [rbp-28h]

  v1 = v15;
  read_six_numbers(a1, v15);
  v2 = 0;
  while ( 1 )
  {
    if ( (unsigned int)(*v1 - 1) > 5 )
      explode_bomb(a1, v15);
    if ( ++v2 == 6 )
      break;
    v3 = v2;
    do
    {
      if ( *v1 == v15[v3] )
        explode_bomb(a1, v15);
      ++v3;
    }
    while ( v3 <= 5 );
    ++v1;
  }
  v4 = (char *)v15;
  do
  {
    *(_DWORD *)v4 = 7 - *(_DWORD *)v4;
    v4 += 4;
  }
  while ( v4 != &v16 );
  v5 = 0LL;
  do
  {
    v8 = v15[v5 / 4];
    if ( v8 <= 1 )
    {
      v6 = &node1;
    }
    else
    {
      v7 = 1;
      v6 = &node1;
      do
      {
        v6 = (_QWORD *)v6[1];
        ++v7;
      }
      while ( v7 != v8 );
    }
    *(__int64 *)((char *)&v17 + 2 * v5) = (__int64)v6;
    v5 += 4LL;
  }
  while ( v5 != 24 );
  v9 = v17;
  v10 = &v18;
  for ( i = v17; ; i = v12 )
  {
    v12 = *(_QWORD *)v10;
    *(_QWORD *)(i + 8) = *(_QWORD *)v10;
    v10 += 8;
    if ( v10 == &v19 )
      break;
  }
  *(_QWORD *)(v12 + 8) = 0LL;
  v13 = 5;
  do
  {
    result = **(unsigned int **)(v9 + 8);
    if ( *(_DWORD *)v9 < (signed int)result )
      explode_bomb(a1, &v19);
    v9 = *(_QWORD *)(v9 + 8);
    --v13;
  }
  while ( v13 );
  return result;
}
```

首先我们画一画这个函数的栈：

```
0x00 rbp
0x08
0x10
0x18
0x20
0x28 char v19[0x28] 0ll
0x30               &node[v15[5]]
0x38               &node[v15[4]]
0x40               &node[v15[3]]
0x48               &node[v15[2]]
0x50 char v18      &node[v15[1]]  <- v10
0x58 long long v17 &node[v15[0]]  v9
0x60 char v16
0x64 v15[5]
0x68 v15[4]
0x6c v15[3]
0x70 v15[2]
0x74 v15[1]
0x78 v15[0]
```

这个栈的图片非常非常重要，首先先保证不会乱，因为后面还有跳出栈外的过程。

还有，在分析的过程中，时刻注意每一个变量到底是值，还是指针！千万不能错！

一步一步分析，不要急，一定要慢慢来：

最开始，从`v15`开始，读入6个`int`类型的整数，存在栈上。（`v15`是个指针）

第一个是嵌套循环，`v1`是当前遍历到的元素的指针，`v2`表示第几个元素（从1开始数），`v3`是循环变量。

每次遍历`v1`，都必须保证`1 <= *v1 <= 6`，关于强转unsigned int的知识点，在最后有总结。然后内层循环表示后面的元素都得跟前面的不一样，意思就是这6个数各不相同。

第二个是单个do-while循环。它做的就是把这6个数都运算一遍，把`x`变成了`7-x`，更改了这6个数。

第三个开始烧脑了！`v8`是循环中被遍历到的值，根据`v8`的数值大小，分别执行若干次从`&node1`开始的`v8 - 1`次地址跳转，最终把栈上原来数组的值重新写为跳转到最后的地址。

这里注意一下，`v6 = (_QWORD *)v6[1];`这句代码是伏笔！（为什么这个值可以强转为地址呢？）

我们点进`node1`，发现在data段，后面刚好延伸到`node6`结束，这是什么意思？

不懂，我们看到下一个代码部分：

一个for循环，从`v17`即`rbp - 0x58`开始，每次循环结束会跳转到`v10`的值。之所以可以直接迭代为`v10`的值，是因为这个数组在第三次操作的时候已经变成了指针数组了！

接下来又是一句意味深长的代码：`*(_QWORD *)(i + 8) = *(_QWORD *)v10;`

我们在IDA开始乱了，用gdb看一看有没有线索，毕竟还没有查过那段`&node1`的奇妙地址。结果非常的意外：

{% qnimg CSAPP-Bomb-Lab-Writeup/phase6_node.png %}

不知为什么，每一个node元素，他的第三个数字，恰好跟下一个node的地址一模一样！

其实突破点就出来了：

**每一个node是一个struct类型！**

**node里面的第三个数字，代表着下一个元素的地址！**

**这就是链表的汇编！**

其他的数字是啥意思呢？第一个数字对应节点的值，第二个数字是id，第三个数字是地址，然后怎么有空出来的0？

不是空出来的0，而是因为地址就是64位的！

在这里，结构体内的元素顺序不同，所占用的空间也会不同，这个在CSAPP中有提到过内存对齐的概念！

那为什么上面的那个伏笔，对应的下标是1呢？

因为`v6`就是一个`QWORD`类型，而node里面的数字都是int，只有32位呀！

接下来就非常简单了，最后一个循环所代表的，就是确保最终的数值是降序排列的。

所以最终的排序是924 > 691 > 477 > 443 > 332 > 168，即`3 4 5 6 1 2`。

别忘记了前面有一个`x = 7 - x;`，所以最终的答案就是`4 3 2 1 6 5`。

## secret phase
~~待补充，今天晚点再做了补上。（咕咕咕）~~

Jan 15 upd：来补上secret phase了！

### 怎么进secret phase

`secret_phase`函数的入口其实在`phase_defused`里面。

懒得看汇编，直接用IDA Pro做了。~~其实反汇编出来的跟看汇编也差不多~~

{% qnimg CSAPP-Bomb-Lab-Writeup/phase_defused(IDA).png %}

这里看到一个`num_input_strings`，是个在bss段上的全局变量。同时，`sscanf`所读入的那个地址，也是在bss段上的，初始化都是0，不过可能会在函数执行的时候被修改。

那到底是什么时候被修改的？我们分别用gdb设断点看一看。

{% qnimg CSAPP-Bomb-Lab-Writeup/secret_phase(num_input_strings).png %}

可以发现这个变量的意思就是记录现在是第几关。所以当第六关的时候就可以了。

{% qnimg CSAPP-Bomb-Lab-Writeup/secret_phase(input_strings).png %}

可以发现是我们在打phase 4的时候，这个`input_strings + 240`所在的字符串就更改成了我们输入的内容。并且后面不会再更改。

所以我们只需要在第四阶段，在第三个位置上输入一个`DrEvil`，就可以在过完第六关之后触发了。

### 分析

{% qnimg CSAPP-Bomb-Lab-Writeup/secret_phase(IDA).png %}

要使这个`func7`返回2，并且输入的数字小于等于0x3e8 + 1，就可以通关了。

这里有一个`&n1`，点进去看看，又是在data段，跟前面的`&node1`很类似。并且，`n1`后面也紧跟着其他类似的东西，应该又是一个struct。

我们用gdb看一看：

{% qnimg CSAPP-Bomb-Lab-Writeup/secret_phase(n1).png %}

可以发现，每个结构体储存了两个地址，我们做下笔记：

```
    n1(n21, n22)  36
    n21(n31, n32) 8
    n22(n33, n34) 50
    n32(n43, n44) 22
    n33(n45, n46) 45
    n31(n41, n42) 6
    n34(n47, n48) 107
    n45 40
    n41 1
    n47 99
    n44 35
    n42 7
    n43 20
    n46 47
    n48 1001
```

这种一对二的关系，其实就是二叉树：

```
                n1
      n21             n22
  n31     n32     n33     n34
n41 n42 n43 n44 n45 n46 n47 n48
```

{% qnimg CSAPP-Bomb-Lab-Writeup/func7(IDA).png %}

想让`func7`为2，首先要落向左边，然后落向右边，然后返回0，这样就能构造出`2 * (2 * 0 + 1) = 2`了。

最后的返回0，也可以走左边再返回0，所以`n32`和`n43`的值都是没问题的，即我们有20跟22两个答案。


终于通关了！芜湖起飞！

{% qnimg CSAPP-Bomb-Lab-Writeup/success.png %}

{% qnimg CSAPP-Bomb-Lab-Writeup/success1.png %}

{% qnimg CSAPP-Bomb-Lab-Writeup/success2.png %}
