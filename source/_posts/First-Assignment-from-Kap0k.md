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

### Inspiration

在搜索如何从汇编到shellcode的过程中，看到了一个教怎么弄出shell的教程，它的汇编是这样的：

```
xor    %eax,%eax
push   %eax
push   $0x68732f2f
push   $0x6e69622f
mov    %esp,%ebx
push   %eax
push   %ebx
mov    %esp,%ecx
mov    $0xb,%al
int    $0x80
```

仔细研究它的写法，我们下面的解决方案就来自这段汇编的细节。（其实改编下就能用了）


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

{% qnimg First-Assignment-from-Kap0k/objdump.png %}

最后通过一个在网上找到的命令，直接提取出了机器码，生成了shellcode，省去了一个字节一个字节手抄出来的麻烦：

```
$ objdump -d ./shellcode|grep '[0-9a-f]:'|grep -v 'file'|cut -f2 -d:|cut -f1-6 -d' '|tr -s ' '|tr '\t' ' '|sed 's/ $//g'|sed 's/ /\\x/g'|paste -d '' -s |sed 's/^/"/'|sed 's/$/"/g'
```

## 汇编快排

直接用汇编写出快排我做不到，就先写个c出来吧。

```c++
#include <stdio.h>
#include <unistd.h>
int a[] = {5, 4, 3, 2, 1, 1, 4, 5, 1, 4};

void swap(int *a, int *b) {
    int t = *a;
    *a = *b;
    *b = t;
}
void qsort(int *start, int *end) {
    int len = (end - start);
    int pivot = *(start + (len >> 1));
    int *i = start, *j = end;
    while(i <= j) {
        while(*i < pivot) i++;
        while(*j > pivot) j--;
        if(i <= j) swap(i++, j--);
    }
    if(i < end) qsort(i, end);
    if(start < j) qsort(start, j);
}
int main() {
    qsort(a, a + 10);
    for(int i = 0; i < 10; i++) printf("%d ", a[i]);
    printf("\n");
    return 0;
}
```
后来发现汇编里面要写指针的话就好麻烦，干脆重新改一改：

```c++
void qsort(int *a, int l, int r) {
    int mid = (l + r) >> 1;
    int pivot = a[mid];
    int i = l, j = r;
    while(i <= j) {
        while(a[i] < pivot) i++;
        while(a[j] > pivot) j--;
        if(i <= j) swap(a, i++, j--);
    }
    if(i < r) qsort(a, i, r);
    if(l < j) qsort(a, l, j);
    return;
}
```

看了师傅的代码，发现可以用r8到r11的这4个寄存器来存，顿时方便了很多。~~本来还以为要一直存在栈上~~

```
global _start

section .data
    a: dd 1, 1, 4, 5, 1, 4, 2, 0, 7, 7
section .text
_start:
    mov rdi, a
    xor rsi, rsi
    mov rdx, 10
    call qsort
    mov rax, 60
    xor rdi, rdi
    syscall

swap:
    ; rdi: a, rsi: i, rdx: j
    mov ebx, QWORD [rdi + 4 * rsi]
    mov ecx, QWORD [rdi + 4 * rdx]
    mov QWORD [rdi + 4 * rsi], ecx
    mov QWORD [rdi + 4 * rdx], ebx

qsort:
    ; rdi: a, rsi: start, rdx: end
    mov r8, rsi ; start
    mov r9, rdx ; end
    mov r10, r8 ; i
    mov r11, r9 ; j
    mov rbx, r9
    add rbx, r8
    sar rbx
    mov ebx, DWORD [r8 + 4 * rbx]
    loop:
        cmp r10, r11
        jg after_loop1
        i_loop:
            mov eax, DWORD [r8 + 4 * r10]
            cmp eax, ebx
            jge j_loop
            inc r10
            jmp i_loop
        j_loop:
            mov eax, DWORD [r8 + 4 * r11]
            cmp eax, ebx
            jle swap_i_j
            dec r11
            jmp j_loop
        swap_i_j:
            cmp r10, r11
            jg loop
            mov rdi, a
            mov rsi, r10
            mov rdx, r11
            call swap
            inc r8
            dec r9
            jmp loop
    after_loop1:
        cmp r10 r9
        jge after_loop2
        mov rdi, a
        mov rsi, r10
        mov rdx, r9
        push r8
        push r9
        push r10
        push r11
        call qsort
        pop r11
        pop r10
        pop r9
        pop r8

    after_loop2:
        cmp r8 r11
        jge return
        mov rdi, a
        mov rsi, r8
        mov rdx, r11
        push r8
        push r9
        push r10
        push r11
        call qsort
        pop r11
        pop r10
        pop r9
        pop r8
    return:
        ret

```

没编译过，不过觉得问题不大。但愿如此（x

Jan 17 upd：重新用熟悉的AT&T语法自己手写了一遍汇编快排，这次用了指针，看上去比较清晰：

```
.globl _start
.section .data
    array:
        .int 1, 1, 4, 5, 1, 4, 2, 0, 7, 7

.section .text
qsort:
    # rdi: int* start, rsi: int* end
    pushq %rbp
    movq %rsp, %rbp
    movq %rsi, %rax
    subq %rdi, %rax
    sarq %rax
    addq %rdi, %rax
    movq %rdi, %r8 # start(backup)
    movq %rsi, %r9 # end(backup)
    movq %rdi, %rbx # i
    movq %rsi, %rcx # j
    jmp _init_loop
 
_init_loop:
    cmpq %rcx, %rbx
    jg _recursive1
    jmp _i_loop

_i_loop:
    cmpq (%rax), (%rbx)
    jge _j_loop
    incq %rbx
    jmp _i_loop

_j_loop:
    cmpq (%rax), (%rcx)
    jle _swap
    decq %rcx
    jmp _j_loop

_swap:
    cmpq %rcx, %rbx
    jg _init_loop
    movq (%rbx), r10
    movq (%rcx), r11
    movq r10, (%rcx)
    movq r11, (%rbx)
    incq %rbx
    decq %rcx

_recursive1:
    cmpq %r9, %rbx
    jge _recursive2
    movq %rbx, %rdi
    movq %r9, %rsi
    call _qsort
    jmp _recursive2

_recursive2:
    cmpq %rcx, %r8
    jge _after_loop
    movq %r8, %rdi
    movq %rcx, %rsi
    call _qsort
    jmp _after_loop

_after_loop:
    movq %rbp, %rsp
    popq %rbp
    retq

_start:
    movq array, %rdi
    leaq (array, 10, 4), %rsi
    call _qsort
    movl $0, %edi
    movl $60, %eax
    syscall

```

## Reference

https://blog.csdn.net/flyoutsan/article/details/62237779

https://www.cnblogs.com/orlion/p/5765339.html

还有CSAPP的Chapter 3。不愧是CSAPP。