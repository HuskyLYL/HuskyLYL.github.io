---
title: 推理过程中的KVcache
published: 2025-10-27
description: 为什么会有KVcache这个东西
tags: [AI Infra, C++, miniInfer]
category: 推理优化
draft: false
cover: https://github.com/HuskyLYL/HuskyLYL.github.io/blob/main/src/contents/img/aaa.jpg
---

## 1. KVcache的由来

  首先，这是由模型的性质来决定的，我们的LLama2在推理的时候是一个token一个token地产生的，这正是自回归语言模型的核心特征，下一次的输入的时候，需要给定前面的所有的token来预测当前的token值。

## 2.llama2中KVchache缓存的体现

```python
xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
self.cache_k[:bsz, start_pos : start_pos + seqlen] = xk
self.cache_v[:bsz, start_pos : start_pos + seqlen] = xv
```

  其实就是将输入的X，经过`wk` `wv` 运算的结果保存到我们的cache之中，后续就不用关注历史的token计算了，只用关心当前输入的token(也就是我们模型生成的或者输入的值)。

## 3.后续的计算

```python
keys = self.cache_k[:bsz, : start_pos + seqlen]    (bs, n_local_heads, cache_len + seqlen, head_dim)
values = self.cache_v[:bsz, : start_pos + seqlen]  (bs, n_local_heads, cache_len + seqlen, head_dim)
scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim) (bs,n_local_heads,seq_len,head_dim)
```

  通过scores计算得到`(bs, n_local_heads, seqlen, cache_len + seqlen)` shape的矩阵，表示**第 i 个 query token 对第 j 个 key 的关注强度**。最后查询我们的`value`矩阵`( cache_len + seqlen, head_dim)` 得到我们的`head_dim` 类型的变量。（seqlen,head_dim）

  最后因为拆分多头，可能会出现维度没对齐和除尽的情况，我么还需要添加一个线性层变换回去、

```python
scores = F.softmax(scores.float(), dim=-1).type_as(xq)
output = torch.matmul(scores, values)  # (bs, n_local_heads, seqlen, head_dim)
output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
return self.wo(output)
```

## 4.总结：

  KVCache存在的目的是为了记录历史的token信息(key,value)值，更方便的获取我们当前seq token的上下文关联，防止重复计算`token_embedding @ wk` 和`token_embedding @ wv`这两个矩阵，因此，在我们将当前的信息添加进入k V矩阵之后，依然需要进行query，计算value操作，这个过程是无法避免的。但是帮助我们防止推理过程中，历史token的重复计算，加快推理速度。
