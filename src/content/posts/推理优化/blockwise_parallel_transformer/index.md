---
title: blockwise_parallel_transformer
published: 2025-11-10
description: 在整个decoder层面，实现attention模块和FFN网络模块的并行计算
tags: [AI Infra, C++]
category: 推理优化
draft: false
image: ./cover.png
---

## 1.出发动机：

当自注意力通过分块的方式降低内存的需求的时候（flashAttention）,将前馈网络与之融合便具备了可行性。这种融合方式无需等待自注意力计算完全结束，在对整个序列执行前馈网络步骤，从而打破了计算流程的时序限制。

## 2.核心思路

$$
Output_{i} = FFN\left(Attention\left(Q_{i}, K, V\right) + Q_{i}\right) + Attention\left(Q_{i}, K, V\right) + Q_{i}
$$

- 这里我感觉是在`flashAttention` 基础上的一个优化
- 共用缓存Attention的缓存（模型中间的缓存）
- 对prefile 和 seq 长度不为1的有显著的优化,但是对`token by token` 的情况不太友好



### 2.1 尴尬的点：

- 相当于是attention 和 FFN 的核函数同步进行了（所以由于硬件的性能不够，这里很可能变成串行执行）