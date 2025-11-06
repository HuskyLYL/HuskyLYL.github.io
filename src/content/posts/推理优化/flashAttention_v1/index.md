---
title: flashAttention_v1
published: 2025-11-06
description: 总结了flashAttention_v1的技术要点与拓展了相关的概念
tags: [AI Infra, C++]
category: 推理优化
draft: true
image: ./cover.png
---

## 1.摘要

1. 通过IO感知，**tiling**技术减少GPU和片上SRAM之间的读写。
2. 扩展了块稀疏注意力
   - 将KV矩阵分块去关注，算法本身并不需要关注所有的KV矩阵



## 2.Introduction

![image-20251106165837704](../../../../../public/assets/images/image-20251106165837704.png)

简单来说就是从HBM加载数据到SRAM上计算，然后写会，emmmm，感觉啥好说的，我写核函数的时候都能想到这点。

### 2.1 Challenge：

- 增量的完成softmax的计算
- 保持softmax的归一化因子。
