---
title: flashAttention_v3
published: 2025-11-10
description: 总结了flashAttention_v3的技术要点与拓展了相关的概念
tags: [AI Infra, C++]
category: 推理优化
draft: true
image: ./cover.png
---

## 1.背景-硬件新特性

### 1.1异步Tensor Cores

Tensor Cores 是GPU专门用于加速矩阵乘法（GEMM）的硬件单元，Hopper架构将其升级为**异步执行模式**：

- 前代的Tensor Cores执行矩阵乘法的时候，会"阻塞"后续指令，需要等待当前运算完成才能继续
- Hopper 的 Tensor Cores 支持 **WGMMA（Warpgroup Matrix Multiply-Accumulate）指令**，运算发起后无需等待结果返回，GPU可以同时调度其他任务。

### 1.2张量内存加速器

- 前代GPU需移开线程手动加载内存，过程线程会因为等待数据而闲置。
- 但是TMA可独立计算线程，异步完成““全局内存->共享内存”的数据块加载 / 存储，并且支持灵活的分块布局。

### 1.3低精度FP8硬件加速



## 2.背景-lashAttention2的表现

- FlashAttention-2 在新一代的GPU上利用率仍比较低
- 未采取Hopper架构的专属指令

### 2.1动机：在flashAttention_2 上明确异步执行和低精度特性

### 2.2创新改进：

1. **Producer-Consumer（生产者 - 消费者）**
   - 生产者负责将数据从HBM搬运到SMEM
   - 消费者负责进行矩阵运算
2. 在异步分块矩阵乘法下隐藏Softmax运算
   - 将softmax中吞吐量相对较低非 GEMM的操作与用于GEMM异步的 WGMMA指令进行并行执行，同时进行调度，规避Softmax 与 GEMM之间特定顺序的依赖关系
3. 硬件加速的低精度矩阵乘法



## 3.补充回顾-GPU的缓存结构

- 全局内存(HBM)：可以被所有的SM访问，属于片外存储，全局内存中的数据会被透明的缓存到L2。
- 二级缓存(L2)：位于芯片内部，作为全局内存与高速缓存之间的中间层。
- 共享内存：每一个SM特留的存储，表现为一个block内部共享。
- 寄存器：一个warp内部的寄存机可以互换数据。

### 3.1线程束组：线程束的“协作单元”负责适配异步张量矩阵乘法运算

Hopper架构的Tensor Core需要专门的`WGMMA`指令调用，而该指令集专门针对WarpGroup，即4个连续的warp，当一个Warpgroup通过`WGMMA` 发起矩阵乘法的时候，无需等待计算完成，就可以释放资源让该`Warpgroup` 执行其余的操作。 

### 3.2Hopper架构新特性

- Hopper架构支持通过`setmaxnreg` 指令在线程束组之间动态重新分配寄存器，因此执行MMA的线程束比仅能发起张量内存的加速器操作的线程束获得更多的寄存器内存
- 发起张量内存加速器仅仅只需要单个线程
- 同时WGMMA对低精度的矩阵乘法运算有更好的支持度



## 4.flashAttenthon_v3

falshAttention_v3 在 flashAttention_v2基础上的改进：

- 将线程专业化与循环共享内存缓冲区整合到了FlashAttention-2之中。
- 利用WGMMA的异步特性，设计出GEMM和sotemax重叠执行的两阶段流水线。
- FP8 精度下算法的修改与内存兼容。

### 4.1 基于线程束专业化与乒乓调度的生产者 - 消费者异步机制

### 结合循环共享内存缓冲区（circular SMEM buffer）的线程束专业化方案:(without intra-consumer overlapping – CTA view)

1. 初始化流水线对象，该对象通过s阶段循环共享内存缓冲区管理屏障同步（一般是生产者和消费者一一对应）
2. 生产者线程组
   - 释放预设数量的寄存器
   - 将Qi矩阵拷贝到共享内存之中
   - 加载完成之后，告知消费者矩阵Qi已经加载
   - 循环加载矩阵 K V，加载到共享内存缓冲区（j%s）
     - 加载完成之后告知消费者已经被加载
3. 消费者线程：
   - 过程同我们的flashAttention_v2总结
   - 但是每次访问我们的存储单元的时候都需要commit一下流水线对象。
   - 同时访问结束后即使的释放缓冲区的存储空间
   - 最后，执行矩阵的惩罚的时候，调用我们的WGMMA指令集进行计算

**总结**：大体上生产者线程和消费者线程一对一的关系（线程数量需要进行二分配），生产者线程基本上和flashAttention_v2的计算过程差不多，只不过每一次访问的时候都需要提交到流水线对象查看缓存是否加载完成

### 4.2乒乓调度

![image-20251110211956069](../../../../../public/assets/images/image-20251110211956069.png)

- 相同的颜色代表了相同的迭代轮次
- 为什么softmax可GEMM可以重叠（**低吞吐量硬件瓶颈型操作**和**高吞吐量硬件友好型操作** ）
- 所以我们在一个warp内可以将其放在一起

