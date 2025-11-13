---
title: flashInfer
published: 2025-11-05
description: 用于大语言模型推理服务的高效且可定制的注意力引擎
tags: [AI Infra, C++]
category: 推理优化
draft: false
image: ./cover.png
---

## 1.flashInfer

### 1.1KV缓存的异构性

- 不同batch的KV缓存长度不同，内存连续分配的效率低
- 存储格式多样化
- 访问频率不同
- 对不同软件和硬件的优化的限制

### 1.2 point

- 通过块稀疏格式和可组合格式处理KV缓存存储的异构性，优化内存访问，减少冗余
- 通过即时的编译（在运行的时候根据输入和环境编译出优化后的机器代码）适配各种场景

### 1.3 challenge

- （前缀重用，prefill和decode，查询长度的不同）容易出现负载不平衡的问题，调度算法需要让底层的注意力计算内核能够动态自适应。

### 1.4 design

- 块稀疏格式来应对KV缓存存储的异构性。
- 可编程结构->可定制的注意力模板。
- 兼容CUDAGraph
- 动态负载均衡调度框架，应对输入的动态变化。

![image-20251106155612097](../../../../../public/assets/images/image-20251106155612097.png)

 首先，从这张图中我们可以总结出，JIT运行时编译主要关注：

- 注意力变体格式
- 任务信息
- KV缓存的布局格式信息



## 2.BPT中状态缩放指令的引入

### 2.1注意力缩放因子：

$$
LSE(\mathcal{I})=\log \sum_{i \in \mathcal{I}} \exp \left(q \cdot k_{i}\right)
$$

- 这里的`i` 实际上对应了`K V` 矩阵的分块，注意这里是**点积**
- LSE(I) 就是 softmax 的“分母部分”在 block 内独立求和后的 log

### 2.2注意力的输出：

$$
O(\mathcal{I}) = \sum_{i \in \mathcal{I}} \frac{\exp(q \cdot k_{i})}{\exp(LSE(\mathcal{I}))} \cdot v_{i} \quad (2)
$$

- 注意，在我们的论文中,`log` 和我们的ln是等价的
- O[I]是标量，拥有完整的当前块的attention方向，但是scale不同



### 2.3将注意力输出和缩放因子打包成状态，便于在不同状态之间组合归纳

$$
[\begin{array}{c}O(\mathcal{I}) \quad LSE(\mathcal{I})\end{array}]
$$

$$
\begin{aligned} {\left[\begin{array}{c} O(\mathcal{I} \cup \mathcal{J}) \\ LSE(\mathcal{I} \cup \mathcal{J}) \end{array}\right] } & =\left[\begin{array}{c} O(\mathcal{I}) \\ L S E(\mathcal{I}) \end{array}\right] \oplus \left[\begin{array}{c} O(\mathcal{J}) \\ L S E(\mathcal{J}) \end{array}\right] \\ & =\left[\begin{array}{c} \frac{exp (L S E(\mathcal{I})) O(\mathcal{I})+exp (LSE(\mathcal{J})) O(\mathcal{J})}{exp (LSE(\mathcal{I}))+exp (LSE(\mathcal{J}))} \\ log (exp (L S E(\mathcal{I}))+exp (L S E(\mathcal{J}))) \end{array} \right] \end{aligned}
$$



### 2.4总结

- 其实和flashAttention中的计算思想是一致的
- 只不过flashAttention_v1更复杂在使用的是softmax的稳定版
- 思路和flashAttention没区别，分块求和重计算



## 3. 块压缩稀疏行（Block Compressed Sparse Row，BSR）

```c++
//大小等于行块数目+1，表示 第 i 行之前有多少个元素
row_ptr = [0, 2, 5, 7]
//记录每个非零索引对应的列ID
col_ind = [0, 3, 1, 2, 4, 1, 3]
//data
data
```





## 4.flashInfer

### 4.1pageAttention怎么去存储BSR

![image-20251113162927363](../../../../../public/assets/images/image-20251113162927363.png)

- 实则这里pageTable 等价于 BSR的列索引
- 这里稀疏矩阵的一行的大小就是整个物理矩阵的维度

### 4.2为什么flashInfer要做这种模式的等价？

- 统一KV Cache的存储结构，底层默认采用BSR的存储方式
- 这样kernei只需要按照BSR的方式去计算KV Cache即可

### 4.3flashInfer 对input和output的存储：ragged tensors

- 就是一个不规则的张量，没什么好说的，不需要填充。
- 本质上是因为query的请求长度不同导致的。还有output的`token by token` 生成的长度也不一致。
- 目的是让来自不同请求的query和output紧凑地打包在同一个张量之中。

### 4.4pageAttention特色：block越大（这里指的应该是query的长度），可共享的KVCache越多

![image-20251113165356206](../../../../../public/assets/images/image-20251113165356206.png)

- 越多的query，我们越能在KV Cache的前缀和中共享我们的内存
- 复用的话，我们直接指向同一块索引就可以了

### 4.5面向内存效率的可组合格式

- 这里我的理解就是不懂稀疏矩阵数据本身，而是修改block的大小，影响到每次加载多少数据区计算attention。
- 还是如上图，我们可以利用不同细粒度的KV组合计算，然后按照`BPT` 中的sotemax分块计算，最后结果组合即可得到最终的结果

**总结**：整个计算不是单个的blockSize，而是多个blockSIze的混合，最大化提高内存利用效率。







