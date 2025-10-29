---
title: RoPE旋转位置编码信息
published: 2025-10-29
description: 描述了llama2中的位置编码信息的计算过程
tags: [LLM]
category: LLM
draft: false
---

## 1.背景

  Transformer模型本身是不具备序列信息的，因为它的注意力机制是"无序"的。所以我们需要额外的注意信息，让模型知道第几个token在前面，第几个在后面。Lamam2使用的是旋转位置编码的信息，而不是传统的绝对位置编码信息，将位置信息以旋转的操作形式引入到Q和K向量之中。

## 2.旋转频率与相位展开的初始化

我们把`q`按照每两维一组，第`j` 组对应索引`2j` 和`2j+1` ，这里主要的目的是吧每个维度视为一个复数。
$$
\tilde{q}_j = q_{2j} + i\,q_{2j+1}
$$
所以我们的第`p` 个序列的第`j` 组数据使用的旋转角度为：
$$
\phi_{p,j} = p \cdot \text{inv\_freq}_j
$$
所以第`p` 哥序列的第`j` 组的数据`a` `b` 应该做出的数据运算是
$$
\text{令相位 } \phi_{p,j} = p \cdot \text{inv\_freq}_j.
$$

$$
\phi_{p,j} = p \cdot 10000^{-\frac{2j}{d}}
$$



旋转后得到
$$
\begin{aligned}
a' &= a \cos \phi_{p,j} - b \sin \phi_{p,j}, \\
b' &= a \sin \phi_{p,j} + b \cos \phi_{p,j}.
\end{aligned}
$$
对应的矩阵的形式为：
$$
\begin{bmatrix}
a' \\[4pt] b'
\end{bmatrix}
=
\begin{bmatrix}
\cos \phi_{p,j} & -\sin \phi_{p,j} \\[4pt]
\sin \phi_{p,j} & \cos \phi_{p,j}
\end{bmatrix}
\begin{bmatrix}
a \\[4pt] b
\end{bmatrix}.
$$

## 3. 与Transformer的计算过程对比

在Transformer中是直接+位置编码的`sin`和`cos` 的信息（虽然也是两两一组），而在我们这里，是将每个向量两两一组看成了虚数，去做虚数变换得到的。

## 4. RoPE位置编码向量的生成

### CPU版本

```C++
void sin_cos_cache_calc_cpu(int head_size, int max_seq_len, float* sin_cache, float* cos_cache) 
{
  for (int pos = 0; pos < max_seq_len; ++pos) 
  {
    for (int head_dim = 0; head_dim < head_size; ++head_dim) 
    {
      float freq =1.0f / std::pow(10000.0f, static_cast<float>(head_dim) / static_cast<float>(head_size));
      float val = static_cast<float>(pos) * freq;
      float fcr = cosf(val);
      float fci = sinf(val);
      *(sin_cache + pos * head_size + head_dim) = fci;
      *(cos_cache + pos * head_size + head_dim) = fcr;
    }
  }
}
```

**GPU版本**

```c++
__global__ void sin_cos_calc(int head_size, int max_seq_len, float* sin_cache, float* cos_cache) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  int head_dim = idx % head_size;
  for (int pos = 0; pos < max_seq_len; ++pos) {
    float freq = 1.0f / pow(10000.0f, static_cast<float>(head_dim) / static_cast<float>(head_size));
    float val = static_cast<float>(pos) * freq;
    float fcr = cosf(val);
    float fci = sinf(val);
    *(sin_cache + pos * head_size + head_dim) = fci;
    *(cos_cache + pos * head_size + head_dim) = fcr;
  }
}
```

因为计算量比较小，我们直接以`head_size` 作为我们的线程数目，然后每个线程都同时处理从0到pos为止，分别计算每个位置编码的`fci` 向量和`fcr` 向量。将他们存入我们的缓存之中。

## 5.RoPE算子

**CPU版本**

```c++
void rope_kernel_cpu(int32_t dim, int32_t kv_dim, int32_t head_size, const tensor::Tensor& input_q,
                     const tensor::Tensor& input_k, const tensor::Tensor& input_pos,
                     const tensor::Tensor& sin_cache, const tensor::Tensor& cos_cache,
                     void* stream) 
{
  UNUSED(stream);
  //pos值取决KVCache里面的缓存的seq_len的数目
  const int32_t pos = *input_pos.ptr<int32_t>(0);
  //注意，这里分组了
  for (int32_t i = 0; i < dim; i += 2) 
  {
    //这里看起来是把整个多头都传入了
    int32_t head_dim = i % head_size;
    float fci = *(sin_cache.ptr<float>() + pos * head_size + head_dim);
    float fcr = *(cos_cache.ptr<float>() + pos * head_size + head_dim);
    //因为这里的kv_dim恒小于我们的 q_dim
    int32_t rotn = i < kv_dim ? 2 : 1;  // how many vectors? 2 = q & k, 1 = q only
    for (int32_t v = 0; v < rotn; v++) 
    {
      //K 和 V矩阵交换运算(那么这里就说得通了)
      float* vec =const_cast<float*>(v == 0 ? input_q.ptr<float>(): input_k.ptr<float>());  // the vector to rotate (query or key)
      float v0 = vec[i];
      float v1 = vec[i + 1];
      vec[i] = v0 * fcr - v1 * fci;
      vec[i + 1] = v0 * fci + v1 * fcr;
    }
  }
}
```

这个算子同时对我们的K 和 Q向量进行处理，可能因为`GQA` 导致我们的K矩阵比我们的Q矩阵要小，所以这里也要分情况进行处理。

**GPU版本**

```C++
__device__ void rope_calc(float fcr, float fci, float* vec, int32_t idx) 
{
  float2* vec_ptr = reinterpret_cast<float2*>(vec + idx);
  float2 vec_value = *vec_ptr;
  *vec_ptr =
      make_float2(vec_value.x * fcr - vec_value.y * fci, vec_value.x * fci + vec_value.y * fcr);
}

__global__ void rope_kernel_cu_fp32(int pos, int dim, int kv_dim, int head_size,
                                    const float* input_q, const float* input_k,
                                    const float* sin_cache, const float* cos_cache) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  idx = idx * 2;
  if (idx >= dim) {
    return;
  }
  int head_dim = idx % head_size;
  float fci = *(sin_cache + pos * head_size + head_dim);
  float fcr = *(cos_cache + pos * head_size + head_dim);
  rope_calc(fcr, fci, const_cast<float*>(input_q), idx);
  if (idx >= kv_dim) {
    return;
  }
  rope_calc(fcr, fci, const_cast<float*>(input_k), idx);
}
```

我们当前的算子选择的是一个`pos` 一个`pos`的输入进行处理，一次核函数的调用则将所有分组展开，一个线程处理一个分组进行计算。tips:在C++的cuda里面，多使用float4 float2这样的数据类型，有利于节约内存访问，减少内存事务，提高吞吐量。

## 5. 总结：

- llama2中的位置编码向量都是两两处理，很适合利用cuda算子并行进行加速，并行度很高。
- 参考了复平面旋转的过程。
- 在模型初始化的时候，应该提前初始化好这些虚函数的信息（根据频率姿态去计算好我们的）
