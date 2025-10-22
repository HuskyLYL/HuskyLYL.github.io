---
title: RMSNorm在miniInfer上的实现
published: 2025-10-21
description: 在miniInfer上编写RMSNorm的接口
tags: [AI Infra, C++, miniInfer,算子]
category: miniInfer
draft: false
cover: https://github.com/HuskyLYL/HuskyLYL.github.io/blob/main/src/contents/img/aaa.jpg
---

## 1.RMSNorm介绍

$$
RMSNorm(x)=\frac{x}{RMS(x)} ⊙g
$$

$$
RMS(x)=\sqrt{\frac{1}{d}\sum_{i=1}^{d}  x_{i}^{2}+𝜖}
$$

  RMSNorm是一种比LayerNorm更轻量化的归一方法，其中𝜖是一个比较小的数字，是为了防止除0

## 2.RMSNorm沿最后一维展开

```c++
const float eps = 1e-6f;
  const int32_t total_size = static_cast<int32_t>(input.size());
  const int32_t size = input.get_dim(input.dims_size() - 1);
  const int32_t dim_size = total_size / size;

  float* in_ptr = const_cast<float*>(input.ptr<float>());
  float* wei_ptr = const_cast<float*>(weight.ptr<float>());
  float* out_ptr = const_cast<float*>(output.ptr<float>());
  constexpr int threads_num = 128;
  if (stream) {
    cudaStream_t stream_ = static_cast<cudaStream_t>(stream);
    row_rmsnorm_f32_dim<<<dim_size, threads_num, 0, stream_>>>(in_ptr, wei_ptr, out_ptr, dim_size,
                                                               size, eps);
  } else {
    row_rmsnorm_f32_dim<<<dim_size, threads_num>>>(in_ptr, wei_ptr, out_ptr, dim_size, size, eps);
  }
```

​    我们可以把 RMSNorm 理解为：对输入张量的每一“行”或者“向量”（最后一维）做归一化。所以这里我们最好以一个维度的大小作为一个block的大小，这样可以方便我们后面进行归约话，因为在一个block之中，他是可以更好地进行线程的块内同步和share内存的访问（归约的基本）。

## 3.RMSNorm算子

### 3.1 利用float4进行x4展开

```c++
  //就是除不尽的情况
  const int pack_off = pack_size * pack_num;
  float sum = 0.0f;
  //float 还是可以有强制转化条件的
  float4* in_pack = reinterpret_cast<float4*>(block_in);

  for (int i = tid; i < pack_num; i += blockDim.x) {
    float4 in_float4 = *(in_pack + i);
    sum += in_float4.x * in_float4.x;
    sum += in_float4.y * in_float4.y;
    sum += in_float4.z * in_float4.z;
    sum += in_float4.w * in_float4.w;
  }
  for (int i = pack_off + tid; i < size; i += blockDim.x) 
    sum += block_in[i] * block_in[i];

```

  尽量的吧tid靠近的数据，访问的内存也应该在地址上具有连续性，这样能够减少内存数据的访问，这里注意一下pack_off的收尾。然后调用cub的块内归约操作

```c++
 using BlockReduce = cub::BlockReduce<float, 128>;
  __shared__ typename BlockReduce::TempStorage temp;
  __shared__ float shared_val;
  sum = BlockReduce(temp).Sum(sum);
  if (threadIdx.x == 0) {
    shared_val = sum;
  }
  __syncthreads();
  sum = shared_val;
  const float scale = rsqrtf(sum / static_cast<float>(size) + eps);

```

  除法操作常用scale因子去做惩罚，注意这里除了threadIdx == 0的sum之外，其他的sum都是未定义的，不可访问，快内同步是为了确保sum被拷贝进入共享变量之后才可读取。

  最后一pack组为单位，进行数据存储

```c++
  __syncthreads();
  sum = shared_val;
  const float scale = rsqrtf(sum / static_cast<float>(size) + eps);
  float4* wei_pack = reinterpret_cast<float4*>(wei);
  float4* out_pack = reinterpret_cast<float4*>(block_out);
  for (int i = tid; i < pack_num; i += blockDim.x) {
    float4 in_float4 = *(in_pack + i);
    float4 wei_float4 = *(wei_pack + i);
    *(out_pack + i) =
        make_float4(scale * in_float4.x * wei_float4.x, scale * in_float4.y * wei_float4.y,
                    scale * in_float4.z * wei_float4.z, scale * in_float4.w * wei_float4.w);
  }
  for (int i = pack_off + tid; i < size; i += blockDim.x) {
    block_out[i] = wei[i] * block_in[i] * scale;
  }
```



## 5.总结

- RMSNorm注意需要进行块内归约，并且所有线程需要利用归约结果进行计算（块内同步就很重要了）
- 注意处理不满足pack_size的剩余线程





