---
title: matmul_kernel_for_miniInfer
published: 2025-10-31
description: 在miniInfer中实现matmul_kernel层
tags: [AI Infra, C++, miniInfer]
category: miniInfer
draft: false
---

## 1.导论

在我们的LLM的网络结构之中，无论是`Q` `K` `V` 矩阵的运算，还是FFN网络中的liner线性层，其本质的核心都离不开矩阵乘法的操作，而如何编写出高效的矩阵算子，对于加速我们整个神经网络的推理都有重要的意义。

## 2.matmul_kernel_for_CPU

```c++
void matmul_kernel_cpu(const tensor::Tensor& input, const tensor::Tensor& weight,
                       const tensor::Tensor& output, float scale,
                       const CudaConfig* config) 
{
  UNUSED(config);
  CHECK(input.is_empty() == false);
  CHECK(weight.is_empty() == false);
  CHECK(output.is_empty() == false);
  CHECK(input.device_type() == base::DeviceType::kDeviceCPU);
  CHECK(weight.device_type() == base::DeviceType::kDeviceCPU);
  CHECK(output.device_type() == base::DeviceType::kDeviceCPU);

  const float* input_ptr = input.ptr<float>();
  const float* weight_ptr = weight.ptr<float>();
  const float* output_ptr = output.ptr<float>();

  int32_t in_dim1 = 1;
  int32_t in_dim0 = 1;
  if (input.dims_size() == 2) {
    in_dim0 = input.get_dim(0);
    in_dim1 = input.get_dim(1);
  } else if (input.dims_size() == 1) {
    in_dim0 = input.get_dim(0);
  } else {
    LOG(FATAL) << "The input tensor has a wrong dim size.";
  }

  CHECK_EQ(weight.dims_size(), 2);
  const int32_t wei_dim0 = weight.get_dim(0);
  const int32_t wei_dim1 = weight.get_dim(1);
  CHECK_EQ(in_dim0, wei_dim1);

  CHECK_EQ(output.size(), wei_dim0 * in_dim1);
  arma::fmat input_mat(const_cast<float*>(input_ptr), in_dim1, in_dim0, false, true);
  arma::fmat weight_mat(const_cast<float*>(weight_ptr), wei_dim1, wei_dim0, false, true);
  arma::fmat output_mat(const_cast<float*>(output_ptr), in_dim1, wei_dim0, false, true);
  output_mat = ((input_mat * weight_mat)) * scale;
}
```

这里我们依然用`Armadillo` 线性代数库去实现执行我们的矩阵乘法操作。
$$
output=(input×weight)×scale
$$
所以在CPU上，我们都是直接调用`Armadillo` 线性库去直接实现的计算



## 3.matmul_kernel_for_GPU

```c++
template <int THREAD_PER_BLOCK, int ROW_PER_BLOCK>
__global__ void matmul_kernel_cu_fp32(const float* input, const float* weight, float* output, int M,int K) {
  __shared__ float sdata[THREAD_PER_BLOCK];
  unsigned int tid = threadIdx.x;

  int start_row = blockIdx.x * ROW_PER_BLOCK;
  int end_row = start_row + ROW_PER_BLOCK;
  if (start_row >= K) {
    return;
  }

  constexpr int pack_size = 4;
    
  //一行有多少个pack
  const int pack_num = M / pack_size;
  const int pack_off = pack_size * pack_num;

#pragma unroll
  for (int p = start_row; p < end_row; ++p) {
    sdata[tid] = 0;
    int row_offset = p * M;
    float4* input_float4_ptr = (float4*)input;
    float4* weight_float4_ptr = (float4*)(weight + row_offset);

#pragma unroll
    for (int i = tid; i < pack_num; i += blockDim.x) {
      float4 input_float4 = *(input_float4_ptr + i);
      float4 weight_float4 = *(weight_float4_ptr + i);
      float part_sum = input_float4.x * weight_float4.x + input_float4.y * weight_float4.y +
                       input_float4.z * weight_float4.z + input_float4.w * weight_float4.w;
      sdata[tid] += part_sum;
    }

    for (int i = pack_off + tid; i < M; i += blockDim.x) {
      sdata[tid] += input[i] * weight[row_offset + i];
    }

    __syncthreads();

    using BlockReduce = cub::BlockReduce<float, THREAD_PER_BLOCK>;
    __shared__ typename BlockReduce::TempStorage temp;
    float part_sum = BlockReduce(temp).Sum(sdata[tid]);
    __syncthreads();

    if (tid == 0) {
      output[p] = part_sum;
    }
    __syncthreads();
  }
}
```

首先我们用模板类来传参，最关键的是`ROW_PER_BLOCK` 每个`block`负责多少行，特备关注，这里我们在存储矩阵的时候，按照了按列连续存储的方式，这样我们在GPU中就可以合并内存进行访问。

同时，这里还保持了`float4`的计算，不过在调用的时候，我们设置为一行是一个block，对应一个`output`地址的数据。最后要记得加上不能被pack整除部分的数据。

关键核心：（权重矩阵转置保存）

## 4. 总结

在实现`matmul_kernel` 的时候，我们在导入权重的时候，通常都是导入转置矩阵以加快内存的访问。所谓Liner层，本质上就是一个一维的向量，乘以一个矩阵，转换成想要的维度。
