---
title: 推理钱处理阶段与embedding_layer
published: 2025-10-27
description: 将我们的输入进行前置处理，并且阐述了embedding_layer层的算子运算方式
tags: [AI Infra, C++, miniInfer]
category: miniInfer
draft: false
---

## 1.调用分词器获取token_id

```c++
auto tokens = model.encode(sentence);
int32_t prompt_len = tokens.size();
LOG_IF(FATAL, tokens.empty()) << "The tokens is empty.";
```

`model.encode` 在这里是调用了我们`encoder_layer_`层进行运算，而我们的`model` 在进行初始化的时候，是会根据分词器的类型去初始化不同的分词器。这里`token` 返回的是`std::vector<int32_t> `类型的函数。

## 2.模型的embedding

```C++
  auto input_tokens = get_buffer(ModelBufferType::kInputTokens);
  auto input_embeddings = get_buffer(ModelBufferType::kInputEmbeddings);
  if (input_tokens.size() != tokens.size()) 
  {
    input_tokens.reshape({static_cast<int32_t>(tokens.size())});
    input_embeddings.reshape({static_cast<int32_t>(tokens.size()), config_->dim_});
  }
  for (int32_t i = 0; i < tokens.size(); ++i) 
    input_tokens.index<int32_t>(i) = tokens.at(i);

```

在这里，根据我们的输入的尺寸，调整我们的分词器的大小比例，然后将我们的tokens拷贝进入模型的缓存tensor之中。

```c++
llama_layers_->embedding_layer_->forward(input_tokens, input_token_num, input_embeddings));
```

## 3.embedding算子

**CPU层**

```c++
void emb_kernel_normal(const tensor::Tensor& input, const tensor::Tensor& weight,const tensor::Tensor& output, int32_t vocab_size, void* stream) 
{
  CHECK(!input.is_empty());
  CHECK(!weight.is_empty());
  const int32_t input_num = static_cast<int32_t>(input.size());
  const int32_t weight_dim = weight.get_dim(1);
  CHECK(weight.device_type() == output.device_type());
  CHECK(input.device_type() == base::DeviceType::kDeviceCPU);
  const auto allocator = base::CPUDeviceAllocatorFactory::get_instance();
  for (int32_t i = 0; i < input_num; ++i) 
  {
    int32_t token = *input.ptr<int32_t>(i);
    if (token > vocab_size) 
      LOG(FATAL) << "Token index is greater than vocab size.";
    else 
    {
      float* dest_ptr = const_cast<float*>(output.ptr<float>(i * weight_dim));
      float* src_ptr = const_cast<float*>(weight.ptr<float>(token * weight_dim));
      if (weight.device_type() == base::DeviceType::kDeviceCPU) 
        allocator->memcpy(src_ptr, dest_ptr, weight_dim * sizeof(float),base::MemcpyKind::kMemcpyCPU2CPU);
      else 
        LOG(FATAL) << "Unknown device type of weight tensor in the embedding layer.";
    }
  }
}
```

在CPU的计算中，每一个token_id就对应了我们的`weight`权重矩阵的每一行，相应的，我们将他们提取出来就可以找到我们对应的token_id的embedding了，这里我们需要进行for循环进行提取。

**GPU层**

```C++
__global__ void emb_kernel_cu_fp32(int32_t vocab_size, int32_t token_num, int32_t weight_dim,
                                   const int32_t* input_ptr, const float* weight_ptr,
                                   float* output_ptr) {
  int32_t token_idx = blockIdx.x;
  if (token_idx >= token_num) {
    return;
  }
  int32_t token = input_ptr[token_idx];
  if (token >= vocab_size) {
    return;
  }

  float* output_ptr_start = output_ptr + token_idx * weight_dim;
  const float* weight_ptr_start = weight_ptr + token * weight_dim;

  for (int32_t i = threadIdx.x; i < weight_dim; i += blockDim.x) {
    output_ptr_start[i] = weight_ptr_start[i];
  }
}

void emb_kernel_cu(const tensor::Tensor& input, const tensor::Tensor& weight,const tensor::Tensor& output, int32_t vocab_size, void* stream) 
{
  tensor::Tensor input_cu;
  if (input.device_type() != base::DeviceType::kDeviceCUDA) 
  {
    input_cu = input.clone();
    input_cu.to_cuda();
  }
  const int32_t input_num = static_cast<int32_t>(input.size());
  const int32_t weight_dim = weight.get_dim(1);
  CHECK(weight.device_type() == output.device_type());
  CHECK(output.device_type() == base::DeviceType::kDeviceCUDA);

  //constexpr 常量计算指针，就是在编译的时候也进行求职
  constexpr int32_t max_seq_len = 512;
  constexpr int32_t thread_num = 128;
    
    
  int32_t* in_ptr = input_cu.ptr<int32_t>();
  float* wei_ptr = const_cast<float*>(weight.ptr<float>());
  float* out_ptr = const_cast<float*>(output.ptr<float>());
  if (stream) {
    cudaStream_t stream_ = static_cast<cudaStream_t>(stream);
    emb_kernel_cu_fp32<<<max_seq_len, thread_num, 0, stream_>>>(vocab_size, input_num, weight_dim,
                                                                in_ptr, wei_ptr, out_ptr);
  } else {
    emb_kernel_cu_fp32<<<max_seq_len, thread_num>>>(vocab_size, input_num, weight_dim, in_ptr,
                                                    wei_ptr, out_ptr);
  }
}
```

这里我们让一个block处理一个token_id的embedding向量，然后为了让内存访问尽可能达到连续性的要求，我们进行分块访问

```c++
for (int32_t i = threadIdx.x; i < weight_dim; i += blockDim.x) 
    output_ptr_start[i] = weight_ptr_start[i];
```

在对大量数据进行拷贝的时候多线程拷贝（通过流或多个线程进行异步拷贝）通常能够提供更好的性能，特别是当有多个流同时进行数据拷贝时，可以充分利用设备的带宽。

## 4. 结果返回与总结

```c++
op::EmbeddingOutput output(input_tokens, input_embeddings, input_token_num);
```

最后通过将结果保存到我们的`EmbeddingOutput` 之中，我们模型的推理处理阶段就基本完成了。所以embedding算子本质上是根据token_id去查询这个权重矩阵，然后将相应的值拷贝出来，只不过在GPU中，我们可以利用多线程拷贝的方法，对于大量的数据的情况下，更有利于利用GPU上的性能带宽。









































































