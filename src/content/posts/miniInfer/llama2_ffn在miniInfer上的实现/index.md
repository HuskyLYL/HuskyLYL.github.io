---
title: llama2_ffn在miniInfer上的实现
published: 2025-10-31
description: llama2_ffn层在miniInfer推理框架上的具体实现
tags: [AI Infra, C++, miniInfer]
category: miniInfer
draft: false
---

## 1.llama2中ffn的计算流程

```python

class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = ColumnParallelLinear(
            dim, hidden_dim, bias=False, gather_output=False, init_method=lambda x: x
        )
        self.w2 = RowParallelLinear(
            hidden_dim, dim, bias=False, input_is_parallel=True, init_method=lambda x: x
        )
        self.w3 = ColumnParallelLinear(
            dim, hidden_dim, bias=False, gather_output=False, init_method=lambda x: x
        )

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

```

整体的计算流程如下图：

![image-20251101112401113](image-20251101112401113.png)

## 2.swiglu_kernel算子的实现

siLU的公式为
$$
\mathrm{SiLU}(x) = x \cdot \sigma(x)
$$

$$
\sigma(x) = \frac{1}{1 + e^{-x}}
$$

SwiGLU 是在 FFN（前馈网络）中使用的“门控激活结构”，它由两条线性变换分支构成，其中一条经过 `SiLU` 激活，然后与另一条逐元素相乘。也就是我们在上图中的向量乘法的那个操作的部分。

**GPU版本**

```c++
__global__ void swiglu_kernel_cu_fp32(int size, const float* in1, const float* in2, float* out) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx >= size) return;

  float x1 = in1[idx];
  float x2 = in2[idx];

  float value = 1.0f / (1.0f + expf(-x1));
  float silu = x1 * value;
  out[idx] = silu * x2;
}
```

这里的并行是一个线程处理一个位置的数据。

**CPU版本**

```c++
void swiglu_kernel_cpu(const tensor::Tensor& input1, const tensor::Tensor& input2,
                       const tensor::Tensor& output, void* stream) {
  UNUSED(stream);
  CHECK_EQ(input1.is_empty(),false);
  CHECK_EQ(input2.is_empty(),false);
  CHECK_EQ(output.is_empty(),false);

  CHECK(input1.device_type() == base::DeviceType::kDeviceCPU);
  CHECK(input2.device_type() == base::DeviceType::kDeviceCPU);
  CHECK(output.device_type() == base::DeviceType::kDeviceCPU);

  arma::fvec input1_vec(const_cast<float*>(input1.ptr<float>()), input1.size(), false,
                        true);
  arma::fvec input2_vec(const_cast<float*>(input2.ptr<float>()), input2.size(), false,
                        true);
  arma::fvec output_vec(const_cast<float*>(output.ptr<float>()), output.size(), false,
                        true);

  input1_vec %= (1.0f / (1.0f + arma::exp(-input1_vec)));
  output_vec = input1_vec % input2_vec;
}
```

这一段代码实际上是在**利用 Armadillo（C++ 的线性代数库）**，把原始的 `float*` 数组或 OpenCV / Tensor 数据包装成 `arma::fvec`（单精度浮点向量）对象，从而方便地使用 Armadillo 提供的矩阵和向量运算接口。

**Armadillo**是一个高层次的C++线性代数哭，目标是提供MATLAB一样的语法风格的数据，是一个智能封装层，在背后调用高度优化的数学库。

## 3.feed_forward

```c++
void LLama2Model::feed_forward(int32_t layer_idx, const tensor::Tensor& input) const {
  CHECK(llama_layers_ != nullptr);
  // residual add
  CHECK_NE(llama_layers_->add_layer_, nullptr)
      << "The add layer in the feedforward block is null pointer";
  STATUS_CHECK(
      llama_layers_->add_layer_->forward(input, get_buffer(ModelBufferType::kAttnOutput), input));

  // ffn rmsnorm
  tensor::Tensor ffn_norm_output = get_buffer(ModelBufferType::kFFNRMSNorm);
  const auto& ffn_rmsnorm = llama_layers_->rmsnorm_layers_.at(layer_idx + config_->layer_num_);
  CHECK_NE(ffn_rmsnorm, nullptr)
      << "The final rmsnorm layer in the feedforward block is null pointer";
  STATUS_CHECK(ffn_rmsnorm->forward(input, ffn_norm_output));

  // w1
  tensor::Tensor w1_output = get_buffer(ModelBufferType::kW1Output);
  const auto& w1_layer = llama_layers_->w1_layers_.at(layer_idx);
  CHECK_NE(w1_layer, nullptr) << "The w1 layer in the feedforward block is null pointer";
  STATUS_CHECK(w1_layer->forward(ffn_norm_output, w1_output));

  // w3
  tensor::Tensor w3_ouput = get_buffer(ModelBufferType::kW3Output);
  const auto& w3_layer = llama_layers_->w3_layers_.at(layer_idx);
  CHECK_NE(w3_layer, nullptr) << "The w3 layer in the feedforward block is null pointer";
  STATUS_CHECK(w3_layer->forward(ffn_norm_output, w3_ouput));

  // SwiGLU
  CHECK_NE(llama_layers_->swiglu_layer_, nullptr)
      << "The swiglu layer in the feedforward block is null pointer";
  STATUS_CHECK(llama_layers_->swiglu_layer_->forward(w1_output, w3_ouput, w1_output));

  // w2
  tensor::Tensor w2_output = get_buffer(ModelBufferType::kW2Output);
  const auto& w2_layer = llama_layers_->w2_layers_.at(layer_idx);
  CHECK_NE(w2_layer, nullptr) << "The w2 layer in the feedforward block is null pointer";
  STATUS_CHECK(w2_layer->forward(w1_output, w2_output));

  // residual add
  CHECK_NE(llama_layers_->add_layer_, nullptr)
      << "The add layer in the feedforward block is null pointer";
  STATUS_CHECK(llama_layers_->add_layer_->forward(input, w2_output, input));
}
```

注意开头的要接收来自`attention_block`的残差连接，最后输出的时候也要调用一次`residual add` 进行一次残差连接，这就是为什么要在我们模型中设置缓存，目的是为了方便进行**残差连接**。

## 4.总结：

- llama2中的ffn注意残差链接要经常用模型中的缓存，保留输入结果
- 其他的按照llama2模型的计算流程进行计算，处理起来不会有太大的问题。

