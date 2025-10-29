---
title: llama2-7b载入model
published: 2025-10-27
description: 从llama的结构出发，将model载入miniInfer
tags: [AI Infra, C++, miniInfer]
category: miniInfer
draft: false
image: ./image-20251027102311174.png
---

## 1.模型属性的初始化

```c++
LLama2Model::LLama2Model(base::TokenizerType tokenizer_type, std::string token_path,std::string model_path, bool is_quant_model): Model(tokenizer_type,base::ModelType::kModelTypeLLama2,std::move(token_path),std::move(model_path), is_quant_model) 
{
    //pass
}
Model::Model(base::TokenizerType tokenizer_type, base::ModelType model_type, std::string token_path,
             std::string model_path, bool is_quant_model)
    : tokenizer_type_(tokenizer_type),
      model_type_(model_type),
      token_path_(std::move(token_path)),
      model_path_(std::move(model_path)),
      is_quant_model_(is_quant_model) {}
```

开始，我们现将模型的一些参数进行初始化:

- 模型类型
- 分词器类型
- 模型路径
- 分词器路径
- 模型是否量化

主要是为我们后面模型`init`做准备，方便我们进行参数权重的读取操作

## 2.初始化模型的embedding层

我们要进行初始化的是`Model` 类下面的`std::unique_ptr<op::EncodeLayerBase> encode_layer_;`

```C++
encode_layer_ = std::make_unique<op::SpeEncodeLayer>(this->token_path_, true, false);
SpeEncodeLayer::SpeEncodeLayer(std::string token_model_path, bool has_bos, bool has_eos)
    : EncodeLayerBase(std::move(token_model_path), has_bos, has_eos) 
{
  using namespace sentencepiece::util;
  spe = std::make_unique<sentencepiece::SentencePieceProcessor>();
  auto rc = spe->Load(token_model_path_);
  if (rc.code() != StatusCode::kOk) {
    LOG(FATAL)
        << "The token model path is not valid, please check the path and type of token model.";
  }
}

```

最终目的是初始化`SpeEncodeLayer` 中的`sentencepiece` 成员，然后通过C++多模态的纯虚函数来间接的调用子类中的`private` 从而实现编码的过程。所以这一层的巧妙之处就是用了C++多模态的技术，未来可以自由选择分词器embedding的类型来完成拓展。这里干的核心操作就是初始化了一个编码器。所以在我们的`model` 中，`encoder_layer_` 构造完成。

## 3.读取模型的配置，将模型数据映射到内存空间

在我们写文件的时候，把模型的配置以把 7 个整数（模型结构参数）按照 C 语言中的二进制格式打包成连续的字节流，作为模型文件头部（header）

```python
header = struct.pack('iiiiiii', p.dim, hidden_dim, p.n_layers, p.n_heads,n_kv_heads, p.vocab_size, p.max_seq_len)
```

`dim` 也就是我们的模型的`attention` 的输入和输出的维度，`hidden_dim` 也就是我们的前馈层的隐藏维度，`layer` 的数目也就是我们的模型中的`transformer` 的个数`n_head` 和`n_kv_head` 主要是用在我们的GQA分组注意力查询的值。`max_seq_len` 就是我们的注意力能够覆盖的最大的token的数目。

``` c++
 if (is_quant_model_) {
    if (fread(&group_size_, sizeof(int32_t), 1, file) != 1) {
      return error::ModelParseError(
          "Failed to retrieve the group size information from the model "
          "file.");
    }
  }
```

在这里，我们要判断我们的模型是否被量化处理过。因为我们在存储的时候，是分组量化，然后存入缩放因子，如下图：

```python
for i, w in enumerate(weights):
    # quantize this weight
    q, s, err = quantize_q80(w, group_size)
    # save the int8 weights to file
    serialize_int8(out_file, q)  # save the tensor in int8
    serialize_fp32(out_file, s)  # save scale factors
    # logging
    ew.append((err, w.shape))
    print(f"{i + 1}/{len(weights)} quantized {tuple(w.shape)} to Q8_0 with max error {err}")
```

然后将我们的model文件的原始数据利用`mmap` 读取到我们的内存地址空间之中，然后将我们的weight_data的指针移动到我们的权重之中，便于下一步的行为处理

```c++
if (!is_quant_model_) 
{
  raw_model_data_->weight_data =
  static_cast<int8_t*>(raw_model_data_->data) + sizeof(ModelConfig);
} 
else 
{
  raw_model_data_->weight_data =
  static_cast<int8_t*>(raw_model_data_->data) + sizeof(ModelConfig) + sizeof(group_size_);
}
```

## 4.访问映射到内存的模型文件，初始化网络的权重层

首先我们先来设计llama2的Layer类

```c++
struct LLama2Layers 
{
  std::shared_ptr<op::Layer> add_layer_;
  std::shared_ptr<op::Layer> rope_layer_;
  std::shared_ptr<op::Layer> swiglu_layer_;
  std::shared_ptr<op::Layer> mha_layer_;

  std::vector<std::shared_ptr<op::Layer>> wq_layers_;
  std::vector<std::shared_ptr<op::Layer>> wk_layers_;
  std::vector<std::shared_ptr<op::Layer>> wv_layers_;
  std::vector<std::shared_ptr<op::Layer>> wo_layers_;

  std::vector<std::shared_ptr<op::Layer>> w1_layers_;
  std::vector<std::shared_ptr<op::Layer>> w2_layers_;
  std::vector<std::shared_ptr<op::Layer>> rmsnorm_layers_;
  std::vector<std::shared_ptr<op::Layer>> w3_layers_;
  
  std::shared_ptr<op::Layer> cls_layer_;
  std::shared_ptr<op::Layer> embedding_layer_;

  void to_cuda(std::shared_ptr<kernel::CudaConfig> config);
};
```

因为我们的llama2是由32个`encoder` 所构成的，所以我们将每一个`Transformer` 的block用我们的vector去存储参数，下面是我们模型每一层的参数举例。

```bash
layers.31.attention.wq.weight: (4096, 4096)
layers.31.attention.wk.weight: (4096, 4096)
layers.31.attention.wv.weight: (4096, 4096)
layers.31.attention.wo.weight: (4096, 4096)
layers.31.feed_forward.w1.weight: (11008, 4096)
layers.31.feed_forward.w2.weight: (4096, 11008)
layers.31.feed_forward.w3.weight: (11008, 4096)
layers.31.attention_norm.weight: (4096,)
layers.31.ffn_norm.weight: (4096,)
```

除了我们的Normal层，其他的层都是线性层，所以这里用我们的矩阵乘法即可很好的模拟出torch中的`liner` 所以这里要记录的权重也就是我们的线性层矩阵。这里的ffn_norm.weight就是我们之前的，可学习权重g了。这里也是我们为什么要进行区分带权重的层和不带权重的层了，带权重的层，要从model的bin中进行初始化读取，不带权重，只用初始化一次，整个过程中通用。

```c++
  if (!llama_layers_) {
    llama_layers_ = std::make_unique<LLama2Layers>();
  }
```

 所以在这里，我们先用结构体去初始化我们的类。接下来，这个过程我们会初始化一个pos值去索引，按照我们写入模型的顺序去初始我们每一层的权重。

```c++
  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    auto wq = std::make_shared<op::MatmulLayer>(device_type_, dim, dim);
    wq->set_weight(0, {dim, dim}, this->raw_model_data_->weight(pos), cpu_device_type);
    llama_layers_->wq_layers_.push_back(wq);
    pos += dim * dim;
  }
```

总结：

- 本质上，这里就是在看我的导出文件是怎么写的，然后从中读取我们的权重到我们模型中的对应层。
- 带有权重的算子层要进行单独初始化。
- 不带有权重的算子，可能需要根据模型的配置来进行初始化。两遍分别初始化就好了。

## 5.初始化内存缓冲区

首先根据推理的类型来加载我们的内存缓冲器

```C++
if (device_type_ == base::DeviceType::kDeviceCPU) 
  alloc = base::CPUDeviceAllocatorFactory::get_instance();
else 
  alloc = base::CUDADeviceAllocatorFactory::get_instance();
```

这个类是全局的工厂类，如果我们需要将模型迁移到GPU，那么我们可以调用之前的`tensor` 操作，TOGPU

```c++
if (add_layer_) 
{
  add_layer_->set_cuda_config(config);
  add_layer_->to_cuda();
}
```

这里会直接把对应的Tensor的layer直接转移到GPU上，所以把模型转移到GPU上运算的本质就是

- 选择GPU算子
- 缓存存放地点选择GPU

初始化输入的token向量

```c++
tensor::Tensor input_tokens(base::DataType::kDataTypeInt32, 1, true, alloc_cpu);
tensor::Tensor input_embeddings(base::DataType::kDataTypeFp32, 1, config_->dim_, true, alloc);
```

这里的维度都设置为1，后期我们的一个一个的进行输入。输入在CPU上，所以需要我们的CPU内存分配器来管理。下面我们初始化我们的旋转位置编码

```
tensor::Tensor sin_cache(base::DataType::kDataTypeFp32, config_->head_size_ * config_->seq_len_,true, alloc);
tensor::Tensor cos_cache(base::DataType::kDataTypeFp32, config_->head_size_ * config_->seq_len_,true, alloc);

```

用一个`  std::map<ModelBufferType, tensor::Tensor> buffers_;` 来管理我们的buffer，我们需要进行查询的之后，直接将相应的buffer的类型提取出来进行分析即可。

```c++
base::Status Model::insert_buffer(ModelBufferType buffer_idx, const tensor::Tensor& tensor) {
  if (buffers_.count(buffer_idx) > 0) {
    return base::error::KeyHasExits(std::to_string(int(buffer_idx)) + " has exits in the buffers");
  }
  if (tensor.is_empty()) {
    return base::error::InvalidArgument("The tensor is empty for inserting buffer.");
  }
  buffers_.insert({buffer_idx, tensor});
  return base::error::Success();
}
```

建立缓存区结果的复用,RMSNorm → Attention → Add → RMSNorm → FFN → Add.也就是说，我们将结果存放于同一个tensor地址之中，这样有助于节省我们的缓存开销。

```c++
  tensor::Tensor rms_output(base::DataType::kDataTypeFp32, config_->dim_, true, alloc);
  CHECK(insert_buffer(ModelBufferType::kOutputRMSNorm, rms_output));
  CHECK(insert_buffer(ModelBufferType::kOutputMHA, rms_output));
  CHECK(insert_buffer(ModelBufferType::kW2Output, rms_output));
  CHECK(insert_buffer(ModelBufferType::kFFNRMSNorm, rms_output));
```

KVCache缓存，经典中的经典，无需多言！

```c++
tensor::Tensor key_cache(base::DataType::kDataTypeFp32, config_->layer_num_, config_->seq_len_,config_->kv_dim_, true, alloc);
tensor::Tensor value_cache(base::DataType::kDataTypeFp32, config_->layer_num_, config_->seq_len_,config_->kv_dim_, true, alloc);
CHECK(insert_buffer(ModelBufferType::kKeyCache, key_cache));
CHECK(insert_buffer(ModelBufferType::kValueCache, value_cache));
```

存储模型中间计算的缓存：

```c++
  tensor::Tensor query(base::DataType::kDataTypeFp32, config_->dim_, true, alloc);
  CHECK(insert_buffer(ModelBufferType::kQuery, query));
  tensor::Tensor pos_tensor(base::DataType::kDataTypeInt32, 1, true, alloc_cpu);
  CHECK(insert_buffer(ModelBufferType::kInputPos, pos_tensor));
  tensor::Tensor attn(base::DataType::kDataTypeFp32, config_->head_num_, config_->seq_len_, true,alloc);
  CHECK(insert_buffer(ModelBufferType::kScoreStorage, attn));
  CHECK(insert_buffer(ModelBufferType::kAttnOutput, query));
  // final forward output
  tensor::Tensor forward_output(base::DataType::kDataTypeFp32, config_->vocab_size_, true, alloc);
  if (device_type_ == base::DeviceType::kDeviceCUDA) 
  {
    tensor::Tensor forward_output_cpu(base::DataType::kDataTypeFp32, config_->vocab_size_, true,alloc_cpu);
   CHECK(insert_buffer(ModelBufferType::kForwardOutputCPU, forward_output_cpu));
  }
  CHECK(insert_buffer(ModelBufferType::kForwardOutput, forward_output));
```

总结：

- 这里最主要的作用是建立缓存，存储我们模型中间推理过程的输入和输出
- 因为模型师一层一层的计算的，所以很多结果用过一遍之后，下一次进来就可以被覆盖掉
- 这样可以很好的节省我们的显存空间的目的，所以我们才需要大费周章的为模型中间的推理留出我们的计算空间值。

## 6. 初始化位置编码矩阵与采样器

```c++
  if (device_type_ == base::DeviceType::kDeviceCPU) 
  {
    kernel::sin_cos_cache_calc_cpu(config_->head_size_, config_->seq_len_,
                                   get_buffer(ModelBufferType::kSinCache).ptr<float>(),
                                   get_buffer(ModelBufferType::kCosCache).ptr<float>());
  } else {
    CHECK_NE(cuda_config_, nullptr);
    kernel::sin_cos_cache_calc_cu(config_->head_size_, config_->seq_len_,
                                  get_buffer(ModelBufferType::kSinCache),
                                  get_buffer(ModelBufferType::kCosCache), cuda_config_->stream);
  }

  sampler_ = std::make_unique<sampler::ArgmaxSampler>(device_type_);
```

  最后，我们完成一些采样器和位置编码矩阵的初始化，那么我们模型的初始化就完成了。



## 7. 总结

- 模型类的属性初始化（为了方便后面初始化layer配置参数）
- 加载分词器embedding
- mmap，将模型的所有数据映射到内存空间方便读取
- 初始化layer层（带权重+不带权重）
- 初始化缓存
  - 内存分配器的初始化，
  - kv Cache
  - 推导过程中中间结果的存放

- 初始化位置编码举着与采样器

这里就是整个模型初始化过程中，要干的事情。
