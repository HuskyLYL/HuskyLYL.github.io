---
title: llama2-7b载入model
published: 2025-10-27
description: 从llama的结构出发，将model载入miniInfer
tags: [AI Infra, C++, miniInfer]
category: miniInfer
draft: false
image:
  url: './image-20251027102311174.png'
  alt: 'llama2模型概述'
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

## 3.加载模型层的权重





