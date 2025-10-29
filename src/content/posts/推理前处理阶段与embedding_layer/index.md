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















































































