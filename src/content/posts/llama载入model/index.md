---
title: llama2-7b载入model
published: 2025-10-27
description: 从llama的结构出发，将model载入miniInfer
tags: [AI Infra, C++, miniInfer]
category: miniInfer
draft: true
cover: https://github.com/HuskyLYL/HuskyLYL.github.io/blob/main/src/contents/img/aaa.jpg
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
