---
title: 移动构造
published: 2025-10-29
description: 一个C—++类构造的小技巧
tags: [AI Infra, C++, miniInfer]
category: C++
draft: false
imgae: https://github.com/HuskyLYL/HuskyLYL.github.io/blob/main/src/contents/img/aaa.jpg
---
## 1. 案例说明：

```c++
struct EmbeddingOutput {
  tensor::Tensor input_tokens;
  tensor::Tensor input_embeddings;
  tensor::Tensor input_token_num;
  explicit EmbeddingOutput(tensor::Tensor input_tokens, tensor::Tensor input_embeddings,
                           tensor::Tensor input_token_num)
      : input_tokens(std::move(input_tokens)),
        input_embeddings(std::move(input_embeddings)),
        input_token_num(std::move(input_token_num)) {}
};
```

`std::move` 主要用来将左值转换为右值引用。这是移动语义的核心。通过这种转换，我们可以**允许资源的转移而不是复制**。因为我们在构造累的时候，就进行了一次参数的传递，然后将传入进去的参数拷贝到类的成员，又会进行一次参数的传递，所以为了避免这样的行为的发生，我们需要使用`move` ，通过将左值变成右值。

左值：指的是 **有名称的对象**，可以出现在赋值表达式的 **左边**，表示一个可以修改的持久对象的地址。换句话说，左值是一个存在于内存中的对象，它可以被修改。

右值：是不具名的临时对象，它不可以取地址，并且通常存在于表达式的 **右边**。右值通常是 **不能修改** 的值，代表的是一个短暂存在的值或者临时对象。

## 2.总结：

当我们进行变量赋值的时候，可以将不再用到的左值变成右值，然后给新的左值赋值，这样就可以避免进行下一次变量拷贝了。
