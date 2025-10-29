---
title: 统一接口，多派分发
published: 2025-10-27
description: 如何利用统一接口，多派分发来实现高内聚、低耦合的基类设计
tags: [C++, 设计模式]
category: C++
draft: false
---

## 1.问题描述

在我们的类的设计中，比如神经网络中编写`layer` 基础类的时候，子类会重写`forward` 函数以实现具体的前向传递细节，但是参数的差异是多样化的，不同layer可能会重写带有不同参数的forward

```c++

  base::Status forward() override;
  base::Status forward(const tensor::Tensor& input1, const tensor::Tensor& output1) override;
  base::Status forward(const tensor::Tensor& input1, const tensor::Tensor& input2,
                       const tensor::Tensor& output1) override;
  base::Status forward(const tensor::Tensor& input1, const tensor::Tensor& input2,
                       const tensor::Tensor& input3, const tensor::Tensor& output1) override;
  base::Status forward(const tensor::Tensor& input1, const tensor::Tensor& input2,
                       const tensor::Tensor& input3, const tensor::Tensor& input4,
                       const tensor::Tensor& output1) override;
  base::Status forward(const tensor::Tensor& input1, const tensor::Tensor& input2,
                       const tensor::Tensor& input3, const tensor::Tensor& input4,
                       const tensor::Tensor& input5, const tensor::Tensor& output1) override;
```

好比上面这个例子，我们的子类覆写的时候可能会关注需要传递的参数细节，来选择需要重写的函数。但是我们如果利用多派分发的话，就可以很好的解决这个问题，让我们的子类不用关心覆写哪个版本的forward，统一覆写`forward()` 即可。

## 2.多派分发

```c++
base::Status Layer::forward(const tensor::Tensor& input1, const tensor::Tensor& output1) {
  this->set_input(0, input1);
  this->set_output(0, output1);
  return this->forward();
}
base::Status Layer::forward(const tensor::Tensor& input1, const tensor::Tensor& input2,
                            const tensor::Tensor& output1) {
  this->set_input(0, input1);
  this->set_input(1, input2);

  this->set_output(0, output1);
  return this->forward();
}
base::Status Layer::forward(const tensor::Tensor& input1, const tensor::Tensor& input2,
                            const tensor::Tensor& input3, const tensor::Tensor& output1) {
  this->set_input(0, input1);
  this->set_input(1, input2);
  this->set_input(2, input3);
  this->set_output(0, output1);
  return this->forward();
}
```

这样我们就可以实现调用父类：参数传递的流程，调用子类：具体计算的实现。所以在我们的子类中，我们统一只用覆写`forward()` 的实现即可，而不用单独的去管理，去思考**我需要覆盖父类forward的哪个传参版本**

## 3.总结

统一接口，多派分发的核心思路就是：

- 父类调用子类的接口

- 子类示例调用父类的接口，父类完成流程处理与参数传递再调用子类的覆写方法（多态）

- 这样子类就可以不用重新处理一遍相同的数据了

  - 子类中就省去了

  ```c++
    this->set_input(0, input1);
    this->set_input(1, input2);
    this->set_output(0, output1);
  ```

  参数传递的部分。