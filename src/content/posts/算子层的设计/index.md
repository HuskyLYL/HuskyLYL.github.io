---
title: 算子层的设计
published: 2025-10-20
description: 设计了算子层，来对我们的tensor量进行处理
tags: [AI Infra, C++, miniInfer]
category: miniInfer
draft: true
cover: https://github.com/HuskyLYL/HuskyLYL.github.io/blob/main/src/contents/img/aaa.jpg
---

## 1.算子层引言

  算子是构建神经网络模型的基础，它们定义了数据在网络中的流动方式。我们主要考虑带参和不带参的算子，后续的算子都会继承这个基类。

## 2.算子基类型

```c++
class BaseLayer {
 public:
  explicit BaseLayer(base::DeviceType device_type, LayerType layer_type,
                     base::DataType data_type, std::string layer_name = "");

  base::DataType data_type() const;

  LayerType layer_type() const;

  ...
  ...
      
  const std::string& get_layer_name() const; // 返回层的名字

  void set_layer_name(const std::string& layer_name); // 设置层的名称

  base::DeviceType device_type() const; // 返回层的设备类型

  void set_device_type(base::DeviceType device_type);

 protected:
  std::string layer_name_; // 层名
  LayerType layer_type_ = LayerType::kLayerUnknown; // 层类型
  base::DataType data_type_ = base::DataType::kDataTypeUnknown; // 层数据类型
  base::DeviceType device_type_ = base::DeviceType::kDeviceUnknown;
};
```

  核心是我们根据`layer_type` 去实现我们的算子，`device_type` 定义了我们的算子是GPU算子还是CPU算子。



## 3.不带权重的算子类的设计

```c++
class Layer : public BaseLayer {
 public:
  explicit Layer(base::DeviceType device_type, LayerType layer_type,
                 std::string layer_name = "");
  void set_input(int32_t idx, const tensor::Tensor& input) override; // 传入输入
  void set_output(int32_t idx, const tensor::Tensor& output) override; // 传入输出
  const tensor::Tensor& get_input(int32_t idx) const override; // 获取输入
  const tensor::Tensor& get_output(int32_t idx) const override; // 获取输出
  size_t input_size() const override; // 获取输入的个数
  size_t output_size() const override; // 获取输出的个数
  void reset_input_size(size_t size);
  void reset_output_size(size_t size);
  virtual void to_cuda();
 private:
  std::vector<tensor::Tensor> inputs_;  // 存放输入的数组
  std::vector<tensor::Tensor> outputs_; // 存放输出的数组
};
```

  这个类的核心设计逻辑是通过`inputs_` 进行传参输入（检查与越界的操作），最后将我们的值放回outputs_进行处理，最终，我们调用不同的算子`base_forward` 进行传参调用，调用我们封装的核函数来进行操作。

```c++
base::Status base_forward() override; // 每个算子的计算过程都有些不同，所以需要重写base_forward.
base::Status VecAddLayer::base_forward() {
  auto status = this->check();
  if (!status) {
    return status;
  }
  auto input1 = this->get_input(0);
  auto input2 = this->get_input(1);
  auto output = this->get_output(0);
  kernel::get_add_kernel(device_type_)(input1, input2, output, nullptr);
  return base::error::Success();
}
```

  这样就可以很好的丰富可拓展性了，要实现不同的算子，我们只需要编写不同的类继承Layer，然后override`base_forward` 函数即可。这里最关键的其实是`check` 函数，调用之前检查tensor的数据类型和dim，这里再次体检了我们前文对`buffer` 指针进行封装的重要性了。



## 4.带参的算子实现 

  这里其实没什么额外的设计，仅仅只是多了一个`weight` 的`tensor` ,然后计算过程也是一样的，额外的传入一个weight的参数，多几个对weight tensor张量的检查就行了，其实逻辑很好理解。

```c++
base::Status RmsNormLayer::base_forward() { // 计算的时候
  auto status = check();
  if (!status) {
    return status;
  }
  auto input = this->get_input(0);
  auto weight = this->get_weight(0);
  auto output = this->get_output(0);
  // 得到一个具体的算子计算实现
  kernel::get_rmsnorm_kernel(device_type_)(input, weight, output,
                                           cuda_config_ ? cuda_config_->stream : nullptr);
  return base::error::Success();
}
```



## 5.获取不同设备的算子

```c++
base::Status VecAddLayer::base_forward() {
  auto status = this->check();
  if (!status) {
    return status;
  }
  auto input1 = this->get_input(0);
  auto input2 = this->get_input(1);
  auto output = this->get_output(0);
  kernel::get_add_kernel(device_type_)(input1, input2, output);
  return base::error::Success();
}

AddKernel get_add_kernel(base::DeviceType device_type) {
  if (device_type == base::DeviceType::kDeviceCPU) {
    return add_kernel_cpu; // 返回一个具体的函数指针
  } else if (device_type == base::DeviceType::kDeviceCUDA) {
    return add_kernel_cu;
  } else {
    LOG(FATAL) << "Unknown device type for get a add kernel.";
    return nullptr;
  }
}
```

  这里我们根据前文传入的`DeviceType` 变量进行读取，返回适应的算子即可。



## 6.链路总结

1. `base_forward` 调用
2. 在`base_forwad` 中检查`output` `input` `weight` 是否合规
3. 然后选择合适的算子核进行调用

所以总而言之，算子层就是根据我们的数据类型去检查tensor是否合规，根据网络类型调用合适的算子进行计算，并返回结构即可。
