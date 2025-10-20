---
title: tensor层的设计
published: 2025-10-20
description: 大模型推理的核心
tags: [AI Infra, C++, miniInfer]
category: miniInfer
draft: false
cover: https://github.com/HuskyLYL/HuskyLYL.github.io/blob/main/src/contents/img/aaa.jpg
---

## 1. 为什么需要在buffer的基础上封装tensor

  buffer是分配内存的最底层，我们需要在buffer上再封装一层tensor的目的是为了管理底层的buffer，让我们更直观的去操作buffer，清楚buffer中描述的数据类型。（buffer的形状控制，设备之间的数据传输）

### 1.1. tensor对buffer托管的兼容

```c++
void Tensor::init_buffer(std::shared_ptr<base::DeviceAllocator> alloc, base::DataType data_type,bool need_alloc, void* ptr) 
{
  if (!alloc && !need_alloc) 
  {
    std::shared_ptr<base::Buffer> buffer =std::make_shared<base::Buffer>(data_type_size(data_type) * size_, nullptr, ptr, true);
    this->buffer_ = buffer;
  } 
  else 
  {
    allocate(alloc, true);
  }
}
```

- use_external = true需要注意，buffer不负责释放这段内存，但是这段内存如果被提前释放
- 那么buffer就会出错

### 1.2. tensor的初始化

```c++
Tensor::Tensor(base::DataType data_type, int32_t dim0, int32_t dim1, bool need_alloc,
               std::shared_ptr<base::DeviceAllocator> alloc, void* ptr)
    : data_type_(data_type) 
{
  dims_.push_back(dim0);
  dims_.push_back(dim1);
  size_ = dim0 * dim1;
  if (need_alloc && alloc) 
  {
    //是否需要重新分配
    allocate(alloc);
  } 
  else 
  {
    init_buffer(alloc, data_type_, need_alloc, ptr);
  }
}
```

- tensor会定义是不是需要自己分配alloc，如果是，那么自己构建buffer
- 不过不需要自己alloc，ptr已经分配好了地址，那么只需要创建一个托管buffer就可以了



### 1.3. tensor设备迁移

```c++
void Tensor::to_cuda(cudaStream_t stream) {
  CHECK_NE(buffer_, nullptr);
  const base::DeviceType device_type = this->device_type();
  if (device_type == base::DeviceType::kDeviceUnknown) {
    LOG(ERROR) << "The device type of the tensor is unknown.";
  } else if (device_type == base::DeviceType::kDeviceCPU) {
    size_t byte_size = this->byte_size();
    auto cu_alloc = base::CUDADeviceAllocatorFactory::get_instance();
    auto cu_buffer = std::make_shared<base::Buffer>(byte_size, cu_alloc);
    cu_alloc->memcpy(buffer_->ptr(), cu_buffer->ptr(), byte_size, base::MemcpyKind::kMemcpyCPU2CUDA,
                     stream);
    this->buffer_ = cu_buffer;
  } else {
    LOG(INFO) << "The device type of the tensor is already cuda.";
  }
}

void Tensor::to_cpu() {
  CHECK_NE(buffer_, nullptr);
  const base::DeviceType device_type = this->device_type();

  if (device_type == base::DeviceType::kDeviceUnknown) {
    LOG(ERROR) << "The device type of the tensor is unknown.";
  } else if (device_type == base::DeviceType::kDeviceCUDA) {
    size_t byte_size = this->byte_size();
    auto cpu_alloc = base::CPUDeviceAllocatorFactory::get_instance();
    auto cpu_buffer = std::make_shared<base::Buffer>(byte_size, cpu_alloc);
    cpu_alloc->memcpy(buffer_->ptr(), cpu_buffer->ptr(), byte_size,
                      base::MemcpyKind::kMemcpyCUDA2CPU);
    this->buffer_ = cpu_buffer;
  } else {
    LOG(INFO) << "The device type of the tensor is already cpu.";
  }
}
```

- 这里其实直接调用我们前面封装的工厂获取allocation实例
- 因为我们这里buffer使用的是智能指针，所以可以很大的的new 一个心buffer ，然后总线pic赋值进去
- 最后将原来的buffer变量直接拷贝为新的变量，这个时候智能指针的值-1
- 完全不用担心内存释放的问题。



### 1.4. 总结

  tensor是为了在buffer上面封装一层，提供一些基础的数据接口，也是为了方便我们后面算子层引用的数据结构。