---
title: 自上而下的RAII和智能指针技术管理buffer
published: 2025-10-18
description: 如何利用C++智能指针对如何管理CPU和GPU上的内存，防止内存泄漏的总结.
tags: [AI Infra, C++]
category: Infra
draft: false
cover: https://github.com/HuskyLYL/HuskyLYL.github.io/blob/main/src/contents/img/aaa.jpg
---

## 1.内存分配器管理

```c++
class CUDADeviceAllocatorFactory {
public:
	static std::shared_ptr<CUDADeviceAllocator> get_instance(){
    	if (instance == nullptr) 
      		instance = std::make_shared<CUDADeviceAllocator>();
    	return instance;
  }
private:
	static std::shared_ptr<CUDADeviceAllocator> instance;
};
```

​	这里利用工厂的形式，创建和管理全局唯一的一个内存分配器，利用C++智能指针技术。这样我们不仅可以保证全局实例的唯一性，还可以继续深入Allocator的析构设计去更好的管理我们的内存，这样就可以避免因为手动释放内存带来的内存泄漏。

## 2.内存分配器设计

```c++
class CUDADeviceAllocator : public DeviceAllocator{
public:
  	explicit CUDADeviceAllocator();
 	void* allocate(size_t byte_size) const override;
  	void release(void* ptr) const override;
private:
    
	mutable std::map<int, size_t> no_busy_cnt_;
    mutable std::map<int, std::vector<CudaMemoryBuffer>> big_buffers_map_;
	mutable std::map<int, std::vector<CudaMemoryBuffer>> cuda_buffers_map_;
};

```

​	在DeviceAllocator类中，有一些基础的内存申请、释放、异步、同步、对齐的操作，这里就不仔细展开了。这里我们利用map去映射我们我们不同设备上的大小内存块。`no_busy_cnt_` 用来表示小内存块的内存。这里分配器要实现的核心的目标：

- 减少free，allocate的使用，提高内存复用率，提高效率
- 然后就是内存大小块的管理，大小块分开，减少碎片
- 要支持多GPU，多设备

### 2.1内存的分配

​	我们需要针对大内存块和小内存块进行分开管理，核心思路是：

1. 先判断属于大内存块还是小内存块。
2. 对已经分配好的内存块，选取一个free且产生碎片最少的。
3. 如果当前设备的空间已经占满，则重新分配空间。

### 2.2内存的释放

​	首先，我们需要清理一下过多的小空间，也就是内存碎片，去手动释放。然后再去释放ptr指针的内存。

```c++
  for (auto& it : cuda_buffers_map_) {
    auto& cuda_buffers = it.second;
    for (int i = 0; i < cuda_buffers.size(); i++) {
      if (cuda_buffers[i].data == ptr) {
        no_busy_cnt_[it.first] += cuda_buffers[i].byte_size;
        cuda_buffers[i].busy = false;
        return;
      }
    }
    auto& big_buffers = big_buffers_map_[it.first];
    for (int i = 0; i < big_buffers.size(); i++) {
      if (big_buffers[i].data == ptr) {
        big_buffers[i].busy = false;
        return;
      }
    }
  }
  state = cudaFree(ptr);  
  CHECK(state == cudaSuccess) << "Error: CUDA error when release memory on device";
```

​	注意这里将busy标记为false之后一定要记得返回，一定不能被`cudaFree` 给释放掉，否则就不在我们的资源池管理之中了，这就体现了全局一个资源分配器的重要性，否则buffer空间不统一容易造成管理混乱。

## 3. Buffer类的设计

​	前面我们的内存分配器类已经设计完成了，但是出现了问题，全局只有一个内存分配器，那怎么确保内存分配器能够即使的释放资源呢？这里还是需要我们手动的去free，会略显麻烦。所以我们可以创建一个继承`std::enable_shared_from_this<Buffer>` 的Buffer类，在离开作用域的时候，能够自己调用内存管理类进行释放！

### 3.1 enable_shared_from_this类的作用

`std::enable_shared_from_this<T>` 的目的主要是为了能够共享引用计数的情况，防止两次析构，例如：

```c++
Buffer* buf = new Buffer();
std::shared_ptr<Buffer> sp1(buf);
std::shared_ptr<Buffer> sp2(buf); 
```

- 这会让sp1 和 sp2各自产生自己的引用计数，最后析构两遍造成内存管理错误

所以我们让Buffer类能够继承`std::enable_shared_from_this<T>` ,在引用的时候能够共享一份引用计数

```c++
auto sp1 = std::make_shared<Buffer>();
auto sp2 = sp1->shared_from_this();
```



###  3.2 类的设计

```c++
  class Buffer : public NoCopyable, std::enable_shared_from_this<Buffer> 
  {
  private:
    size_t byte_size_ = 0;
    void* ptr_ = nullptr;
    bool use_external_ = false;
    DeviceType device_type_ = DeviceType::kDeviceUnknown;
    std::shared_ptr<DeviceAllocator> allocator_;

  public:
    explicit Buffer() = default;
    explicit Buffer(size_t byte_size, std::shared_ptr<DeviceAllocator> allocator = nullptr,
                  void* ptr = nullptr, bool use_external = false);
    virtual ~Buffer();
    bool allocate();
    void copy_from(const Buffer& buffer) const;
    void copy_from(const Buffer* buffer) const;
    void* ptr();
    const void* ptr() const;
    size_t byte_size() const;
    std::shared_ptr<DeviceAllocator> allocator() const;
    DeviceType device_type() const;
    void set_device_type(DeviceType device_type);
    std::shared_ptr<Buffer> get_shared_from_this();
    bool is_external() const;
  };
```



### 3.3 析构类的时候自动释放资源

```c++
Buffer::~Buffer() {
  if (!use_external_) {
    if (ptr_ && allocator_) {
      allocator_->release(ptr_);
      ptr_ = nullptr;
    }
  }
}
```

- 这里就是精髓了，一般我们通过智能指针动态的管理资源就可以了
- 只需要将一个量设置为use_external_,确保最后只有一个成员会对Buffer析构
- 不过这里也会有一定的风险，`use_external_ = true` 的类，可能会面临资源已经被释放的情况



### 3.4判断不同的拷贝类型，调用allocate的不同拷贝函数

```c++
void Buffer::copy_from(const Buffer* buffer) const 
{
  CHECK(allocator_ != nullptr);
  CHECK(buffer != nullptr || buffer->ptr_ != nullptr);
  //这里是考虑数据的大小问题
  size_t dest_size = byte_size_;
  size_t src_size = buffer->byte_size_;
  size_t byte_size = src_size < dest_size ? src_size : dest_size;
  const DeviceType& buffer_device = buffer->device_type();
  const DeviceType& current_device = this->device_type();
  CHECK(buffer_device != DeviceType::kDeviceUnknown &&
        current_device != DeviceType::kDeviceUnknown);
  if (buffer_device == DeviceType::kDeviceCPU &&
      current_device == DeviceType::kDeviceCPU) {
    return allocator_->memcpy(buffer->ptr_, this->ptr_, byte_size);
  } 
  else if (buffer_device == DeviceType::kDeviceCUDA &&
             current_device == DeviceType::kDeviceCPU) {
    return allocator_->memcpy(buffer->ptr_, this->ptr_, byte_size,
                              MemcpyKind::kMemcpyCUDA2CPU);
  } else if (buffer_device == DeviceType::kDeviceCPU &&
             current_device == DeviceType::kDeviceCUDA) {
    return allocator_->memcpy(buffer->ptr_, this->ptr_, byte_size,
                              MemcpyKind::kMemcpyCPU2CUDA);
  } else {
    return allocator_->memcpy(buffer->ptr_, this->ptr_, byte_size,
                              MemcpyKind::kMemcpyCUDA2CUDA);
  }
}
```

## 4.总结：

核心思路：

- 内存管理与分配器确保全局唯一性，static静态成员：便于管理全局唯一的buffer空间
- 智能指针设计buffer，配合内存分配器析构，确保不会出现内存泄漏等问题







