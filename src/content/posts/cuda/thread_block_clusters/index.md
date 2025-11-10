---
title: thread_block_clusters
published: 2025-11-10
description: 总结了thread_block_clusters的技术要点与拓展了相关的概念
tags: [AI Infra, C++, cuda]
category: cuda
draft: true
image: ./cover.png
---

## 1.导论-基础概念

一个簇内的所有线程块也会被保证在同一个GPU处理簇上同时调度执行，最大簇的数量一般随着硬件的改变而改变。所以我们可以这样解答一开始的困惑，只是更大层面的对齐：

- 单个的block依然在单个SM上调度执行
- 线程块簇只是确保了线程块簇内的元素在我们特定的GPC上执行也就是特定的GPC（GPU处理簇）上运行

## 2.Example调用例子

```c++
// Kernel definition
// Compile time cluster size 2 in X-dimension and 1 in Y and Z dimension
__global__ void __cluster_dims__(2, 1, 1) cluster_kernel(float *input, float* output)
{

}

int main()
{
    float *input, *output;
    // Kernel invocation with compile time cluster size
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks(N / threadsPerBlock.x, N / threadsPerBlock.y);

    // The grid dimension is not affected by cluster launch, and is still enumerated
    // using number of blocks.
    // The grid dimension must be a multiple of cluster size.
    cluster_kernel<<<numBlocks, threadsPerBlock>>>(input, output);
}
```

- 现成块簇的启动方法：编译期间指定，这里没啥说的，只是在核函数那里多可一个声明制定了现成块簇的大小。

```c++
// Kernel definition
// No compile time attribute attached to the kernel
__global__ void cluster_kernel(float *input, float* output)
{

}

int main()
{
    float *input, *output;
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks(N / threadsPerBlock.x, N / threadsPerBlock.y);

    // Kernel invocation with runtime cluster size
    {
        cudaLaunchConfig_t config = {0};
        // The grid dimension is not affected by cluster launch, and is still enumerated
        // using number of blocks.
        // The grid dimension should be a multiple of cluster size.
        config.gridDim = numBlocks;
        config.blockDim = threadsPerBlock;

        cudaLaunchAttribute attribute[1];
        attribute[0].id = cudaLaunchAttributeClusterDimension;
        attribute[0].val.clusterDim.x = 2; // Cluster size in X-dimension
        attribute[0].val.clusterDim.y = 1;
        attribute[0].val.clusterDim.z = 1;
        config.attrs = attribute;
        config.numAttrs = 1;

        cudaLaunchKernelEx(&config, cluster_kernel, input, output);
    }
}
```

- 线程块簇也可以在启动的时候指定我们的大小来判断
- 这里只是一种启动的时候动态指定我们的线程块簇的大小

## 3.现成块簇存在的意义：

- 这使得多个block块之间可以执行硬件级别的同步操作
- 更大的存储空间（DSM）（硬件级别的“簇级共享内存”），比全局内存访问要快



