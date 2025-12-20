---
title: dangoInfer开发日志
published: 2025-11-21
description: 主要用于记录项目进度与思路
tags: [AI Infra, C++, DangoInfer]
category: dangoInfer
draft: false
image: ./cover.png
---

## 11.18日

- 完成底层内存分配器的工厂类
  - 计划在工厂类的基础上打造`pageAttention` 的内存管理方式。

## 11.19日

- alloc重写了一下
  - 目的是为了兼容多GPU,管理多GPU设备之间的内存、
- 优化了alloc->memcpy 的结构，不需要复杂的参数传递啦



## 11.21日：

- 完成Tensor类的重构
- 不需要传入设备分配器，而是根据传入的参数自己获取全局设备分配器



## 11.24日:

- tensor的 tocuda 和 toCPU 需要完成 buffer 的 copy from 的异步拷贝
- 所以今天需要兼容buffer的 async的异步拷贝
- 全局设备管理器做了优化，不需要筛选CPU 和 CUDA 的设备管理器


## 12.16日:

- 第一阶段完成
- 调研flashAttention的实现
