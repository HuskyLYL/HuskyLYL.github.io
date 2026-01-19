---
title: slang导论
published: 2026-01-19
description: 从 HLSL 出发，系统讲解 Slang 的编译模型与 API 调用流程，理解 Session、EntryPoint 与多后端 GPU 内核生成的核心机制。
tags: [slang]
category: slang
draft: false
image: ./cover.png
---

## 1.相关概念

### HLSL

**High-Level Shader Language** 是 **微软为 GPU 编程设计的高级着色语言**。

| HLSL 概念   | 类比 CUDA     |
| ----------- | ------------- |
| thread      | CUDA thread   |
| group       | CUDA block    |
| Dispatch    | kernel launch |
| groupshared | shared memory |

### 如何理解slang

Slang 的重心在于：**把 GPU 程序从源码，稳定、可组合地编译到不同后端**。

- Slang 语言：源码（HLSL 超集）
- Slang IR（统一中间表示）
- 目标后端选择
  - DirectX（DXIL）
  - SPIR-V （Vulkan）
  - Metal
  - CUDA/CPU

### 常见后端（图形计算API）：GPU计算层之上

#### OPENGL：

- 驱动在 API 调用时：

  - 自动管理资源状态

  - 自动插同步

  - 自动做大量校验

- 很多逻辑 **藏在驱动里**

#### Vulkan/DirectX：

- API 几乎只做：

  - 参数合法性检查

  - 命令转发

- **资源状态、同步、生命周期**都交给应用



| 层级         | OpenGL | Vulkan / DX12 |
| ------------ | ------ | ------------- |
| API 抽象     | 高     | 低            |
| 驱动工作量   | 重     | 轻            |
| 应用复杂度   | 低     | 高            |
| 性能可预测性 | 低     | 高            |

#### Slang的层级：

将HLSL转换成 不同后端能够识别到的中间表示。



## 2.Slang 编译API的调用流程

### 2.1常用头文件

```c++
#include "slang.h"
#include "slang-com-ptr.h"
#include "slang-com-helper.h"
```

`slang.h` : 核心COM接口

`slang-com-ptr.h` :slang风格的职能指针，确保slang的资源被释放

`slang-com-helper.h` 一些辅助查询等操作



### 2.2创建全局会话

```c++
ComPtr<slang::IGlobalSession> slangGlobalSession;
RETURN_ON_FAIL(slang::createGlobalSession(slangGlobalSession.writeRef()));
```

- 会话的主要作用是缓存与重用。
- 使用 Slang API 的长生命周期会话比使用独立`slangc`编译器可执行文件进行编译具有很大的优势（每次调用Slangc都会创建一个新的会话对象）。
- 宏的配置不同，缓存就不能安全的复用，最好使用不用的session来隔离不同的宏的组合。



### 2.3加载模块

```c++
slang::IModule* slangModule = nullptr;
{
    ComPtr<slang::IBlob> diagnosticBlob;
    Slang::String path = resourceBase.resolveResource("hello-world.slang");
    slangModule = session->loadModule(path.getBuffer(), diagnosticBlob.writeRef());
    diagnoseIfNeeded(diagnosticBlob);
    if (!slangModule)
        return -1;
}
```

- 创建作用域，出了作用域进行稀构,这里的disgnosticBlob作为out进行接受输出。
- slangModule = session->loadModule(path.getBuffer(), diagnosticBlob.writeRef());
  - 这里把`slangModule ` 理解为把模块编入了session的一个句柄。

- 模块的寿命和会话是一体的，只要会话有效，那么模块就有效。



### 2.4找到入口点

```c++
 ComPtr<slang::IEntryPoint> entryPoint;
 slangModule->findEntryPointByName("computeMain", entryPoint.writeRef());


[shader("compute")]
[numthreads(1,1,1)]
void computeMain(uint3 threadId : SV_DispatchThreadID)
{
    uint index = threadId.x;
    result[index] = buffer0[index] + buffer1[index];
}

```

- slang在编译的时候可能有很多入口点，所以我们需要通过名字来找到。
- 而且需要通过入口函数来获取GPU的组织等信息。
- 同时，需要通过入口函数为挂钩，来编译所能够触及到的所有的函数。



### 2.5构建模块和入口点

```c++
Slang::List<slang::IComponentType*> componentTypes;
componentTypes.add(slangModule);
componentTypes.add(entryPoint);

// 创建组合 component 可能失败（如重复模块），因此要检查诊断信息。
//
ComPtr<slang::IComponentType> composedProgram;
{
    ComPtr<slang::IBlob> diagnosticsBlob;
    SlangResult result = session->createCompositeComponentType
    (
        componentTypes.getBuffer(),
        componentTypes.getCount(),
        composedProgram.writeRef(),
        diagnosticsBlob.writeRef()
    );
    diagnoseIfNeeded(diagnosticsBlob);
    RETURN_ON_FAIL(result);
}
```



### 2.6link

通常是用来补全缺失的依赖项目。



### 2.7获取目标内核代码

```c++
Slang::ComPtr<slang::IBlob> spirvCode;
{
    Slang::ComPtr<slang::IBlob> diagnosticsBlob;
    SlangResult result = linkedProgram->getEntryPointCode
    (
        0, // entryPointIndex
        0, // targetIndex
        spirvCode.writeRef(),
        diagnosticsBlob.writeRef()
    );
    diagnoseIfNeeded(diagnosticsBlob);
    SLANG_RETURN_ON_FAIL(result);
}
```

- 调用此函数`IComponentType::getEntryPointCode()`将执行最终编译。
- 生成目标语言代码并返回`IBlob`指向该代码的指针。



