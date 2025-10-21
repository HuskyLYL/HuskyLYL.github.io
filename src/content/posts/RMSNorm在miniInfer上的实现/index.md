---
title: RMSNorm在miniInfer上的实现
published: 2025-10-21
description: 在miniInfer上编写RMSNorm的接口
tags: [AI Infra, C++, miniInfer,算子]
category: miniInfer
draft: true
cover: https://github.com/HuskyLYL/HuskyLYL.github.io/blob/main/src/contents/img/aaa.jpg
---

## 1.RMSNorm介绍

$$
RMSNorm(x)=\frac{x}{RMS(x)} ⊙g
$$

$$
RMS(x)=\sqrt{\frac{1}{d}\sum_{i=1}^{d}  x_{i}^{2}+𝜖}
$$

  RMSNorm是一种比LayerNorm更轻量化的归一方法，其中𝜖是一个比较小的数字，是为了防止除0