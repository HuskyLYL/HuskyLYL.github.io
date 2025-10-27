---
title: 推理过程中的KVcache
published: 2025-10-27
description: 为什么会有KVcache这个东西
tags: [AI Infra, C++, miniInfer]
category: 推理优化
draft: false
cover: https://github.com/HuskyLYL/HuskyLYL.github.io/blob/main/src/contents/img/aaa.jpg
---

## 1. KVcache的由来

  首先，这是由模型的性质来决定的，我们的LLama2在推理的时候是一个token一个token地产生的，这正是自回归语言模型的核心特征，下一次的输入的时候，需要给定前面的所有的token来预测当前的token值。
