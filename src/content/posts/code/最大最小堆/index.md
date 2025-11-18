---
title: 最大最小堆
published: 2025-11-18
description: 最大最小堆模板
tags: [C++, code]
category: code
draft: false
image: ./cover.png
---

## 1.上浮操作

- 就是和自己的父节点比较，如果小于父节点就上浮动
- 然后让父节点也开始上浮
- 递推此过程

```c++
void heapifyUp(int index) 
{
	int parent = (index - 1) / 2;
    //注意这里是index > 0 而不是parent > 0
    //这个问题要理清了
    if (index > 0 && compare(heap[index], heap[parent])) 
    {
        std::swap(heap[index], heap[parent]);
        heapifyUp(parent);
    }
}
```

## 2.下沉操作

- 下沉操作就是从当前节点开始，找到最小值的子节点（），然后交换
- 必须比父节点小
- 从上往下依次递推

```c++
    void heapifyDown(int index) 
    {
        int leftChild = 2 * index + 1;
        int rightChild = 2 * index + 2;
        int smallestOrLargest = index;

        if (leftChild < heap.size() && compare(heap[leftChild], heap[smallestOrLargest]))
            smallestOrLargest = leftChild;

        if (rightChild < heap.size() && compare(heap[rightChild], heap[smallestOrLargest]))
            smallestOrLargest = rightChild;
       
        if (smallestOrLargest != index)
        {
            std::swap(heap[index], heap[smallestOrLargest]);
            heapifyDown(smallestOrLargest);
        }
        
    }
```

## 3.插入操作

```c++
    // 插入一个元素
    void push(int value) {
        heap.push_back(value);
        heapifyUp(heap.size() - 1);
    }
```

- 直接把数据加入到数组的末尾
- 然后开始上浮操作

## 4.删除操作

```c++
    void pop() {
        if (heap.empty()) {
            throw std::out_of_range("Heap is empty");
        }
        std::swap(heap.front(), heap.back());
        heap.pop_back();
        heapifyDown(0);
    }
```

- 将头部的元素移动到末尾，和末尾元素交换
- 末尾元素直接下沉
- 然后完成此步骤













