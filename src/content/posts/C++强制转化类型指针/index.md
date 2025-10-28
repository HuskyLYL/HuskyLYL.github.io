---
title: C++强制转化指针类型
published: 2025-10-21
description: 主要面向了C++的几种强制转化类型的指针
tags: [C++]
category: C++
draft: false
---

## 1. 背景

  在大模型推理框架中，我们对模型中的某一层的精度进行量化是我们常用的加速办法，而C++的几种强制类型转化对我们的推理框架中经常使用，为了确保数据内存的安全，我们来对这几种强制指针类型转化做一个总结。

## 2. static_cast

```c++
using namespace std;
using namespace std::chrono;
int main()
{
    double* arr = (double*)malloc(100 * sizeof(double));
    for (int i = 0; i < 100; ++i) 
        arr[i] = i * 0.1;
    auto start1 = high_resolution_clock::now();
    float* f1 = static_cast<float*>(static_cast<void*>(arr));  // 间接转换
    auto end1 = high_resolution_clock::now();
    cout << "[static_cast] 转换时间: "
         << duration_cast<nanoseconds>(end1 - start1).count()
         << " ns" << endl;
    for(int i=0;i<5;i++)
        cout<<f1[i]<<" ";
    cout<<endl;
    return 0;
}
```

  static_cast只做循序的转化，不允许乱转（子类 <-> 父类）.如果想要转化，那么就先转换成void*，但是底层的数据读取精度并没有发生变化哦，所以最后的结果就会出现问题。

```c++
[static_cast] 转换时间: 203 ns
0 0 -1.58819e-23 1.45 -1.58819e-23 
```

## 3.reinterpret_cast

  就是告诉编译器怎么去解释内存，只是改变了指针的标签，但是地址完全保持一致。

```c++
#include <iostream>
#include <cstdlib>
#include <chrono>
using namespace std;
using namespace std::chrono;
int main()
{
    double* arr = (double*)malloc(100 * sizeof(double));
    if (!arr) {
        cerr << "内存分配失败！" << endl;
        return -1;
    }
    for (int i = 0; i < 100; ++i) 
        arr[i] = i * 0.1;
    auto start1 = high_resolution_clock::now();
    float* f1 = reinterpret_cast<float*>(arr);  // 重解释指针类型
    auto end1 = high_resolution_clock::now();
    cout << "[reinterpret_cast] 转换时间: "
         << duration_cast<nanoseconds>(end1 - start1).count()
         << " ns" << endl;
    for(int i = 0; i < 5; i++)
        cout << f1[i] << " ";
    cout << endl;
    free(arr);
    return 0;
}
```

程序结果：

```
[reinterpret_cast] 转换时间: 170 ns
0 0 -1.58819e-23 1.45 -1.58819e-23 
```

看来和我们前面的static_cast一致，只是改变了内存的底层解释方法！



## 4. dynamic_cast

  作用和我们的static_cast比较像，static是在编译期进行转化的，所以不太安全

```c++
struct Base {
    virtual void foo() {}
};

struct Derived : Base {
    void bar() {}
};

Base* b = new Base();            // 实际对象是 Base
Derived* d = static_cast<Derived*>(b); // ❌ 下行转换
Derived* d = dynamic_cast<Derived*>(b); // 这个时候会转化失败，d返回nullptr
```

这个时候，d->bar()会产生未定义的行为，而我们的dynamic_cast会在运行的过程中去检查，这个时候会返回指针，所以当b不是Derived，也不是Derived的子类对象的时候（父类不能算了），那么就会范围空指针，所以这个时候是安全的

## 5. const_cast

  作用比较单一，临时去掉const/volatile

```c++
#include <iostream>
using namespace std;
int main() {
    int x = 10;                
    const int* p = &x;         
    cout << "原始值: " << *p << endl;
    int* q = const_cast<int*>(p);
    *q = 20;  
    cout << "修改后的值: " << x << endl;
    const int y = 100;
    const int* py = &y;
    int* unsafe = const_cast<int*>(py);
    *unsafe = 5;
    cout<<y<<endl;
    return 0;
}
```

```
原始值: 10
修改后的值: 20
100
```

  所以当一个值真的不能修改的时候，const_cast是没有意义的，只有去掉一些const 指针的时候才有意义

## 6.总结

  无论哪种指针都没有对底层数据改变的权限，static_cast和reinterpret_cast最多只改变指针的解读方式。dynamic_cast和static_cast常见第类对象进行类型转换，一个会在运行是检查，更安全，一个则会产生可能产生未定义的行为。在const_casr之中，去掉const关键字是指去掉const指针的，如果底层的数据真的是一个const的，那么即使对这个指针进行修改，也不会有效果的，这一点需要注意。















