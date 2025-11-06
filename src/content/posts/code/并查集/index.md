---
title: 并查集
published: 2025-11-06
description: 并查集的构造，以力扣200为例
tags: [C++, code]
category: code
draft: false
image: ./cover.png
---

## 1. 并查类模板

### 1.1成员函数

```c++
vector<int> parent;
vector<int> rank;
int count;
```

### 1.2节点的初始化

```c++
UnionFind(vector<vector<char>>& grid) 
{
    count = 0;
    int m = grid.size();
    int n = grid[0].size();
    for (int i = 0; i < m; ++i) 
    {
        for (int j = 0; j < n; ++j) 
        {
                if (grid[i][j] == '1')
                {
                    parent.push_back(i * n + j);
                    ++count;
                }
                else {
                    parent.push_back(-1);
                }
                rank.push_back(0);
            }
        }
    }
```

- 开始所有的节点的parent都初始化为自身
- rank设置为0，代表层级，我们应该让层级少的挂在层级大的下面

### 1.3节点的合并

```c++
    void unite(int x, int y) {
        int rootx = find(x);
        int rooty = find(y);
        if (rootx != rooty) {
            if (rank[rootx] < rank[rooty]) {
                swap(rootx, rooty);
            }
            parent[rooty] = rootx;
            //相同才会出现层级+1的情况
            if (rank[rootx] == rank[rooty]) rank[rootx] += 1;
            --count;
        }
    }
```

- 如果father不一样那么考虑合并
- rank小的挂在rank大的下面
- 如果rank一样，那么被挂的rank要+=1.

### 1.4 找父节点

```c++
    int find(int i) {
        if (parent[i] != i) {
            parent[i] = find(parent[i]);
        }
        return parent[i];
    }
```





