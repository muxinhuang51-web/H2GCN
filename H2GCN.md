## 问题：
当周边节点并不是同类型占绝对多数时，直接使用GNN可能会导致传递的噪声比信息多，因为他把邻居和自身当作同一种节点，此时直接聚合信息可能导致两种相反的节点成为彼此的噪声
## 论文的结果：
1. 在低同质性的前提下，通过设计模块实现预测目标准确率提升。
2. 同质性问题中也保持竞争力

## 方法说明：
1. 自我是与邻居分离的，每一个节点我们不过早与邻居混合，先通过权重变换+非线性激活得到特征编码

普通 GCN 的一阶传播可以写成：

```text
H = (A + I)XW
```

`(A + I)X` 表示把节点自身特征和一阶邻居特征直接混合。  
在低同质性图中，一阶邻居可能大多是异类节点，因此这种早期混合可能把噪声引入节点表示。

H2GCN 的第一步是先对节点自身特征做编码：

```text
R_v^(0) = sigma(X_v W_e)
```


```text
ego embedding 和 neighbor embedding 分离。
```

/home/hmx42/Projects/H2GCN/figure/模块1.png


2. 我们设计了更高阶的节点而不是单纯的邻居

一阶邻居表示可以写成：

```text
R_v^(1) = AGGREGATE({R_u^(0) : u in N_1(v)})
```

二阶邻居表示可以写成：

```text
R_v^(2) = AGGREGATE({R_u^(0) : u in N_2(v)})
```

矩阵形式上可以粗略理解为：

```text
AX      : 一阶邻居信息
A^2X    : 二阶邻居信息
```

/home/hmx42/Projects/H2GCN/figure/模块2.png

3. 结合不同的中间层的输出，最后使用combine合并自己和每一轮邻居，拼接后线性变换加非线性变换
第 k 层表示可以写成：

```text
R_v^(k) = AGGREGATE({R_u^(k-1) : u in N(v)})
```

如果只使用最后一层：

```text
R_v^(final) = R_v^(K)
```

可能会因为多次邻域传播而过度平滑，丢失低同质性图中的高频/局部判别信息。

H2GCN 使用中间层组合：

```text
R_v^(final) = COMBINE(R_v^(0), R_v^(1), ..., R_v^(K))
```

常见实现是拼接：

```text
R_v^(final) = CONCAT(R_v^(0), R_v^(1), ..., R_v^(K))
```

简要理解：

```text
浅层保留更多原始/高频/局部信息；
深层包含更多传播后/低频/高阶邻域信息；
拼接中间层可以让模型自己选择不同深度的信息。
```
也即combine函数

/home/hmx42/Projects/H2GCN/figure/模块三.png

## 实验
1. 使用官方 `syn-cora` npz 数据。
2. 在原 `h2gcn/run_experiments.py` 入口上读取 npz 数据。
3. 对比低/中/高同质性图上的 H2GCN-1 和 H2GCN-2 表现。
4. 观察模型在不同 homophily 水平下的趋势。

这是 npz 数据上的 method-level reproduction，不是论文原始 signac split 的严格复现。
## 总结：
同时在同质性和异质性中学习的模型


## 后续：https://www.jiongzhu.net/revisiting-heterophily-GNNs/
#### 重新探讨了GNN的异亲性问题：他们观察到，在某些合成和现实世界的异亲数据集中，GCN仍能展现出竞争性能，在某些情况下甚至超过具有异亲性设计的模型（例如H2GCN）。这些观察似乎与以往研究中GCN通常不适合异性恋图的结论相矛盾。
#### 生成图的方法不一样
