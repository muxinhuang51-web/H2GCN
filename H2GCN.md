https://arxiv.org/pdf/2006.11468
首先的，我们研究的是一种模型，这种模型我们希望他具有的特点是能够高效的（更多的信息与更少的噪声）建模消息的传递，所有边和点的设计都应该为在不同的情形下实现这个目标服务
研究的是什么：表征能力
节点也是异质的，边也是异质的情况下怎么建模呢？（半监督）
## 核心方法：
自节点与邻居嵌入分离：
更高阶的邻居：
中间表示的组合：

## 测试目标：
部分标签推断全部标签

## 前提：我们希望建模在没有同质性这一前提之下

## 方法说明：
1. 自我是与邻居分离的
每一个节点我们不过早与邻居混合
在h大于某一特点值鲁棒性更差
2. 我们设计了更高阶的节点而不是单纯的邻居
3. combine函数：
拼接，lstm注意力
结合不同的中间层的输出

GraphSAGE：先聚合邻居信息在融合自己
GCN：直接平滑融合

图谱理论：
低频-平滑
高频-剧烈
同质性比率越低，高频能量越高
我们要捕捉更多的高频成分，因为里面含有更多的信息

## 完整流程
权重变换+非线性激活得到特征编码
对邻居使用AGGR+combine函数
最后使用combine合并自己和每一轮邻居
拼接后线性变换加非线性变换

## 后续：https://www.jiongzhu.net/revisiting-heterophily-GNNs/

## 实验过程

### 1. README 数据下载问题

项目 README 里给了两套数据路线：

1. `experiments/h2gcn/scripts/*.sh`：原论文复现实验使用的 signac 格式数据。
2. `npz-datasets/scripts/*.sh`：作者后来补充的 `syn-cora` 和 `syn-products` 的 npz 简化格式。

我们先尝试了原始 signac 数据：

```bash
cd experiments/h2gcn
bash scripts/get-syn-cora.sh
bash scripts/get-syn-products.sh
bash scripts/get-real-cora_full.sh
```

但是 `gdown` 无法从 Google Drive 获取文件：

```text
Cannot retrieve the public link of the file.
```

浏览器打开链接也无法正常看到数据文件。结合 GitHub issue 中已有的数据链接失效反馈，可以判断：原始 signac 数据链接目前不稳定，甚至可能已经不再公开可访问。

所以当前不能宣称严格复现论文原始实验表格。

### 2. signac 数据和 npz 数据的区别

signac 格式数据是论文实验工程包，里面不只是图本身，还包括：

- 图结构
- 节点特征
- 标签
- 原始 train/val/test 划分
- feature 设置
- signac 实验状态信息

npz 数据是作者后来补充的简化数据包，主要包括：

- 邻接矩阵
- 节点特征
- 标签
- metadata

我们检查 `npz` 文件后发现它只有：

```text
adj_data, adj_indices, adj_indptr, adj_shape
attr_data, attr_indices, attr_indptr, attr_shape
labels, metadata
```

它没有：

```text
idx_train, idx_val, idx_test
```

所以 npz 版本不能保留论文原始划分。它适合做 method-level reproduction，也就是使用作者提供的 synthetic graph 重新固定随机种子划分数据，观察不同同质性下模型趋势；但不能直接对齐论文表格数字。

### 3. 成功下载 npz 数据

我们切换到官方补充的 npz 数据：

```bash
cd npz-datasets
bash scripts/get-syn-cora-npz.sh
bash scripts/get-syn-products-npz.sh
```

下载后得到：

```text
npz-datasets/syn-cora/h0.00-r1.npz ... h1.00-r3.npz
npz-datasets/syn-products/h0.00-r1.npz ... h1.00-r3.npz
```

其中 `h` 表示目标同质性，`r` 表示不同重复生成版本。

### 4. 数据结构检查脚本

我们写了：

```text
my code/feature-test.py
```

用于检查 npz 数据结构和异配/同配性质。这个脚本做了几件事：

1. 用 `npz-datasets/dataset.py` 中的 `CustomDataset` 加载数据。
2. 输出节点数、边数、特征维度、类别数量。
3. 输出 train/val/test 划分。
4. 计算实际边同质性 `actual edge homophily`。
5. 计算异配比例 `heterophily ratio`。
6. 比较同类边和异类边两端节点特征的平均余弦相似度。

运行命令：

```bash
/home/hmx42/miniconda3/envs/paper/bin/python "my code/feature-test.py"
```

代表性输出：

```text
dataset: syn-cora/h0.00-r1
nodes: 1490
edges: 2968
features: (1490, 1433), density: 0.012747
classes: 5, labels: [298 298 298 298 298]
split: train=100, val=500, test=890
actual edge homophily: 0.0000
heterophily ratio: 1.0000

dataset: syn-cora/h0.50-r1
actual edge homophily: 0.5115
heterophily ratio: 0.4885

dataset: syn-cora/h1.00-r1
actual edge homophily: 1.0000
heterophily ratio: 0.0000
```

结论：

- `h0.00` 是强异配图，边几乎都连不同类节点。
- `h0.50` 是混合图，实际同质性约为 0.51。
- `h1.00` 是强同配图，边几乎都连同类节点。
- `syn-cora` 是稀疏 bag-of-words 特征，特征密度约 `0.012747`。
- `syn-products` 特征更稠密，特征密度约 `0.972300`。

这一步验证了 synthetic dataset 的核心变量：同质性 `h` 确实控制了图结构中同类边比例。

### 5. 为原 H2GCN 训练入口新增 npz 数据入口

原项目训练入口是：

```text
h2gcn/run_experiments.py
```

原来只支持：

```text
planetoid
```

为了在尽量少改原代码的情况下使用 npz 数据，我们新增了：

```text
h2gcn/datasets/npz.py
```

这个文件实现了 `NPZData`，继承并复用 `PlanetoidData` 的接口，使原训练脚本可以继续调用：

- `row_normalize_features()`
- `adj_remove_eye()`
- `getTensors()`
- `num_labels`
- `train_mask / val_mask / test_mask`
- `y_train / y_val / y_test`

新增后，`run_experiments.py` 已经可以识别：

```text
{npz, planetoid}
```

也就是说可以使用类似命令：

```bash
cd h2gcn
python run_experiments.py H2GCN npz \
  --dataset h0.50-r1 \
  --dataset_path ../npz-datasets/syn-cora
```

由于 npz 不包含原始 split，我们在 `NPZData` 里用固定随机种子重新划分：

- 每类 `20` 个训练节点。
- `500` 个验证节点。
- 默认最多 `1000` 个测试节点。
- 剩余节点进入 wild mask。

对于 `syn-cora`：

```text
train=100, val=500, test=890
```

对于 `syn-products`：

```text
train=200, val=500, test=1000, wild=8300
```

### 6. 兼容性问题

这个仓库依赖较老，当前环境中遇到几个兼容问题：

1. `DeepRobust` 使用了已经被新版 NumPy 移除的 `np.int`。

解决方式是在测试脚本里临时加入：

```python
np.int = int
```

2. 原代码使用旧版 SciPy 导入路径：

```python
from scipy.sparse.linalg.eigen.arpack import eigsh
```

新版 SciPy 中这个路径不可用。我们在：

```text
h2gcn/datasets/_dataset.py
```

加了 fallback：

```python
try:
    from scipy.sparse.linalg.eigen.arpack import eigsh
except ModuleNotFoundError:
    from scipy.sparse.linalg import eigsh
```

3. 当前 `paper` 环境原来没有 TensorFlow。

H2GCN 主模型使用 TensorFlow 2 / Keras：

```python
class H2GCN(tf.keras.Model)
```

因此安装了：

```bash
pip install tensorflow-cpu==2.15.1
```

安装后确认：

```text
TensorFlow 2.15.1
```

### 7. 最小训练 smoke test

为了验证新增 npz 入口能接入原训练脚本，我们跑了一个很小的 H2GCN-1 smoke test：

```bash
cd h2gcn

/home/hmx42/miniconda3/envs/paper/bin/python run_experiments.py H2GCN npz \
  --dataset h0.50-r1 \
  --dataset_path ../npz-datasets/syn-cora \
  --network_setup M16-R-T1-G-V-C1-D0.5-MO \
  --epochs 2 \
  --run_id npz_smoke_test \
  --checkpoint_dir ../my\ code/checkpoints
```

这里使用较小 hidden size `M16` 和 `2` 个 epoch，只是验证训练链路，不看最终性能。

运行成功，输出：

```text
Epoch: 0001    Train Acc: 18.00%    Val Acc: 21.00%    Test Acc: 19.78%
Epoch: 0002    Train Acc: 20.00%    Val Acc: 21.00%    Test Acc: 20.22%
Best performance: Epoch 0002
```

checkpoint 保存到了：

```text
my code/checkpoints/H2GCN_h0.50-r1_0002_ta0.2022_va0.2100/
```

结论：原始 `h2gcn/run_experiments.py` 已经可以通过新增 `npz` 数据入口读取 `syn-cora` 的 npz 数据，并完成 H2GCN 训练链路。

### 8. 当前实验定位

当前工作不应描述为“严格复现论文完整结果”，而应描述为：

```text
method-level reproduction
```

原因：

- 原始 signac 数据无法稳定获取。
- npz 数据不包含论文原始 train/val/test split。
- 我们使用固定 seed 重新划分数据。

更准确的表述：

> 原始复现实验数据源失效后，我使用作者后来提供的 npz synthetic dataset，先验证同质性变量和数据结构，再为原 H2GCN 训练入口新增 npz loader，使模型能够在 npz 数据上跑通。这个实验不能直接对齐论文表格数值，但可以用于验证 H2GCN 在不同 homophily 水平下的趋势。

### 9. 下一步

接下来可以做三件事：

1. 批量统计所有 `h` 的 actual homophily，生成一张数据 sanity check 表。
2. 用相同 split 跑 MLP / H2GCN-1 / H2GCN-2，在 `syn-cora` 的不同 `h` 上比较趋势。
3. 如果要更接近论文，需要等待作者回复原始 signac 数据，或者重新构造与论文一致的 split 设置。
