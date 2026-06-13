from pathlib import Path
import re
import sys

import numpy as np

# DeepRobust 0.2.1 里还在使用 np.int。
# 新版 NumPy 已经删除了 np.int，所以这里临时把它指回 Python 内置 int。
np.int = int


# 当前文件在 "my code/" 下，parents[1] 就是项目根目录 H2GCN。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
NPZ_DATASET_DIR = PROJECT_ROOT / "npz-datasets"

# 让 Python 可以 import npz-datasets/dataset.py 里的 CustomDataset。
sys.path.insert(0, str(NPZ_DATASET_DIR))

from dataset import CustomDataset  # noqa: E402


def parse_target_h(dataset_name):
    """从文件名中解析作者设定的目标同质性 h，例如 h0.50-r1 -> 0.50。"""
    match = re.match(r"h([0-9.]+)-r\d+", dataset_name)
    return float(match.group(1)) if match else None


def edge_index(adj):
    """把稀疏邻接矩阵转成无向边列表，只保留 row < col，避免一条边算两次。"""
    row, col = adj.nonzero()
    undirected = row < col
    return row[undirected], col[undirected]


def feature_matrix_to_dense(features):
    """把 scipy sparse 特征矩阵转成普通 numpy array，方便后面算余弦相似度。"""
    if hasattr(features, "toarray"):
        return features.toarray()
    return np.asarray(features)


def row_normalize(features):
    """对每个节点的特征向量做 L2 归一化，用于计算节点之间的余弦相似度。"""
    norm = np.linalg.norm(features, axis=1, keepdims=True)
    norm[norm == 0] = 1
    return features / norm


def safe_mean(values):
    """有些图 h=0 时没有同类边，h=1 时没有异类边；空数组就返回 None。"""
    return float(values.mean()) if len(values) else None


def format_optional(value):
    """把 None 显示成 N/A，其它数值保留 4 位小数。"""
    return "N/A" if value is None else f"{value:.4f}"


def inspect_dataset(root, name, seed=15):
    """读取一个 npz 数据集，并打印图结构、特征、标签和异配/同配统计。"""
    dataset = CustomDataset(root=str(root), name=name, setting="gcn", seed=seed)

    adj = dataset.adj
    features = feature_matrix_to_dense(dataset.features)
    labels = dataset.labels
    row, col = edge_index(adj)

    # same_class[i] 表示第 i 条边两端节点是否属于同一类别。
    same_class = labels[row] == labels[col]

    # 边同质性：同类边数量 / 总边数量。
    # 这是 H2GCN 论文里最核心的数据性质之一。
    homophily = same_class.mean()

    # 下面不是论文的主要指标，只是帮助理解“结构是否同类”和“特征是否相似”。
    # 如果两端节点特征越相似，余弦值越大。
    normalized_features = row_normalize(features)
    edge_cosine = np.sum(normalized_features[row] * normalized_features[col], axis=1)
    same_edge_cosine = safe_mean(edge_cosine[same_class])
    diff_edge_cosine = safe_mean(edge_cosine[~same_class])

    print("=" * 80)
    print(f"dataset: {root.name}/{name}")
    print(f"target h from filename: {parse_target_h(name)}")
    print(f"nodes: {adj.shape[0]}")
    print(f"edges: {len(row)}")

    # density 表示特征矩阵非零元素比例。
    # syn-cora 是稀疏 bag-of-words 特征，syn-products 是更密集的连续特征。
    print(f"features: {features.shape}, density: {(features != 0).mean():.6f}")
    print(f"classes: {len(np.unique(labels))}, labels: {np.bincount(labels)}")
    print(
        "split: "
        f"train={len(dataset.idx_train)}, "
        f"val={len(dataset.idx_val)}, "
        f"test={len(dataset.idx_test)}"
    )
    print(f"actual edge homophily: {homophily:.4f}")
    print(f"heterophily ratio: {1 - homophily:.4f}")
    print(f"same-class edge feature cosine: {format_optional(same_edge_cosine)}")
    print(f"diff-class edge feature cosine: {format_optional(diff_edge_cosine)}")


def main():
    syn_cora = NPZ_DATASET_DIR / "syn-cora"
    syn_products = NPZ_DATASET_DIR / "syn-products"

    # 对 syn-cora 选三个代表点：
    # h=0.00 代表强异配，h=0.50 代表混合，h=1.00 代表强同配。
    for name in ["h0.00-r1", "h0.50-r1", "h1.00-r1"]:
        inspect_dataset(syn_cora, name)

    # 再看一个 syn-products，确认另一个 synthetic 数据集也能正常读取。
    inspect_dataset(syn_products, "h0.00-r1")


if __name__ == "__main__":
    main()
