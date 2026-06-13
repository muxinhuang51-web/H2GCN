from . import *
from ._dataset import PlanetoidData, sp


def add_subparser_args(parser):
    subparser = parser.add_argument_group("NPZ Format Data Arguments (datasets/npz.py)")
    subparser.add_argument("--dataset", type=str, required=True,
                           help="Dataset file name without .npz, e.g. h0.50-r1")
    subparser.add_argument("--dataset_path", type=str, dest="_dataset_path", required=True,
                           help="Folder containing npz files, e.g. ../npz-datasets/syn-cora")
    subparser.add_argument("--split_seed", type=int, default=15)
    subparser.add_argument("--train_per_class", type=int, default=20)
    subparser.add_argument("--val_size", type=int, default=500)
    subparser.add_argument("--test_size", type=int, default=1000,
                           help="Use all remaining samples if fewer than this value")
    parser.function_hooks["argparse"].appendleft(argparse_callback)


def argparse_callback(args):
    dataset = NPZData(
        args.dataset,
        args._dataset_path,
        split_seed=args.split_seed,
        train_per_class=args.train_per_class,
        val_size=args.val_size,
        test_size=args.test_size,
    )
    args.objects["dataset"] = dataset
    print(f"===> NPZ dataset loaded: {args.dataset}")


class NPZData(PlanetoidData):
    """NPZ dataset adapter for the synthetic H2GCN datasets.

    This keeps the same interface as PlanetoidData so existing H2GCN training
    code can reuse row_normalize_features(), adj_remove_eye(), and getTensors().
    """

    def __init__(self, dataset_str, dataset_path, split_seed=15,
                 train_per_class=20, val_size=500, test_size=1000):
        self._sparse_data = dict()
        self._dense_data = dict()
        self.dataset_str = dataset_str
        self.dataset_path = dataset_path
        self.split_seed = split_seed
        self.train_per_class = train_per_class
        self.val_size = val_size
        self.test_size = test_size
        self.load_data(dataset_str, dataset_path)
        self._original_data = (self._sparse_data.copy(),
                               self._dense_data.copy())

    @staticmethod
    def _load_npz_graph(file_path):
        with np.load(file_path, allow_pickle=True) as loader:
            adj = sp.csr_matrix(
                (loader["adj_data"], loader["adj_indices"], loader["adj_indptr"]),
                shape=loader["adj_shape"],
            )
            features = sp.csr_matrix(
                (loader["attr_data"], loader["attr_indices"], loader["attr_indptr"]),
                shape=loader["attr_shape"],
            )
            labels = loader["labels"].astype(np.int32)
        return adj, features, labels

    @staticmethod
    def _sample_mask(idx, length):
        mask = np.zeros(length, dtype=bool)
        mask[idx] = True
        return mask

    def _split_indices(self, labels):
        rng = np.random.RandomState(self.split_seed)
        all_indices = np.arange(labels.shape[0])

        train_indices = []
        for label in np.unique(labels):
            label_indices = np.where(labels == label)[0]
            rng.shuffle(label_indices)
            take = min(self.train_per_class, len(label_indices))
            train_indices.extend(label_indices[:take])

        train_indices = np.array(train_indices, dtype=np.int64)
        train_set = set(train_indices.tolist())
        remaining = np.array([i for i in all_indices if i not in train_set], dtype=np.int64)
        rng.shuffle(remaining)

        val_size = min(self.val_size, len(remaining))
        val_indices = remaining[:val_size]
        remaining = remaining[val_size:]

        test_size = len(remaining) if self.test_size is None else min(self.test_size, len(remaining))
        test_indices = remaining[:test_size]
        wild_indices = remaining[test_size:]

        return train_indices, val_indices, test_indices, wild_indices

    def load_data(self, dataset_str, dataset_path):
        file_path = os.path.join(dataset_path, dataset_str + ".npz")
        if not os.path.exists(file_path):
            raise FileNotFoundError(file_path)

        adj, features, labels = self._load_npz_graph(file_path)

        # Match npz-datasets/dataset.py: make graph undirected, unweighted, and loop-free.
        adj = adj + adj.T
        adj = adj.tolil()
        adj[adj > 1] = 1
        adj.setdiag(0)
        adj = adj.astype(np.float32).tocsr()
        adj.eliminate_zeros()

        features = features.astype(np.float32).tocsr()

        y_all = np.zeros((labels.shape[0], labels.max() + 1), dtype=np.float32)
        y_all[np.arange(labels.shape[0]), labels] = 1

        idx_train, idx_val, idx_test, idx_wild = self._split_indices(labels)
        train_mask = self._sample_mask(idx_train, labels.shape[0])
        val_mask = self._sample_mask(idx_val, labels.shape[0])
        test_mask = self._sample_mask(idx_test, labels.shape[0])
        wild_mask = self._sample_mask(idx_wild, labels.shape[0])

        y_train = np.zeros_like(y_all)
        y_val = np.zeros_like(y_all)
        y_test = np.zeros_like(y_all)
        y_wild = np.zeros_like(y_all)
        y_train[train_mask, :] = y_all[train_mask, :]
        y_val[val_mask, :] = y_all[val_mask, :]
        y_test[test_mask, :] = y_all[test_mask, :]
        y_wild[wild_mask, :] = y_all[wild_mask, :]

        self._sparse_data["sparse_adj"] = adj
        self._sparse_data["features"] = features
        self._dense_data["y_all"] = y_all
        self._dense_data["train_mask"] = train_mask
        self._dense_data["val_mask"] = val_mask
        self._dense_data["test_mask"] = test_mask
        self._dense_data["wild_mask"] = wild_mask
        self._dense_data["y_train"] = y_train
        self._dense_data["y_val"] = y_val
        self._dense_data["y_test"] = y_test
        self._dense_data["y_wild"] = y_wild
        self._PlanetoidData__preprocessedAdj = None
        self._PlanetoidData__preprocessedFeature = None
