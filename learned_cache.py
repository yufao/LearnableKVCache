# 离线训练：从 trace 提取特征与 RD 标签，拟合回归决策树 / 随机森林

from __future__ import annotations

import random
from dataclasses import dataclass

from config import (
    BLOCK_SIZE,
    FOREST_FEATURE_FRACTION,
    FOREST_N_TREES,
    LOGICAL_BLOCK_STRIDE,
    PREFIX_RATIO,
    TRAIN_SEQ_LENGTH,
    TREE_MAX_DEPTH,
)
from trace_gen import decode_logical, generate_trace


@dataclass
class TreeNode:
    feature_idx: int = -1
    threshold: float = 0.0
    left: TreeNode | None = None
    right: TreeNode | None = None
    value: float = 0.0

    @property
    def is_leaf(self) -> bool:
        return self.feature_idx < 0


FEATURE_NAMES = ["age", "block_pos", "req_id", "access_count", "is_prefix"]


def build_next_access_index(trace: list[tuple[int, int]]) -> dict[int, list[int]]:
    index: dict[int, list[int]] = {}
    for step, logical_id in trace:
        index.setdefault(logical_id, []).append(step)
    return index


def future_reuse_distance(logical_id: int, step: int, index: dict[int, list[int]]) -> float:
    steps = index.get(logical_id, [])
    for s in steps:
        if s > step:
            return float(s - step)
    return float(TRAIN_SEQ_LENGTH * 4)


def prefix_flag(block_id: int) -> float:
    max_blocks = max(1, LOGICAL_BLOCK_STRIDE // BLOCK_SIZE)
    threshold = max(0, int(max_blocks * PREFIX_RATIO) - 1)
    return 1.0 if block_id <= threshold else 0.0


def make_features(
    logical_id: int,
    step: int,
    last_access: int,
    access_count: int,
) -> list[float]:
    req_id, block_id = decode_logical(logical_id)
    age = float(step - last_access) if last_access >= 0 else float(step)
    return [age, float(block_id), float(req_id), float(access_count), prefix_flag(block_id)]


def collect_samples_from_trace(trace: list[tuple[int, int]]) -> tuple[list[list[float]], list[float]]:
    index = build_next_access_index(trace)
    last_access: dict[int, int] = {}
    access_count: dict[int, int] = {}
    xs: list[list[float]] = []
    ys: list[float] = []

    for step, logical_id in trace:
        la = last_access.get(logical_id, -1)
        ac = access_count.get(logical_id, 0)
        xs.append(make_features(logical_id, step, la, ac))
        ys.append(future_reuse_distance(logical_id, step, index))
        last_access[logical_id] = step
        access_count[logical_id] = ac + 1
    return xs, ys


def _variance(values: list[float]) -> float:
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    return sum((v - mean) ** 2 for v in values) / len(values)


def _best_split(
    xs: list[list[float]],
    ys: list[float],
    feature_indices: list[int] | None = None,
) -> tuple[int, float, float]:
    n = len(ys)
    if n < 4:
        return -1, 0.0, _variance(ys)

    if feature_indices is None:
        feature_indices = list(range(len(FEATURE_NAMES)))

    base_var = _variance(ys)
    best_gain = 0.0
    best_feat = -1
    best_thr = 0.0

    for feat in feature_indices:
        values = sorted({row[feat] for row in xs})
        if len(values) < 2:
            continue
        candidates = [(values[i] + values[i + 1]) / 2 for i in range(len(values) - 1)]
        for thr in candidates:
            left_y = [ys[i] for i in range(n) if xs[i][feat] <= thr]
            right_y = [ys[i] for i in range(n) if xs[i][feat] > thr]
            if not left_y or not right_y:
                continue
            gain = base_var - (
                len(left_y) * _variance(left_y) + len(right_y) * _variance(right_y)
            ) / n
            if gain > best_gain:
                best_gain = gain
                best_feat = feat
                best_thr = thr

    if best_feat < 0:
        return -1, 0.0, sum(ys) / len(ys)
    return best_feat, best_thr, best_gain


def _build_tree(
    xs: list[list[float]],
    ys: list[float],
    depth: int,
    max_depth: int,
    feature_indices: list[int] | None = None,
) -> TreeNode:
    if depth >= max_depth or len(ys) < 6:
        return TreeNode(value=sum(ys) / len(ys))

    feat, thr, gain = _best_split(xs, ys, feature_indices)
    if feat < 0 or gain <= 1e-9:
        return TreeNode(value=sum(ys) / len(ys))

    left_x, left_y, right_x, right_y = [], [], [], []
    for row, target in zip(xs, ys):
        if row[feat] <= thr:
            left_x.append(row)
            left_y.append(target)
        else:
            right_x.append(row)
            right_y.append(target)

    if not left_y or not right_y:
        return TreeNode(value=sum(ys) / len(ys))

    return TreeNode(
        feature_idx=feat,
        threshold=thr,
        left=_build_tree(left_x, left_y, depth + 1, max_depth, feature_indices),
        right=_build_tree(right_x, right_y, depth + 1, max_depth, feature_indices),
        value=sum(ys) / len(ys),
    )


class ReuseDistancePredictor:
    def __init__(self, root: TreeNode | None = None):
        self.root = root

    def predict_one(self, features: list[float]) -> float:
        node = self.root
        while node and not node.is_leaf:
            if features[node.feature_idx] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.value if node else float(TRAIN_SEQ_LENGTH)

    def predict_batch(self, rows: list[list[float]]) -> list[float]:
        return [self.predict_one(r) for r in rows]


class RandomForestPredictor:
    def __init__(self, trees: list[ReuseDistancePredictor]):
        self.trees = trees

    def predict_one(self, features: list[float]) -> float:
        if not self.trees:
            return float(TRAIN_SEQ_LENGTH)
        return sum(tree.predict_one(features) for tree in self.trees) / len(self.trees)

    def predict_batch(self, rows: list[list[float]]) -> list[float]:
        return [self.predict_one(r) for r in rows]

    @property
    def n_trees(self) -> int:
        return len(self.trees)


def _load_training_data(train_seeds: list[int], seq_length: int) -> tuple[list[list[float]], list[float]]:
    all_x: list[list[float]] = []
    all_y: list[float] = []
    for seed in train_seeds:
        trace = generate_trace(seq_length, mode="stochastic", seed=seed)
        xs, ys = collect_samples_from_trace(trace)
        all_x.extend(xs)
        all_y.extend(ys)
    return all_x, all_y


def _sample_feature_indices(rng: random.Random) -> list[int]:
    n_features = len(FEATURE_NAMES)
    k = max(1, int(n_features * FOREST_FEATURE_FRACTION))
    return sorted(rng.sample(range(n_features), k))


def train_predictor(train_seeds: list[int], seq_length: int = TRAIN_SEQ_LENGTH) -> ReuseDistancePredictor:
    all_x, all_y = _load_training_data(train_seeds, seq_length)
    root = _build_tree(all_x, all_y, depth=0, max_depth=TREE_MAX_DEPTH)
    return ReuseDistancePredictor(root)


def train_forest_predictor(
    train_seeds: list[int],
    seq_length: int = TRAIN_SEQ_LENGTH,
    n_trees: int = FOREST_N_TREES,
) -> RandomForestPredictor:
    all_x, all_y = _load_training_data(train_seeds, seq_length)
    n = len(all_y)
    trees: list[ReuseDistancePredictor] = []

    for tree_id in range(n_trees):
        rng = random.Random(1000 + tree_id)
        indices = [rng.randrange(n) for _ in range(n)]
        boot_x = [all_x[i] for i in indices]
        boot_y = [all_y[i] for i in indices]
        feat_idx = _sample_feature_indices(rng)
        root = _build_tree(boot_x, boot_y, depth=0, max_depth=TREE_MAX_DEPTH, feature_indices=feat_idx)
        trees.append(ReuseDistancePredictor(root))

    return RandomForestPredictor(trees)


def describe_tree(node: TreeNode | None, indent: int = 0) -> str:
    if node is None or node.is_leaf:
        val = 0.0 if node is None else node.value
        return " " * indent + f"leaf -> {val:.2f}\n"
    name = FEATURE_NAMES[node.feature_idx]
    text = " " * indent + f"if {name} <= {node.threshold:.2f}:\n"
    text += describe_tree(node.left, indent + 2)
    text += " " * indent + "else:\n"
    text += describe_tree(node.right, indent + 2)
    return text
