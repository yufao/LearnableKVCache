# 替换策略：FIFO、LRU、Learned-RD（含可选预取）

from collections import deque

from learned_cache import ReuseDistancePredictor, make_features


class FIFO:
    def __init__(self, num_physical_blocks: int):
        self.num_physical_blocks = num_physical_blocks
        self.deque = deque()

    def on_hit(self, phys_id: int, step: int):
        pass

    def on_alloc(self, phys_id: int, logical_id: int, step: int):
        self.deque.append(phys_id)

    def on_evict(self, phys_id: int):
        try:
            self.deque.remove(phys_id)
        except ValueError:
            pass

    def select_victim(self, valid, phys2logical, step: int):
        while self.deque:
            pid = self.deque[0]
            if valid[pid]:
                return pid
            self.deque.popleft()
        for i, v in enumerate(valid):
            if v:
                return i
        raise RuntimeError("No valid physical block found")

    def metadata_bytes(self) -> int:
        return 2 * self.num_physical_blocks


class LRU:
    def __init__(self, num_physical_blocks: int):
        self.num_physical_blocks = num_physical_blocks
        self.last_access_step = [-1] * num_physical_blocks

    def on_hit(self, phys_id: int, step: int):
        self.last_access_step[phys_id] = step

    def on_alloc(self, phys_id: int, logical_id: int, step: int):
        self.last_access_step[phys_id] = step

    def on_evict(self, phys_id: int):
        self.last_access_step[phys_id] = -1

    def select_victim(self, valid, phys2logical, step: int) -> int:
        best_pid = None
        best_time = step + 1
        for pid, v in enumerate(valid):
            if v and self.last_access_step[pid] < best_time:
                best_pid = pid
                best_time = self.last_access_step[pid]
        if best_pid is None:
            raise RuntimeError("No valid physical block found")
        return best_pid

    def metadata_bytes(self) -> int:
        return 5 * self.num_physical_blocks


class LearnedRD:
    # 预测重用距离，驱逐 RD 估计值最大的块

    def __init__(
        self,
        num_physical_blocks: int,
        predictor: ReuseDistancePredictor,
        prefetch: bool = False,
        prefetch_threshold: float = 2.0,
    ):
        self.num_physical_blocks = num_physical_blocks
        self.predictor = predictor
        self.prefetch = prefetch
        self.prefetch_threshold = prefetch_threshold
        self.last_access_step = [-1] * num_physical_blocks
        self.access_count = [0] * num_physical_blocks
        self.prefetch_candidates: set[int] = set()

    def on_hit(self, phys_id: int, step: int):
        self.last_access_step[phys_id] = step
        self.access_count[phys_id] += 1

    def on_alloc(self, phys_id: int, logical_id: int, step: int):
        self.last_access_step[phys_id] = step
        self.access_count[phys_id] = max(1, self.access_count[phys_id])

    def on_evict(self, phys_id: int):
        self.last_access_step[phys_id] = -1
        self.access_count[phys_id] = 0

    def _features_for(self, phys_id: int, logical_id: int, step: int) -> list[float]:
        la = self.last_access_step[phys_id]
        ac = self.access_count[phys_id]
        return make_features(logical_id, step, la, ac)

    def predict_rd(self, phys_id: int, logical_id: int, step: int) -> float:
        return self.predictor.predict_one(self._features_for(phys_id, logical_id, step))

    def predict_rd_for_logical(self, logical_id: int, step: int, last_access: int, access_count: int) -> float:
        return self.predictor.predict_one(make_features(logical_id, step, last_access, access_count))

    def select_victim(self, valid, phys2logical, step: int) -> int:
        best_pid = None
        best_rd = -1.0
        for pid, v in enumerate(valid):
            if not v:
                continue
            logical_id = phys2logical[pid]
            if logical_id is None:
                continue
            rd = self.predict_rd(pid, logical_id, step)
            if rd > best_rd:
                best_rd = rd
                best_pid = pid
        if best_pid is None:
            raise RuntimeError("No valid physical block found")
        return best_pid

    def metadata_bytes(self) -> int:
        n_trees = getattr(self.predictor, "n_trees", 1)
        return 8 * self.num_physical_blocks + 512 * n_trees
