# KV Cache 块管理仿真器：块表 + 命中/缺失/驱逐统计（不存 K/V 张量）

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import groupby

from config import PREFETCH_HORIZON


@dataclass
class SimStats:
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    accesses: int = 0
    prefetch_issued: int = 0
    prefetch_hits: int = 0
    prefetch_waste: int = 0

    @property
    def hit_rate(self) -> float:
        return self.hits / self.accesses if self.accesses else 0.0


class KVCacheSimulator:
    def __init__(self, num_physical_blocks: int, policy):
        self.n = num_physical_blocks
        self.policy = policy
        # 块表：logical_id -> physical_id
        self.logical_to_phys: dict[int, int] = {}
        self.phys2logical: list[int | None] = [None] * num_physical_blocks
        self.valid: list[bool] = [False] * num_physical_blocks
        self.stats = SimStats()
        # 逻辑块级访问历史（预取与特征用）
        self._logical_last_access: dict[int, int] = {}
        self._logical_access_count: dict[int, int] = {}

    def _find_free_slot(self) -> int | None:
        for i, v in enumerate(self.valid):
            if not v:
                return i
        return None

    def _allocate(self, logical_id: int, phys_id: int, step: int) -> None:
        self.logical_to_phys[logical_id] = phys_id
        self.phys2logical[phys_id] = logical_id
        self.valid[phys_id] = True
        self.policy.on_alloc(phys_id, logical_id, step)

    def _evict(self, phys_id: int) -> None:
        logical_id = self.phys2logical[phys_id]
        if logical_id is not None:
            del self.logical_to_phys[logical_id]
        self.phys2logical[phys_id] = None
        self.valid[phys_id] = False
        self.policy.on_evict(phys_id)
        self.stats.evictions += 1

    def _touch_logical(self, logical_id: int, step: int) -> None:
        la = self._logical_last_access.get(logical_id, -1)
        ac = self._logical_access_count.get(logical_id, 0)
        self._logical_last_access[logical_id] = step
        self._logical_access_count[logical_id] = ac + 1

    def access(self, logical_id: int, step: int) -> None:
        self.stats.accesses += 1
        self._touch_logical(logical_id, step)

        if logical_id in self.logical_to_phys:
            phys_id = self.logical_to_phys[logical_id]
            self.stats.hits += 1
            if hasattr(self.policy, "on_hit"):
                self.policy.on_hit(phys_id, step)
            return

        self.stats.misses += 1
        free = self._find_free_slot()
        if free is not None:
            self._allocate(logical_id, free, step)
        else:
            victim = self.policy.select_victim(self.valid, self.phys2logical, step)
            self._evict(victim)
            self._allocate(logical_id, victim, step)

    def _maybe_prefetch(self, step: int, upcoming: list[int]) -> None:
        policy = self.policy
        if not getattr(policy, "prefetch", False):
            return

        seen: set[int] = set()
        for logical_id in upcoming:
            if logical_id in seen:
                continue
            seen.add(logical_id)
            if logical_id in self.logical_to_phys:
                continue

            la = self._logical_last_access.get(logical_id, -1)
            ac = self._logical_access_count.get(logical_id, 0)
            rd = policy.predict_rd_for_logical(logical_id, step, la, ac)
            if rd > policy.prefetch_threshold:
                continue

            self.stats.prefetch_issued += 1
            free = self._find_free_slot()
            if free is not None:
                self._allocate(logical_id, free, step)
                self.stats.prefetch_hits += 1
            else:
                self.stats.prefetch_waste += 1

    def run_trace(self, trace: list[tuple[int, int]]) -> SimStats:
        if not trace:
            return self.stats

        trace_by_step: list[tuple[int, list[int]]] = []
        for step, group in groupby(trace, key=lambda x: x[0]):
            logical_ids = [lid for _, lid in group]
            trace_by_step.append((step, logical_ids))

        step_list = [s for s, _ in trace_by_step]

        for i, (step, logical_ids) in enumerate(trace_by_step):
            for logical_id in logical_ids:
                self.access(logical_id, step)

            if getattr(self.policy, "prefetch", False):
                upcoming: list[int] = []
                for j in range(i + 1, min(i + 1 + PREFETCH_HORIZON, len(trace_by_step))):
                    upcoming.extend(trace_by_step[j][1])
                self._maybe_prefetch(step, upcoming)

        return self.stats


def run_simulation(num_physical_blocks: int, policy, trace: list[tuple[int, int]]) -> SimStats:
    sim = KVCacheSimulator(num_physical_blocks, policy)
    return sim.run_trace(trace)
