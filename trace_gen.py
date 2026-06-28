# trace 生成：输出 (step, logical_id) 访问序列，供仿真器回放

from __future__ import annotations

import math
import random
from typing import Literal

from config import (
    BLOCK_SIZE,
    LOGICAL_BLOCK_STRIDE,
    MAX_REQUESTS,
    MIN_REQUESTS,
    MULTI_REQUEST_COUNT,
    PREFIX_RATIO,
    SEQ_LENGTH_JITTER,
    SPARSE_SKIP_PROB,
)


def encode_logical(req_id: int, block_id: int) -> int:
    return req_id * LOGICAL_BLOCK_STRIDE + block_id


def decode_logical(logical_id: int) -> tuple[int, int]:
    return logical_id // LOGICAL_BLOCK_STRIDE, logical_id % LOGICAL_BLOCK_STRIDE


def _prefix_threshold(max_blocks: int) -> int:
    return max(0, int(max_blocks * PREFIX_RATIO) - 1)


def generate_trace_deterministic(seq_length: int, block_size: int = BLOCK_SIZE):
    # 单请求理想 decode：每步读取当前全部逻辑块
    trace = []
    for step in range(1, seq_length + 1):
        num_blocks = math.ceil(step / block_size)
        for block_id in range(num_blocks):
            trace.append((step, block_id))
    return trace


def generate_trace(
    seq_length: int,
    block_size: int = BLOCK_SIZE,
    *,
    mode: Literal["deterministic", "stochastic"] = "stochastic",
    seed: int = 0,
    num_requests: int = MULTI_REQUEST_COUNT,
    seq_length_jitter: float = SEQ_LENGTH_JITTER,
    sparse_skip_prob: float = SPARSE_SKIP_PROB,
):
    if mode == "deterministic":
        if num_requests == 1:
            return generate_trace_deterministic(seq_length, block_size)
        return generate_multi_request_trace_deterministic(
            seq_length, num_requests=num_requests, block_size=block_size
        )
    return generate_stochastic_trace(
        seq_length,
        block_size=block_size,
        seed=seed,
        num_requests=num_requests,
        seq_length_jitter=seq_length_jitter,
        sparse_skip_prob=sparse_skip_prob,
    )


def generate_multi_request_trace_deterministic(
    seq_length_per_req: int,
    num_requests: int = MULTI_REQUEST_COUNT,
    block_size: int = BLOCK_SIZE,
):
    # 多请求轮流推进，无随机扰动
    local_steps = [0] * num_requests
    trace = []
    global_step = 0

    while any(s < seq_length_per_req for s in local_steps):
        for req_id in range(num_requests):
            if local_steps[req_id] >= seq_length_per_req:
                continue
            local_steps[req_id] += 1
            global_step += 1
            local_step = local_steps[req_id]
            num_blocks = math.ceil(local_step / block_size)
            for block_id in range(num_blocks):
                trace.append((global_step, encode_logical(req_id, block_id)))
    return trace


def _jittered_length(base: int, rng: random.Random, jitter: float) -> int:
    low = max(1, int(base * (1 - jitter)))
    high = max(low, int(base * (1 + jitter)))
    return rng.randint(low, high)


def generate_stochastic_trace(
    seq_length_per_req: int,
    *,
    block_size: int = BLOCK_SIZE,
    seed: int = 0,
    num_requests: int | None = None,
    seq_length_jitter: float = SEQ_LENGTH_JITTER,
    sparse_skip_prob: float = SPARSE_SKIP_PROB,
):
    # 多请求随机调度 + 长度扰动 + 非前缀块稀疏跳过（论文 §5.2）
    rng = random.Random(seed)
    if num_requests is None:
        num_requests = rng.randint(MIN_REQUESTS, MAX_REQUESTS)

    target_lengths = [
        _jittered_length(seq_length_per_req, rng, seq_length_jitter)
        for _ in range(num_requests)
    ]
    local_steps = [0] * num_requests
    trace: list[tuple[int, int]] = []
    global_step = 0
    max_blocks = max(1, LOGICAL_BLOCK_STRIDE // block_size)
    prefix_limit = _prefix_threshold(max_blocks)

    while any(local_steps[i] < target_lengths[i] for i in range(num_requests)):
        active = [i for i in range(num_requests) if local_steps[i] < target_lengths[i]]
        req_id = rng.choice(active)
        local_steps[req_id] += 1
        global_step += 1
        local_step = local_steps[req_id]
        num_blocks = math.ceil(local_step / block_size)
        block_id = num_blocks - 1
        logical_new = encode_logical(req_id, block_id)
        trace.append((global_step, logical_new))

        for bid in range(num_blocks - 1):
            if bid <= prefix_limit:
                trace.append((global_step, encode_logical(req_id, bid)))
            elif rng.random() >= sparse_skip_prob:
                trace.append((global_step, encode_logical(req_id, bid)))

    return trace


generate_multi_request_trace = generate_multi_request_trace_deterministic


if __name__ == "__main__":
    det = generate_trace(64, mode="deterministic", num_requests=2)
    sto = generate_trace(64, mode="stochastic", seed=42, num_requests=2)
    print(f"deterministic accesses={len(det)}")
    print(f"stochastic   accesses={len(sto)}")
