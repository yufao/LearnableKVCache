# 批量实验：多 seed 对比 FIFO / LRU / Learned-RD / Learned-RF，输出 CSV 与图

from __future__ import annotations

import csv
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

from config import (
    BLOCK_SIZE,
    MULTI_SEQ_LENGTHS,
    POOL_SIZE,
    PREFETCH_THRESHOLD,
    RESULTS_CSV_DIR,
    RESULTS_FIG_DIR,
    TEST_SEEDS,
    TRAIN_SEEDS,
)
from kv_cache_sim import run_simulation
from learned_cache import train_forest_predictor, train_predictor
from policies import FIFO, LRU, LearnedRD
from trace_gen import generate_trace

ROOT = Path(__file__).resolve().parent

POLICY_SPECS = [
    ("fifo", "FIFO", lambda: FIFO(POOL_SIZE)),
    ("lru", "LRU", lambda: LRU(POOL_SIZE)),
]


def ensure_dirs() -> tuple[Path, Path]:
    csv_dir = ROOT / RESULTS_CSV_DIR
    fig_dir = ROOT / RESULTS_FIG_DIR
    csv_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    return csv_dir, fig_dir


def run_one(policy, trace) -> dict:
    stats = run_simulation(POOL_SIZE, policy, trace)
    return {
        "hits": stats.hits,
        "misses": stats.misses,
        "evictions": stats.evictions,
        "accesses": stats.accesses,
        "hit_rate": round(stats.hit_rate, 6),
        "prefetch_issued": stats.prefetch_issued,
        "prefetch_hits": stats.prefetch_hits,
        "prefetch_waste": stats.prefetch_waste,
    }


def run_multi_seed_experiments(csv_dir: Path) -> tuple[list[dict], list[dict]]:
    dt_predictor = train_predictor(TRAIN_SEEDS)
    rf_predictor = train_forest_predictor(TRAIN_SEEDS)

    learned_specs = [
        ("learned_rd", "Learned-RD (DT)", dt_predictor, False),
        ("learned_rf", "Learned-RF", rf_predictor, False),
        ("learned_rd_prefetch", "Learned-RD+Prefetch", dt_predictor, True),
    ]

    per_seed_rows: list[dict] = []

    for seq_len in MULTI_SEQ_LENGTHS:
        for seed in TEST_SEEDS:
            trace = generate_trace(seq_len, mode="stochastic", seed=seed)
            for key, label, factory in POLICY_SPECS:
                row = run_one(factory(), trace)
                per_seed_rows.append(
                    {
                        "policy": key,
                        "policy_label": label,
                        "seq_length": seq_len,
                        "seed": seed,
                        **row,
                    }
                )
            for key, label, predictor, prefetch in learned_specs:
                policy = LearnedRD(
                    POOL_SIZE,
                    predictor,
                    prefetch=prefetch,
                    prefetch_threshold=PREFETCH_THRESHOLD,
                )
                row = run_one(policy, trace)
                per_seed_rows.append(
                    {
                        "policy": key,
                        "policy_label": label,
                        "seq_length": seq_len,
                        "seed": seed,
                        **row,
                    }
                )

    per_seed_path = csv_dir / "exp_learned_multi_per_seed.csv"
    fieldnames = [
        "policy",
        "policy_label",
        "seq_length",
        "seed",
        "hits",
        "misses",
        "evictions",
        "accesses",
        "hit_rate",
        "prefetch_issued",
        "prefetch_hits",
        "prefetch_waste",
    ]
    with per_seed_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(per_seed_rows)
    print(f"Wrote {per_seed_path}")

    summary_rows: list[dict] = []
    grouped: dict[tuple[str, str, int], list[float]] = defaultdict(list)
    for row in per_seed_rows:
        grouped[(row["policy"], row["policy_label"], row["seq_length"])].append(row["hit_rate"])

    for (policy, label, seq_len), rates in sorted(grouped.items()):
        summary_rows.append(
            {
                "policy": policy,
                "policy_label": label,
                "seq_length": seq_len,
                "n_seeds": len(rates),
                "hit_rate_mean": round(statistics.mean(rates), 6),
                "hit_rate_std": round(statistics.pstdev(rates), 6) if len(rates) > 1 else 0.0,
                "hit_rate_min": round(min(rates), 6),
                "hit_rate_max": round(max(rates), 6),
            }
        )

    summary_path = csv_dir / "exp_learned_multi_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "policy",
                "policy_label",
                "seq_length",
                "n_seeds",
                "hit_rate_mean",
                "hit_rate_std",
                "hit_rate_min",
                "hit_rate_max",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"Wrote {summary_path}")

    return per_seed_rows, summary_rows


def _plot_policy_band(summary_rows: list[dict], policies: list[str], style: dict, title: str, out: Path) -> None:
    plt.figure(figsize=(9, 5.5))
    for policy in policies:
        subset = sorted(
            [r for r in summary_rows if r["policy"] == policy],
            key=lambda r: r["seq_length"],
        )
        if not subset:
            continue
        xs = [r["seq_length"] for r in subset]
        means = [r["hit_rate_mean"] for r in subset]
        stds = [r["hit_rate_std"] for r in subset]
        meta = style[policy]
        label = subset[0]["policy_label"]
        n = subset[0]["n_seeds"]
        color = meta["color"]
        plt.plot(xs, means, marker=meta["marker"], color=color, label=f"{label} (n={n})")
        lower = [m - s for m, s in zip(means, stds)]
        upper = [m + s for m, s in zip(means, stds)]
        plt.fill_between(xs, lower, upper, color=color, alpha=0.2)

    plt.xlabel("Sequence Length per Request (tokens)")
    plt.ylabel("Hit Rate")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Wrote {out}")


def plot_bands(summary_rows: list[dict], fig_dir: Path) -> None:
    main_style = {
        "fifo": {"color": "#1f77b4", "marker": "o"},
        "lru": {"color": "#ff7f0e", "marker": "s"},
        "learned_rd": {"color": "#2ca02c", "marker": "^"},
        "learned_rf": {"color": "#9467bd", "marker": "v"},
        "learned_rd_prefetch": {"color": "#d62728", "marker": "D"},
    }
    _plot_policy_band(
        summary_rows,
        ["fifo", "lru", "learned_rd", "learned_rf", "learned_rd_prefetch"],
        main_style,
        f"Learned KV Cache vs Baselines (stochastic trace, N={POOL_SIZE}, B={BLOCK_SIZE})",
        fig_dir / "exp_learned_hit_rate_band.png",
    )

    ablation_style = {
        "learned_rd": {"color": "#2ca02c", "marker": "^"},
        "learned_rf": {"color": "#9467bd", "marker": "v"},
        "lru": {"color": "#ff7f0e", "marker": "s"},
    }
    _plot_policy_band(
        summary_rows,
        ["lru", "learned_rd", "learned_rf"],
        ablation_style,
        f"Decision Tree vs Random Forest (N={POOL_SIZE}, B={BLOCK_SIZE})",
        fig_dir / "exp_dt_vs_rf_band.png",
    )


def print_summary(summary_rows: list[dict]) -> None:
    print("\n=== Summary @ T=1024 (mean ± std over test seeds) ===")
    for policy in ["fifo", "lru", "learned_rd", "learned_rf", "learned_rd_prefetch"]:
        row = next(r for r in summary_rows if r["policy"] == policy and r["seq_length"] == 1024)
        print(
            f"{row['policy_label']:22s}  "
            f"{row['hit_rate_mean']:.4f} ± {row['hit_rate_std']:.4f}"
        )


def main() -> None:
    csv_dir, fig_dir = ensure_dirs()
    _, summary = run_multi_seed_experiments(csv_dir)
    plot_bands(summary, fig_dir)
    print_summary(summary)
    print("\nDone.")


if __name__ == "__main__":
    main()
