# KV Cache 块管理仿真原型

课程论文《学习型缓存在 LLM KV Cache 推理中的适用性研究》配套代码。  
trace 驱动，不接入真实大模型，用于对比 FIFO / LRU / Learned-RD / Learned-RF 的块级命中率。


## 目录结构

```
kv-cache-sim/
├── config.py           # 实验参数（B、N、seed 划分等）
├── trace_gen.py        # 随机 / 确定性 trace 生成
├── learned_cache.py    # 特征、RD 标签、决策树与随机森林训练
├── policies.py         # FIFO、LRU、Learned-RD
├── kv_cache_sim.py     # 块表、access、预取、统计
├── run_experiments.py  # 一键跑实验并出图
├── requirements.txt
└── results/            # CSV 与论文用图（已含一次运行结果）
```

## 环境要求

- Python 3.10+
- 依赖见 `requirements.txt`（matplotlib，纯 CPU 即可）

## 运行

```powershell
cd kv-cache-sim
python -m venv .venv
.\.venv\Scripts\pip install -r requirements.txt
.\.venv\Scripts\python.exe run_experiments.py
```

运行后更新 `results/csv/` 与 `results/figures/`。  
论文中图 6-1、6-2 对应 `exp_learned_hit_rate_band.png`、`exp_dt_vs_rf_band.png`。

## 策略说明

| 策略 | 说明 |
|------|------|
| FIFO | 基线 |
| LRU | 基线 |
| Learned-RD | 单棵回归树预测重用距离，驱逐 RD 估计最大块 |
| Learned-RF | 5 棵回归树取平均 |
| Learned-RD+Prefetch | Learned-RD + 短 RD 预取（τ=2，H=3） |

训练 seed：0–4；测试 seed：5–9（见 `config.py`）。


## 说明

- 仿真器不存储 K/V 张量，仅统计 hit / miss / evict / prefetch。
- `results/` 下 CSV 与 PNG 为论文对应数据，可随仓库一并提交。
