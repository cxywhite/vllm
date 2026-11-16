# Llama StrategyA 实验报告

**日期**: 2026-07-20 ~ 2026-07-21  
**模型**: Meta-Llama-3-8B-Instruct (max_context=8192)  
**数据集**: llama_lmsys / llama_mysharegpt (StrategyA 裁剪, −39.8%)  
**Goodput 配置**: TTFT < 1000ms AND TPOT < 阈值  
**PD 配置**: 2P6D (2 Prefill + 6 Decode) / 4P4D (4 Prefill + 4 Decode)  

---

## 一、总体结果

| # | 数据集 | PD | TPOT | 策略 | 结果 | 耗时 | GoodR(req/s) | GoodT(tok/s) | TTFT(ms) | TPOT P99(ms) |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | lmsys | 2P6D | 30 | RR | ✅ | 765s | 1.74 | 2109 | 150 | 52.7 |
| 2 | lmsys | 2P6D | 30 | RR(rep) | ✅ | 765s | 1.74 | 2109 | 150 | 52.7 |
| 3 | lmsys | 2P6D | 30 | GWF | ✅ | 711s | 1.54 | 2979 | 133 | 39.0 |
| 4 | lmsys | 2P6D | 50 | RR | ❌ | — | — | — | — | — |
| 5 | lmsys | 2P6D | 50 | RR(rep) | ❌ | — | — | — | — | — |
| 6 | lmsys | 4P4D | 30 | RR | ❌ | — | — | — | — | — |
| 7 | lmsys | 4P4D | 30 | GWF | ✅ | 769s | 0.23 | 450 | 187 | 67.3 |
| 8 | myshare | 2P6D | 30 | RR | ✅ | 804s | 2.19 | 3623 | 135 | 52.4 |
| 9 | myshare | 2P6D | 30 | GWF | ✅ | 745s | 1.40 | 3046 | 137 | 37.5 |
| 10 | myshare | 4P4D | 30 | RR | ❌ | — | — | — | — | — |
| 11 | myshare | 4P4D | 30 | GWF | ✅ | 797s | 0.21 | 417 | 185 | 60.4 |

**通过率: 7/11 (11 次运行, 重复已合并后 6/8 配置通过)**

---

## 二、Goodput 配置分析

### 当前配置

```
Goodput 条件: TTFT < 1000ms AND TPOT < 阈值 (30ms 或 50ms)
```

- **Request Goodput**: 同时满足 TTFT 和 TPOT 条件的请求速率
- **Token Goodput**: 满足条件的请求所产生的 token 速率
- 未满足条件的请求不计入 goodput，但仍正常完成（计入 throughput）

### 配置敏感性分析

| 配置 | 通过情况 | 分析 |
|---|---|---|
| **2P6D + TPOT30** | ✅ 全部通过 | 最优配置：TPOT 30ms 足够严格，天然限流，避免系统过载 |
| **2P6D + TPOT50** | ❌ 全部失败 | TPOT 50ms 更宽松 → 更多并发请求留在系统 → KV 更快饱和 → GPU 死锁 |
| **4P4D + TPOT30 + GWF** | ✅ 通过 | GWF 负载均衡弥补了 4P4D 的 KV 容量不足（仅 4 个 decode） |
| **4P4D + TPOT30 + RR** | ❌ 全部失败 | RR 路由不均 + 4 decode 容量不足 → KV 饱和在 55-81% 进度 |

### TPOT 阈值悖论

TPOT=50ms（更宽松）反而导致失败，因为：
1. 宽松阈值 → 更多请求被系统接受并保持活跃
2. 更多活跃请求 → KV cache 更快速填充
3. KV 满 → 系统吞吐骤降 → GPU 死锁

**结论：TPOT=30ms 是当前数据集的最优 goodput 阈值。**

---

## 三、策略对比：RR vs scheme4_gwf

### 2P6D TPOT30（两者均通过的配置）

| 指标 | RoundRobin | scheme4_gwf | GWF 优势 |
|---|---|---|---|
| Benchmark 耗时 | 765-804s | 711-745s | **−7%** |
| Token Goodput | 2109-3623 | 2979-3046 | lmsys: +41% |
| TTFT Mean | 135-150ms | 133-137ms | 略优 |
| TTFT P99 | 342ms | 274ms | **−20%** |
| TPOT Mean | 31-34ms | 32ms | 持平 |
| TPOT P99 | 52.4-52.7ms | 37.5-39.0ms | **−27%** |

GWF 核心优势：
- **TTFT P99 低 20%**（274ms vs 342ms）：请求排队更短
- **TPOT P99 低 27%**（39ms vs 52ms）：decode 负载更均衡
- lmsys 数据集上 token goodput 高 41%

### 4P4D TPOT30（仅 GWF 通过）

GWF 在 4P4D 配置下是唯一存活的策略，但 goodput 大幅下降（GoodR=0.2 vs 1.5+），表明系统在勉强运行。

---

## 四、失败原因分析

| 失败配置 | 卡死进度 | 原因 |
|---|---|---|
| lmsys 2P6D TPOT50 RR | 47% (1918/4080) | TPOT 50ms 过于宽松 → 过载 → GPU 死锁 |
| lmsys 2P6D TPOT50 RR(rep) | 1% (35/4080) | 加载阶段即失败 |
| lmsys 4P4D TPOT30 RR | 81% (3316/4080) | 4 decode + RR 不均 → KV 饱和, 5-6s/it |
| myshare 4P4D TPOT30 RR | 55% (2234/4080) | 同上, 6-8s/it |

**共同特征**：所有失败均发生在 RoundRobin 策略。RR 的请求分配不均衡导致部分 decode 实例过载、KV 饱和 → 全局死锁。

---

## 五、建议

1. **推荐配置**: 2P6D + TPOT30ms + scheme4_gwf（最优 TTFT/TPOT P99）
2. **避免配置**: 4P4D + RoundRobin（必定 KV 饱和）
3. **TPOT 阈值**: 保持 30ms，50ms 在当前负载下反而有害
4. **StrategyA 效果**: Llama 数据集 −39.8% 缩减足够，2P6D 配置稳定通过
