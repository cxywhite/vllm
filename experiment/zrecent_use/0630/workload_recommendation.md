# 负载建议（数据驱动，对齐论文，给实测侧）

> **重大修正（2026-11-07）**：论文第 1163–1173 行明确说 θ=120ms 是"窗口反馈的可达工作点"，
> **30ms TPOT SLO 由 eV 驱逐/demote 单独保证**。所以：
> - ✅ 正解：用论文 3 条**真实 trace 原样（不封顶）**，让长尾请求（max 32k）把 TPOT 推过 120ms →
>   θ 窗口收缩 → GWF 生效。
> - ❌ 之前 `cap8192`（把长尾砍掉）→ TPOT 上不去 → 窗口永不收缩 → GWF 退化成 no-op（−5.4%）。
> - ❌ 我之前自造的双峰/密集 trace 都是错误方向；`WT_GWF_WINDOW_CAP/TPOT_TARGET` 改 θ 的建议**作废**。

## 1. 论文的 3 条 trace（原样使用，不要封顶）

路径根：`/root/.cache/huggingface/hub/wt_predictor/0630/dataset/evaluation_dataset/0630_evaluation_dataset/`

| trace | 文件（qwen 版） | n | out p50 | p99 | max | >8192 | span |
|---|---|---|---|---|---|---|---|
| Qwen | `qwen/qwen_lmsys_qwentrace_4080_600s.csv` | 4080 | 2710 | 32719 | 32735 | 1101 | 598s |
| Mooncake | `mooncake/qwen_lmsys_mooncake_1790_new.csv` | 1790 | 514 | 28013 | 30944 | 618 | 296s |
| BurstGPT | `burstgpt/qwen_lmsys_burstgpt_673.csv` | 673 | 291 | 32731 | 32738 | 150 | 300s |

（Llama 模型用同目录 `llama_*` 前缀版本。）

## 2. 为什么这 3 条能触发 GWF（而合成 trace 不能）

- 单 decode 实例 GPU KV = **881,616 tokens**；长请求（8k–32k）会迅速吃满 KV →
  decode 步时间暴涨（旧 4080 实测：running 257+ 时 meas_tpot p50 = 5264ms）。
- 只要 `meas_tpot > θ=120ms`，GWF 窗口就 `A // 2` 收缩（256→128→64→32），
  把并发压回 SLO 可达区；greedy 则会一直堆到 KV 打满 + 40 分钟 stall。
- 实测佐证（row 1，qwen 未封顶）：greedy p99 TPOT = **156ms**（5× SLO）→ 必然越过 120ms 阈值。
- 封顶 8192 的实验（row 3）：p99 TPOT 只有 37ms → 永远到不了 120ms → GWF 计数器全 0、反而 −5.4%。

## 3. 给实测侧的具体配置（与论文一致）

- trace：上表 3 条，**原样**（不要 cap output、不要改 timestamp）。
- PD：**4P4D 与 2P6D 各跑**（论文 Figure 9 两列）。
- SLO：`--goodput ttft:1000 tpot:30`（论文 30ms TPOT + 1000ms TTFT）。
- 每组合取 **3 次中位数**（论文原文 "median of three runs"）。
- 模型：`QWEN_MODEL_PATH=/root/myshare/models/Qwen-2.5-7b-Instruct`（Llama 同理论文两模型）。
- 预计单轮时长：qwen ≈1.7h（32k 长尾）、mooncake ≈40min、burstgpt ≈55min；先跑快的两条看信号。

## 4. 拟合结论（供解释，不改实验）

1. **干净负载**（短请求）decode meas_tpot ≈ 20 + 0.11×running（ms），近似线性——
   这是 `dense_short` 合成负载的曲线，**不代表真实 trace**；真实 trace 的长尾会引出 5264ms 悬崖。
2. **goodput 阈值 ≈ running 64**（30ms SLO 对应），窗口上限 256 更大——但按论文设计，
   30ms 是由 **eV/demote** 保证，不是窗口；窗口只负责防悬崖。
3. 合成 `dense_short` 的 +2.8%/+5.0% 只是"无悬崖时的尾部改善"，不能代表论文场景。

## 5. 已否决的方向

- 封顶 output（cap8192）——砍掉长尾，机制失效。
- 自造双峰/密集 trace——偏离论文，结论外推性差。
- 改 θ / 窗口上限——论文明确设计如此，属"调参数赢"，不做。
- 冲 KV 悬崖的合成 trace——greedy 会 thrash 卡死（exit 137）。
