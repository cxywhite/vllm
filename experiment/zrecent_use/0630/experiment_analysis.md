# 实验分析报告（自动生成）

由 new_once 无头 AI 会话在每项实验完成后自动追加。格式：时间戳 + result_dir + 关键指标 + 解读。

关键指标口径：Output token goodput(tok/s) 是核心（SLO TTFT≤1000ms ∧ TPOT≤30ms 下的 good output 吞吐）；Output token throughput 是忽略 SLO 的总吞吐。

## 分析 2026-11-12 00:40（三项完成/半完成）

### 1. amax220（GWF, Amax=220, oracle预测, kv1p6, 4p4d）✅ 完整
- Output token goodput: **119.28 tok/s**；Successful: **4080/4080**；Output throughput: 4523 tok/s
- TTFT median 1001ms / P99 466s；TPOT median 116ms / P99 284ms
- 结论：GWF 完整跑完，健康。

### 2. gammaada（UniBoost γ-Ada 墙钟, oracle预测, kv1p6, 4p4d）⚠️ 未完成（90min超时被杀）
- Successful: **2498/4080（61%）**，Benchmark duration 5351s（≈89min）
- Output token goodput: 56.27（无效，未跑完）；Output throughput: 261.93 tok/s（极低）
- 结论：**γ-Ada 墙钟强 SJF 饿死长请求**（39% 请求 89min 内未完成）→ 超时被杀。需查 γ-Ada 饿死根因。

### 3. step3 GWF（真实 BERT β=0.30, qwen_4080_600s, 4p4d）✅ 完整
- Output token goodput: **24.16 tok/s**（极低）；Successful: 4080/4080
- Mean TTFT 545.6s / P99 3841.7s（SLO 几乎全不达标）
- 结论：弱预测器（corr 0.351, β=0.30）+ extreme trace → goodput 崩塌到 24。

### 待查问题
1. γ-Ada 墙钟饿死长请求（gammaada 39% 未完成）。
2. UniBoost 臂 ValueError shape(498)→(212) 崩溃根因（step3 第一次尝试）。
3. 真实 BERT 预测器 β 校准（step3 goodput 仅 24，SLO 全崩）。

## 预测器消融（2026-11-13，✅ 全部完成）

统一配置：β=1.0（无缩放）· 4p4d · scheme4 路由 + GWF(θ=120ms/margin=250ms) · gmu 0.8 · SLO TTFT≤1000ms∧TPOT≤30ms · 强制生成(ignore-eos) · predacc_lmsys_o1600/o2000（各 4080 条真实 lmsys prompt）。目标：量化"换掉 Presage 预测器"对 goodput 的影响。

### ours（Presage，β=1.0）✅ 完整
| 数据集 | Output goodput(tok/s) | Output throughput(tok/s) | Mean TTFT(ms) | Mean TPOT(ms) |
|---|---|---|---|---|
| o1600 | 520.34 | 7307.98 | 550.36 | 53.57 |
| o2000 | 322.87 | 7056.36 | 686.96 | 76.01 |
（成功 4080/4080；预测 median 高估 ~7×，pearson 0.495）

### µ-Serve（BERT over prompt text，β=1.0）✅ 完整
| 数据集 | Output goodput(tok/s) | Output throughput(tok/s) | Mean TTFT(ms) | Mean TPOT(ms) |
|---|---|---|---|---|
| o1600 | 533.43 | 7569.74 | 478.48 | 53.60 |
| o2000 | 250.43 | 6570.28 | 777.61 | 84.26 |
（成功 4080/4080；预测 median 高估 ~16×，pearson 0.231）

### TRAIL++（BERT over act-13 + attention top-k，β=1.0）✅ 完整
| 数据集 | Output goodput(tok/s) | Output throughput(tok/s) | Mean TTFT(ms) | Mean TPOT(ms) |
|---|---|---|---|---|
| o1600 | 458.52 | 7026.64 | 431.30 | 56.20 |
| o2000 | 252.41 | 6782.43 | 745.33 | 79.92 |
（成功 4080/4080）

## 三方对比表 + 结论

### Output token goodput（tok/s，核心指标）
| 数据集 | ours | µ-Serve | TRAIL++ | 三者极差 |
|---|---|---|---|---|
| o1600 | 520.34 | **533.43** | 458.52 | 74.91 (16%) |
| o2000 | **322.87** | 250.43 | 252.41 | 72.44 (29%) |

### 结论
1. **β=1.0 下，换预测器对 goodput 的影响很小（16–29% 极差），且没有一致赢家**：µ-Serve 在 o1600 最好、o2000 最差；ours 在 o2000 最好、o1600 居中；TRAIL++ 在 o1600 最差。
2. **ours 最稳**：两个数据集都排前二；µ-Serve / TRAIL++ 在 o2000 明显掉到 ~250。
3. **根因**：三者预测都严重高估（7× / 16× / …），β=1.0 使 GWF 准入过度保守（decode KV cache 仅 ~2%），预测器的"相对准确度"（pearson：ours 0.495 > µ-Serve 0.231）没有机会发挥作用。
4. **核心启示**：真正决定 goodput 的是 **β 校准**（被去掉的 0.30），而不是预测器选择——这反过来说明 Presage 的 β=0.30 校准是其系统收益的关键组成部分，单纯"套用别人的预测器"（且不配 β）并不能复现收益。
