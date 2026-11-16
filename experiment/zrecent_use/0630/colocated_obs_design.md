# PD 融合（colocated）部署观测实验设计

> 目的：用观测实验支撑 rebuttal 结论「三维负载建模在融合部署下更有必要性」。
> 状态：**只设计、未执行**。等 8 卡空闲后由用户确认再跑。
> 日期：2026-11-12

## 一句话结论

可行性高：负载模型、MONITOR 日志、3D/最小KV/RR 三路路由、trace、画图代码**全部复用**；唯一缺口是
**融合部署模式**（现有 proxy + vLLM 全是 P/D 分离架构，无 colocated 模式），需要新增一个极简融合代理
（约 100–150 行）+ 8 个独立 vLLM 实例，不碰 KV 迁移、不碰预测器。

## 可复用的现有件（已核实）

| 件 | 位置 | 作用 |
|---|---|---|
| 负载模型 | `disagg_proxy_p2p_nccl_xpyd_reviewfix.py` `DecodeMonitor` + `DecodeLoadModel` | 从 vLLM `/metrics` 读 `vllm:kv_cache_usage_perc` / `vllm:running_tokens` / `vllm:num_requests_running`，算 `Load_Compute` `Load_Memory` `Load_Memory_Capacity` `Load_Bottle`，打 `[WT][MONITOR]` 日志 |
| 3D 快照路由 | `p2d_current_3Dload.py` `_select_decode_min_current_bottle` | `argmin max(compute, memory, capacity)`，即三维快照路由，几行即用 |
| 最小 KV 路由 | `disagg_proxy_p2p_nccl_xpyd_reviewfix.py` `decode_capacity` + `p2d_current_capacity` 变体 | `argmin kv_cache_usage` |
| RR 路由 | `request_count % len(instances)` | 一行 |
| trace | `/root/.cache/huggingface/hub/wt_predictor/0630/dataset/evaluation_dataset/old/llama/burstgpt/sampled_llama-lmsys-chat-updated_n1683_seed42_20260420_093130.csv` | n=1683 LMSYS，与 fig6 一致 |
| 模型 | `/root/myshare/models/Meta-Llama-3-8B-Instruct` | Llama-3-8B-Instruct |
| 画图 | `experiment/paper_figures/fig6_cross_instance_imbalance/plot.py`、`fig4_bottleneck_evolution/plot.py` | 解析 `[WT][MONITOR] decode=... bottle=... compute=... memory=... capacity=...`，跨实例/单实例时序 |

## 缺口（需新增）

1. **融合代理**（极简，~100–150 行，独立文件 `colocated_proxy.py`）：
   - 注册 8 个融合实例（复用 ping 注册机制或直接静态地址列表）。
   - 路由：RR / 最小 KV / 3D 快照三路（复用 `DecodeMonitor` 取负载 + 复制 `_select_decode_min_current_bottle`）。
   - 把 HTTP 请求转发到选中的实例（复用现有 forward 逻辑，去掉 prefill/decode 拆分、ZMQ 路由、预测器）。
   - 打 `[WT][MONITOR]` 日志（把 key 从 `decode=` 改成 `instance=`，画图时同样改）。
2. **8 个独立 vLLM 实例**：`vllm serve` 各自占一个端口，**不设 kv_transfer、不分 P/D**，每实例自跑 prefill+decode。
3. **负载模型微调（可选）**：`_calculate_load` 的 `Load_Compute` 是 decode 导向的近似（`flops = n_req*linear + attn_coeff*running_tokens`）。融合下 `running_tokens` 会含 prompt token，compute 维会自动抬升；若观测发现抬升不明显，再加一个 prefill 计算项（按 prompt token 数 × 全 attention flops）。观测实验用近似即可，先不加。

## 实验矩阵（3 路，控制变量只有路由）

| 路由 | 选实例判据 | 对应现有件 |
|---|---|---|
| RR | 轮询 | 一行 |
| 最小 KV | `argmin kv_cache_usage` | `p2d_current_capacity` |
| 3D 快照 | `argmin max(compute, memory, capacity)` | `p2d_current_3Dload` |

- 每路：8 融合 Llama-3-8B 实例，replay n=1683 LMSYS，1 rep。
- 预估每路 ~30–60 min，三路 ~1.5–3h，8 卡。

## 产出两张图（照现有画法）

1. **跨实例 bottleneck 时序**（照 fig6）：RR / 最小KV / 3D 各一个 panel，看各实例 bottleneck 是否发散。预期 RR 发散严重、最小 KV 略好、3D 更均衡。
2. **单实例三维时序**（照 fig4）：挑一个实例画 compute/memory/capacity 三条线，直接展示融合下 compute 维周期性冲高（prefill 计算瓶颈）。

## 能证明 / 不能证明（诚实边界）

- 能：融合下 compute 维是活的、最小 KV 只看容量会漏掉大 prefill 实例、三维感知在融合下更有必要（与 Observation 1 同层次）。
- 不能直接证明：本方法能均衡（那是「我们行」的间接推论）。**补第三路 3D 快照路由可把间接变直接**：若 3D 路的三维方差都低于前两路，即直接坐实「三维感知同时均衡实例间与实例内」。

## 成本

- 代码：~150 行（融合代理）+ 8 实例启动脚本。
- GPU：8 卡 × ~1.5–3h（一轮）。

## 待办（卡空后按序）

1. 写 `colocated_proxy.py`（复用 DecodeMonitor + 3D 路由）。
2. 写 8 融合实例 + 三路 replay 的启动脚本。
3. 跑三路，确认 `[WT][MONITOR]` 与 fig6 画图输入兼容。
4. 画两张图，与 fig6/fig4 并排对比。
