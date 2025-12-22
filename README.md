# Disaggregated Serving 实验说明

> 目录：文件放置、参数说明、运行步骤、日志与结果组织、常见修改点、清理建议与示例命令。

---

## 文件与目录放置（必须位置）

1. **启动脚本**
   - `predict-schedule/batchsize_experiment/disaggeration_pd_experiment/wt_disagg_p2p_nccl_xpyd.sh`
   - 放置目录：`vllm/examples/online_serving/disaggregated_serving_p2p_nccl_xpyd/`

2. **性能 benchmark 脚本**
   - `benchmark_serving_batchsize.py`
   - 放置目录：`vllm/benchmarks/`

3. **实验输出目录**（脚本会创建）：
   - `vllm/examples/online_serving/disaggregated_serving_p2p_nccl_xpyd/experiment_result/`
   - 该目录下包含 `log/`、`fig/`、`dataset/`（可选，见下）等子目录。

---

## `benchmark_serving_batchsize.py`：新增参数说明

- 新增参数：`--save-input-request`
  - 启用后，会在 `.../experiment_result/dataset/` 中为每次运行创建一个目录并保存当次运行的输入数据集（JSON 格式），用于结果复现。
  - 使用示例：
    ```bash
    python benchmark_serving_batchsize.py --save-input-request
    ```
  - 若不启用：不要加该参数。

- **可修改保存路径的位置**：
  - 如果要改变默认保存路径，可在 `benchmark_serving_batchsize.py` 中查看并修改第 **905 行** 和 **906 行**（脚本中用于创建/写入 dataset 的位置）。

- **注意**：长期积累会占用大量磁盘空间，建议定期清理不需要的 dataset（见“清理建议”）。

---
## 修改 `disagg_proxy_p2p_nccl_xpyd.py`
在启动`wt_disagg_p2p_nccl_xpyd.sh`前先修改vllm/examples/online_serving/disaggregated_serving_p2p_nccl_xpyd/disagg_proxy_p2p_nccl_xpyd.py文件的**188行**和**189行**
```python
    t = start_service_discovery("0.0.0.0", 20001)
    app.run(host="0.0.0.0", port=20006)
```
---
## 启动 `wt_disagg_p2p_nccl_xpyd.sh` 脚本

1. 给脚本增加执行权限：
   ```bash
   cd vllm/examples/online_serving/disaggregated_serving_p2p_nccl_xpyd/
   chmod +x wt_disagg_p2p_nccl_xpyd.sh
   ./wt_disagg_p2p_nccl_xpyd.sh
   ```

2. 脚本运行后会在 `experiment_result/log/` 下创建日志文件，例如：
   - `prefill1.log`
   - `decode1.log`
   - `proxy.log`
   - 若干 `benchmark_np1_YYYYMMDD_HHMMSS.log`（按运行产生）

3. **若要修改脚本中日志/结果路径**，请打开 `wt_disagg_p2p_nccl_xpyd.sh`，查找并修改以下变量/行：
   - **第 22 行**：`LOG_DIR` 变量（日志根目录）
   - **第 261 行、293 行、319 行**：与日志文件路径相关的变量/写入点
   - **第 341 行**：`log_file` 变量（单次 benchmark 的具体 log 名称/位置）

---

## 根据 GPU 利用率组织日志与绘图流程

1. 运行不同 decode 节点 GPU 利用率（例如 `0.3`、`0.7`）时，建议为每个配置创建独立目录：
   ```text
   vllm/examples/online_serving/disaggregated_serving_p2p_nccl_xpyd/experiment_result/log/gpu_0.3/
   vllm/examples/online_serving/disaggregated_serving_p2p_nccl_xpyd/experiment_result/log/gpu_0.7/
   ```

2. 每次实验完成后，将 `log/` 下对应的 `benchmark_np1_*.log` 等文件移动到对应 `gpu_X` 目录下，便于后续对比分析。

3. 使用 `log_parser_and_plot.py`（仓库中应有此脚本）对不同 GPU 利用率的目录进行解析并绘图：
   - 输出目录：`.../experiment_result/fig/`
   - **脚本使用示例**：请参照 `log_parser_and_plot.py` 文件顶部的使用说明（脚本顶部通常有示例命令）。

---

## 日志/数据组织示例（推荐）

```
vllm/examples/online_serving/disaggregated_serving_p2p_nccl_xpyd/
├─ wt_disagg_p2p_nccl_xpyd.sh
├─ experiment_result/
│  ├─ log/
│  │  ├─ gpu_0.3/
│  │  │  ├─ benchmark_np1_20251122_083610.log
│  │  │  └─ prefill1.log
│  │  └─ gpu_0.7/
│  │     └─ ...
│  ├─ fig/
│  │  └─ (生成的图表 png/pdf)
│  └─ dataset/
│     └─ run_YYYYMMDD_HHMMSS/  # 由 --save-input-request 生成，内含 json
└─ ...
```

---

## 定期清理建议（避免磁盘耗尽）

建议定时删除 `experiment_result/dataset/` 中不再需要的历史数据集。示例：删除 30 天之前的 json 文件（在 Linux 上慎用）：

```bash
# 仅示例，运行前请确认路径与策略
find vllm/examples/online_serving/disaggregated_serving_p2p_nccl_xpyd/experiment_result/dataset -type f -name '*.json' -mtime +30 -print
# 若确认要删除：
# find ... -mtime +30 -delete
```

---

## 常见问题与提示

- **脚本没有执行权限**：运行 `chmod +x` 后再执行。
- **没有看到日志**：检查 `LOG_DIR` 设置（脚本第 22 行）和执行用户的写入权限。
- **想改变保存 dataset 的位置**：编辑 `benchmark_serving_batchsize.py` 中第 905、906 行。
- **想改变生成日志的文件名/位置**：编辑 `wt_disagg_p2p_nccl_xpyd.sh` 中第 261、293、319、341 行。
- **存储空间不足**：先备份然后删除旧的 `dataset` 条目；或只在关键实验启用 `--save-input-request`。

---

## 示例一键流程（参考）

```bash
# 进入目录并给脚本可执行权限
cd vllm/examples/online_serving/disaggregated_serving_p2p_nccl_xpyd/
chmod +x wt_disagg_p2p_nccl_xpyd.sh

# 启动分布式实验（在脚本内已配置好参数）
./wt_disagg_p2p_nccl_xpyd.sh

# 在另一个终端运行 benchmark 并保存输入请求
cd vllm/benchmarks/
python benchmark_serving_batchsize.py --save-input-request

# 实验结束后，将 log 文件移动到 gpu_x 目录
mv experiment_result/log/benchmark_np1_*.log experiment_result/log/gpu_0.3/

# 运行解析与绘图脚本（参照脚本顶部示例）
python path/to/log_parser_and_plot.py
```
---

