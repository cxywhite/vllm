# baseline_oracle_rr 交付位置说明

由于根分区（overlay）**可用空间为 0**，19 GB 的实验材料无法放进 workspace，
因此「oracle + rr」这套内容放在了数据盘上：

```
/root/myshare/baseline_oracle_rr/
```

（`/root/myshare` 是 `/dev/nvme0n1p2`，打包后仍有约 72 GB 余量。）

## 目录结构

```
/root/myshare/baseline_oracle_rr/
├── README.md                                   实验说明 / 环境 / 运行步骤 / 数据映射
├── MANIFEST.md                                 文件来源 + sha256 + 压缩包校验值
├── run_oracle_rr_loadbalance1p3d.sh            ★ 主入口（路径已相对化）
├── scripts/                                    12 个运行时依赖
│   ├── disagg_proxy_p2p_nccl_xpyd.py           rr 侧：纯轮询 proxy
│   ├── test_disagg_proxy_p2d_loadbalance.py    oracle 侧 proxy
│   ├── test_disagg_proxy.py / test_disagg_proxy_p2d.py   可选分支
│   ├── wt_metadata.py / plot_decode_load2.py
│   └── benchmark_serving_load_balance.py + backend_request_func / benchmark_dataset
│       / benchmark_utils / customfunction
├── dataset/                                    12 个 loadbalance CSV（1.4 GB）
├── plot/                                       plotloadbalance2{,_max}.py
└── results/1p3d_results.tar.zst                ★ 已有结果，18.4 GiB → 4.69 GiB
```

## 如何取到本地

该目录在 workspace 之外，GUI 的文件列表可能不显示它。可选：

1. **VS Code**：这台机器上有 VS Code Server 在运行，直接打开
   `/root/myshare/baseline_oracle_rr/` 下载（结果包 4.69 GB 建议单独拉）。
2. **scp / rsync**：从你的终端拉取。
3. 只需要脚本和数据集的话（约 1.4 GB），可以不拉 `results/`。

## 内容要点

- 本目录 = 论文 **Figure 7（image5）**：**轮询（rr） vs oracle**。
- 它与 `experiment/baseline_307/`（**点预测 vs oracle**，Figure 8 / image6）是姊妹实验，
  两者路由规则相同，唯一区别是信号：本目录 rr 侧完全不看长度，baseline_307 的 rr 侧用 `mean_len`。
- 警告：两个包的 `rr/` 目录名相同但语义不同。
- 结果压缩包校验值：`4a82e1e1ec6c0ea92a880e16c5e693906bebf2bfb23f57de1b070e593635f917`
- 所有文件均为副本，**原始目录未做任何修改**（来源与 mtime 见 `MANIFEST.md`）。

## 与 baseline_307 的对照

| | baseline_oracle_rr（本包） | baseline_307 |
|---|---|---|
| 位置 | `/root/myshare/baseline_oracle_rr/` | `/root/predict-schedule/experiment/baseline_307/` |
| 对比 | 轮询 vs oracle | 点预测 vs oracle |
| 论文图 | Fig.7（image5.png） | Fig.8（image6.png） |
| 结果数据 | ✅ 已有，压缩为 4.69 GB | ❌ 原始已丢失，需重跑 |
