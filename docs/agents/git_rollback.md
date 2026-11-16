# 回滚安全点（git）

> 需要回滚、丢弃未提交改动、找安全点前读。

已建立并推送到远程的回滚点（记录于 2026-11-07）：

| 仓库 | remote | 分支 | commit | 内容 |
|---|---|---|---|---|
| 主仓库 `/root/predict-schedule` | `cxywhite/vllm` | `experiment/20260630` | `a0bdf55` | worked_example 脚本 + workload 假设 + 双峰 trace 构造器 |
| vllm 子模块 `/root/predict-schedule/vllm` | `cxywhite/Custom-vllm` | `v0.11.0-local-v2` | `f3296d1d1` | 简化预测（`WT_SIMPLIFY_PREDICT`）+ GWF/原生 scheduler 都记 `preempted_recompute_tokens`（重算代价）|

回滚命令（**丢弃未提交改动**，先 `git status` 确认不要了再执行）：

```bash
# 主仓库
cd /root/predict-schedule && git checkout experiment/20260630 && git reset --hard a0bdf55
# vllm 子模块（必须单独 checkout，见下）
cd /root/predict-schedule/vllm && git checkout v0.11.0-local-v2 && git reset --hard f3296d1d1
```

注意：

- 主仓库 @ `a0bdf55` 里记录的 vllm gitlink 是 `8669f4d02`，**不是** `f3296d1d1`，
  所以 `git submodule update` 不会给到 f3296d1d1，必须按上面单独 checkout 子模块；
- 两仓库当前都有未提交改动（主仓库删掉的 `bandwith_test/raw_trace_stats_20260629/` 图与 log；
  vllm 的 AIMD→GWF 迁移未提交改动），`reset --hard` 会清掉它们；
- 两 commit 均已推送：`origin/experiment/20260630`、`origin/v0.11.0-local-v2`，本地丢了也能从远程拉。
