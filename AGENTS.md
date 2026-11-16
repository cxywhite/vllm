# AGENTS.md

本仓库是 vLLM 调度 / 预测相关的实验代码（`predict-schedule`）。
本文件只做**路由索引**：按功能拆分的约定放在 `docs/agents/` 下（rebuttal 规范在 `paper_final/rebuttal/`）。
先看下面的路由表，按当前任务加载对应的功能文件，不必通读全部。

## 路由表

| 你要做什么 | 读哪个文件 | 何时必读 |
|---|---|---|
| 写 / 改压测脚本、测量、带宽 / IOPS 测试 | `docs/agents/measurement.md` | 新增或修改压测脚本前必读 |
| 跑分离式 vLLM（disagg）实验、A/B 准入隔离 | `docs/agents/disagg_experiments.md` | 跑之前必查 |
| 回滚、丢弃未提交改动、找安全点 | `docs/agents/git_rollback.md` | 需要回滚时读 |
| 回答审稿人问题 / 起草 rebuttal | `paper_final/rebuttal/rebuttal_guide.md`；**论文唯一权威版本是 `paper_final/(1).pdf`**（LaTeX 草稿 `paper/paper_latex/content/*.tex` 已与 PDF 不一致，公式/表述一律以 PDF 为准，不要引用草稿里的内容） | 写回复前必读 |
| 找历史实验结果 / 预测数据（RR/Dynamo/Ours、5p1d 预测结果、预测实验目录） | `/root/myshare/experiment_inventory.md`（兜底：相对仓库根 `../myshare/experiment_inventory.md`；`/root/myshare` 与 `/openrlhf/myshare` 是同一份，也可用 `/openrlhf/myshare/experiment_inventory.md`） | 需要定位任何实验结果目录/文件时先读，再从该文件路由到具体路径 |

## 全局红线（任何操作都适用）

1. **实验结果不写进代码注释**：实测数字、测量日期、实测表格只进文档（`experiment_log.md` / `workload_trials.md` / 报告）；代码注释只留设计逻辑、结构理由与配置约定。
2. **数字必须可溯源**：文档中的每个数字都应能追溯到某个 JSON 字段或某条命令的输出。
3. **变量名带单位**（`_bytes` / `_mib` / `_s` / `_ms`）：单位错误不会报错，只会让结论系统性偏差。
4. **GPU 资源分配优先级（共享服务器）**：8 卡能空出来 → **优先跑 8 卡（4p4d）实验**；否则做能分配出卡的实验（如 1p1d 只用 2 卡）。详细约定与当前待办见 `docs/agents/disagg_experiments.md` §8。
5. **汇总实验结果要写清配置**：任何结果汇总表 / 对比表，每个数据点都必须带完整配置——trace 生成规则、PD 比例、gpu util / gmu、SLO，以及各调度器的专有旋钮（GWF 的 θ / margin / 窗口上限，UniBoost 的 γ / clock / victim）。只给 goodput 数字、不写配置的表视为不合格。格式参考 `experiment/zrecent_use/0630/gwf_vs_uniboost_configs.md`。

> 各功能文件的详细条目（含踩坑与事故记录）以上表对应文件为准。
