# 论文图片 — 画图脚本汇总

按论文图片顺序排列。

```bash
cd <目录>
python plot.py
```

---

## 目录与状态

| 论文 | 目录 | 类型 | 脚本状态 | 数据状态 |
|---|---|---|---|---|
| Fig.1 | `fig1_inter_scheduling_concept/` | 📐 手绘概念图 | 无脚本，源文件为 `figure1-2.pptx` 中的 figure1 | — |
| Fig.2 | `fig2_intra_scheduling_concept/` | 📐 手绘概念图 | 无脚本，源文件为 `figure1-2.pptx` 中的 figure2 | — |
| Fig.3 | `fig3_temperature_impact/` | 📐 手绘概念图 | 无脚本，原始数据和画图脚本均不存在 | ❌ 丢失 |
| Fig.4 | `fig4_bottleneck_evolution/` | ✅ Python | 原始脚本保留 | ✅ 数据完好 |
| Fig.5 | `fig5_bottleneck_regimes/` | ⚠️ 重建 | 原始脚本和数据均已丢失，从 PDF 提取格点重建 | ❌ 丢失 |
| Fig.6 | `fig6_cross_instance_imbalance/` | ✅ Python | 原始脚本保留 | ✅ 数据完好 |
| Fig.7 | `fig7_oracle_routing/` | ✅ Python | 原始脚本保留 | ✅ 数据完好 |
| Fig.8 | `fig8_predictor_gap/` | ⚠️ 替代 | 原始数据已丢失，改用同批替代数据 | ⚠️ 原始丢失 |
| Fig.9a | `fig9a_fixed_length/` | ✅ Python | 原始脚本保留 | ✅ 数据完好 |
| Fig.9b | `fig9b_token_composition/` | ✅ Python | 原始脚本保留 | ✅ 数据完好 |
| Fig.10 | `fig10_system_architecture/` | 📐 手绘架构图 | 无脚本，源文件为 `figure10-11.pptx` 中的 figure10 | — |
| Fig.11 | `fig11_predictor_architecture/` | 📐 手绘架构图 | 无脚本，源文件为 `figure10-11.pptx` 中的 figure11 | — |
| Fig.12 | `fig12_goodput_comparison/` | ✅ Python | 原始脚本保留 | ✅ 数据完好 |
| Fig.13 | `fig13_ablation/` | ✅ Python | 原始脚本保留 | ✅ 数据完好（脚本内 hardcoded） |
| Fig.14 | `fig14_predictor_latency/` | ✅ Python | 原始脚本保留 | ✅ 数据完好 |
