# 第四轮审核反馈（终轮）

审核时间: 2026-04-10
审阅人: 审核工程师
复核对象:
- review_feedback_round3_response.md（源码调研工程师对第三轮的回应）
- llama_prefill_decode_p2p_nccl_call_flow.txt（当前版本）

---

## 一、对第三轮回应的评价

第三轮提出的 4 项问题逐一复核：

| 问题 | 我的判断 | 工程师复核结论 | 实际验证 |
|------|---------|---------------|---------|
| 2.1 最小调用图函数名不符 | 成立 | 成立 | ✓ 正确 |
| 2.2 Step 10 `:916` 语义偏差 | 成立 | 成立 | ✓ 正确 |
| 2.3 Step 11 "并行链路"措辞 | 轻微成立 | 部分成立，纠正了"不同 step" | ✓ 工程师纠正合理 |
| 2.4 Step 8 else 分支漏标 | 轻微成立 | 成立 | ✓ 正确 |

**关于 2.3 "不同 step vs 同一 step" 的更精确判断**:

我的第三轮反馈写的是"两条链路在不同 step 触发"。工程师纠正为"同一 step 内串行触发"，理由是 `EngineCore.step` 内 `execute_model` 先执行（finally 中 `get_finished`），后执行 `scheduler.update_from_output`（其中 `request_finished`）。

**本轮核实结论：工程师的纠正正确。**

理由：
```
EngineCore.step（core.py:272）内顺序:
  1. scheduler.schedule()        → 返回 SchedulerOutput
  2. execute_model(...)          → finally 中 get_finished (mixin.py:122)
  3. scheduler.update_from_output() → 其中 _free_request → request_finished (scheduler.py:1277)
```

`get_finished` 和 `request_finished` 在同一 `EngineCore.step` 内先后触发，但时机不同：前者是 GPU 前向完成后的清理，后者是该 step 的 `update_from_output` 循环中检测到 `stopped` 时触发。"同一 step 的不同时机串行触发"比"不同 step 触发"更准确。工程师的纠正成立。

---

## 二、对当前文档的完整核验

经过逐行核实源码，以下是最终验证结果：

### 2.1 最小调用图（第五章）

| 引用 | 源码实际 | 验证结果 |
|------|---------|---------|
| `forward_request(prefill_url, ...)` | proxy.py:475 | ✓ |
| `forward_request(decode_url, ...)` | proxy.py:481 | ✓ |
| `connector.build_connector_meta` | scheduler.py:648 | ✓（缩进在 Scheduler.schedule 下） |
| `bind_connector_metadata` | mixin.py:107 | ✓ |
| `start_load_kv` | mixin.py:114 | ✓ |
| `impl.forward` / `save_kv_layer` / `send_tensor` | layer.py:574/:577/:556, connector.py:277 | ✓ |
| `_get_kv_connector_output finally` → `wait_for_save` | mixin.py:119 | ✓ |
| `_get_kv_connector_output finally` → `get_finished` | mixin.py:122 | ✓ |
| decode 路径 `scheduler.update_from_output` | scheduler.py:916 | ✓ |
| decode 路径 `请求结束后 _free_request` | scheduler.py:1189 | ✓ |

### 2.2 各 Step 行号逐行核实

| 章节 | 引用内容 | 源码行号 | 是否正确 |
|------|---------|---------|---------|
| Step 0 proxy 路由 | `:445, :448` | proxy.py:445, 448 | ✓ |
| Step 0 构造 request_id | `:469-470` | proxy.py:469, 470 | ✓ |
| Step 0 prefill 转发 | `:475` | proxy.py:475 | ✓ |
| Step 0 decode 转发 | `:481` | proxy.py:481 | ✓ |
| Step 1 EngineCore.step | `:272` | core.py:272 | ✓ |
| Step 1 scheduler.schedule | `:283` | core.py:283 | ✓ |
| Step 1 SchedulerOutput 构造 | `:624` | scheduler.py:624 | ✓ |
| Step 1 connector metadata | `:648` | scheduler.py:648 | ✓ |
| Step 2 execute_model 入口 | `:2231` | gpu_model_runner.py:2231 | ✓ |
| Step 2 total_num_scheduled_tokens | `:935` | gpu_model_runner.py:935 | ✓ |
| Step 2 num_scheduled_tokens | `:947` | gpu_model_runner.py:947 | ✓ |
| Step 2 positions | `:960-964` | gpu_model_runner.py:960-964 | ✓ |
| Step 2 input_ids gather | `:973-983` | gpu_model_runner.py:973-983 | ✓ |
| Step 3 `_get_kv_connector_output` finally | `:119, :122` | mixin.py:119, 122 | ✓ |
| Step 4 Llama 各行 | `:381/:393/:401/:405/:309/:322/:328` | llama.py 对应行 | ✓ |
| Step 5 unified_attention | `:560` | layer.py:560 | ✓ |
| Step 5 调用点 | `:566/:539/:574/:577/:556` | layer.py 对应行 | ✓ |
| Step 5 impl.forward 直接路径 | `:341` | layer.py:341 | ✓ |
| Step 5 torch.ops 路径 | `:344` | layer.py:344 | ✓ |
| Step 6 save_kv_layer | `:219/:233-235/:271/:273/:276-277` | connector.py 对应行 | ✓ |
| Step 7 PUT_ASYNC | `:128-129/:138/:140/:219/:411/:416/:419/:432/:448/:450/:461` | engine.py 对应行 | ✓ |
| Step 8 listener | `:155/:157/:334/:344/:367` | engine.py 对应行 | ✓ |
| Step 8 buffer 分支 | `:345/:346-350/:356` | engine.py 对应行（已修正） | ✓ |
| Step 9 start_load_kv | `:95/:111/:182/:186/:198/:205/:265/:272` | connector.py 对应行 | ✓ |
| Step 10 decode 转发 | `:481` | proxy.py:481 | ✓ |
| Step 10 logits 采样 | `:2330-2331/:2367` | gpu_model_runner.py:2330-2331, 2367 | ✓ |
| Step 10 update_from_output | `:903/:916` | scheduler.py:903, 916 | ✓ |
| Step 10 token 追加 | `:1076` | scheduler.py:1076（`request.append_output_token_ids`） | ✓ |
| Step 11 scheduler 路径 | `:974-975/:1189/:1265/:1277` | scheduler.py 对应行 | ✓ |
| Step 11 connector 返回 | `:437/:455` | connector.py:437, 455 | ✓ |
| Step 11 worker 路径 | `:119/:122` | mixin.py:119, 122 | ✓ |
| Step 11 get_finished 清理 | `:468/:483-486` | engine.py:468, 483-486 | ✓ |
| 第四章易误解点 1-4 | 各行号 | connector.py/engine.py 对应行 | ✓ |

---

## 三、最终结论

**审查通过。**

经过四轮迭代：
- 第一轮：发现 Step 5 行号混淆、Step 11 get_finished 归属错误、Step 3 start_load_kv 归属
- 第二轮：发现最小调用图 build_connector_meta 层级错误（另更正了我自己 impl.forward 行号的笔误）
- 第三轮：发现最小调用图函数名不存在、Step 10 `:916` 语义偏差、Step 11 措辞、Step 8 else 分支
- 第四轮（终轮）：逐行核实源码，确认所有修正均已正确落地，所有行号与源码一致，无遗留问题

当前版 llama_prefill_decode_p2p_nccl_call_flow.txt 经逐行核验，行号引用和逻辑描述均与源码一致。
