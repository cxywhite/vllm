# 对 review_feedback.md 的复核结论（基于源码二次核验）

复核时间: 2026-04-10  
复核对象: review_feedback.md  
复核范围: baseline_experiment/vllm 与当前调用链文档 llama_prefill_decode_p2p_nccl_call_flow.txt

---

## 结论总览

本次反馈的核心问题判断**基本正确**，但并非“反馈说什么就全对”。基于源码逐条核验后，结论如下：

1. 问题 1（Step 5 行号引用混淆）: **成立（核心成立）**
2. 问题 2（Step 11 `get_finished` 归属）: **成立（核心成立）**
3. 问题 3（Step 3 `start_load_kv` 归属表达）: **成立（轻微成立）**
4. 反馈中存在的细节偏差: **有 1 处行号细节与当前版本不一致**（`impl.forward` 在本地版本是 `layer.py:574`，而非 `575`）

因此：
- 反馈没有方向性错误；
- 但存在一个“行号细节”的版本差异，不应直接照抄为绝对结论。

---

## 逐条复核

## A. 问题 1 复核（Step 5）

反馈主张：文档把函数定义处/函数内部行号和 unified_attention 内调用点混在一起。  
复核结论：**成立**。

源码证据（baseline）:
- `def wait_for_kv_layer_from_connector`: `attention/layer.py:528`
- `connector.wait_for_layer_load(...)`: `attention/layer.py:539`（该函数内部）
- `def maybe_save_kv_layer_to_connector`: `attention/layer.py:542`
- `connector.save_kv_layer(...)`: `attention/layer.py:556`（该函数内部）
- `def unified_attention`: `attention/layer.py:560`
- `wait_for_kv_layer_from_connector(layer_name)`: `attention/layer.py:566`（实际调用点）
- `output = self.impl.forward(...)`: `attention/layer.py:574`（实际调用点）
- `maybe_save_kv_layer_to_connector(...)`: `attention/layer.py:577`（实际调用点）

补充判定：
- 反馈中“应写 575 而非 574”在我当前本地版本不成立；当前文件确实是 `574`。  
- 这是行号漂移级别差异，不影响其“定义行/调用行被混淆”这一核心判断。

---

## B. 问题 2 复核（Step 11）

反馈主张：`get_finished` 调用点不在 scheduler 释放链路，而在 GPU model runner 的 connector 上下文 finally 块中。  
复核结论：**成立**。

源码证据:
- `gpu_model_runner.execute_model` 使用上下文:
  - `gpu_model_runner.py:2296` (`self.maybe_get_kv_connector_output(scheduler_output) as ...`)
- `_get_kv_connector_output` 中:
  - `kv_connector.start_load_kv(...)`: `kv_connector_model_runner_mixin.py:114`
  - `kv_connector.wait_for_save()`: `kv_connector_model_runner_mixin.py:119`
  - `kv_connector.get_finished(scheduler_output.finished_req_ids)`: `kv_connector_model_runner_mixin.py:122`
- scheduler 释放链路对应:
  - `_free_request`: `scheduler.py:1189`
  - `_connector_finished`: `scheduler.py:1265`
  - `return self.connector.request_finished(...)`: `scheduler.py:1277`

进一步说明:
- 原文 Step 11 的“并列列点”容易让读者理解为都在 scheduler 清理阶段发生；
- 更准确写法应区分两条路径：
  - scheduler 侧 `request_finished`
  - model runner finally 侧 `get_finished`

---

## C. 问题 3 复核（Step 3）

反馈主张：`start_load_kv` 的调用归属应写在 `_get_kv_connector_output` 上下文内部，而不是 `maybe_get_kv_connector_output` 本体。  
复核结论：**成立（轻微）**。

源码证据:
- `maybe_get_kv_connector_output` 只是返回上下文管理器:
  - `kv_connector_model_runner_mixin.py:87`
- 真正执行 `start_load_kv` 的位置:
  - `_get_kv_connector_output`: `kv_connector_model_runner_mixin.py:97`
  - 调用 `start_load_kv`: `kv_connector_model_runner_mixin.py:114`

---

## D. “哪些反馈是错的”

严格按源码与当前文件版本复核：

- 反馈中没有发现方向性错误（即三条问题的核心判断都成立）。
- 存在 1 个可明确指出的细节偏差：
  - 反馈正文中对 `impl.forward` 给出的行号 `575` 与当前本地版本不一致；本地是 `574`。
  - 该偏差属于版本行号漂移，不影响技术结论。

---

## E. 修订动作

已据此执行文档修订目标：

1. Step 3: 明确 `start_load_kv` 在 `_get_kv_connector_output` 内执行。  
2. Step 5: 区分“调用点”与“函数内部落点”并修正引用。  
3. Step 11: 拆分 scheduler 清理路径与 model runner finally 路径。  
4. 最小调用图: 改为 `_get_kv_connector_output` 归属写法，避免误导。
