# Baseline PastFuture Schedule

## 1. 目录功能说明

本文件夹实现如下功能：

- 在 **decode 实例** 中实现pastfuture 调度逻辑  
- 在 **./design_predict_activation_experiment/wt_predict_activation.sh文件中的USE_PASTFUTURE_SCHEDULER选项决定是否开启**

---

## 2. 文件修改内容与放置位置

### 2.1 修改文件

实现pastfuture调度逻辑：

```python
# [WT]custom schedule 2025-12-19 13:55:34
# 修改内容
# [WT] end
```

具体修改文件与路径如下：

- `arg_utils.py`  
  放置路径：  
  ```
  vllm/vllm/engine

  ```
  
- `scheduler.py`
  放置路径：  
  ```
  vllm/vllm/config
  ```

- `pastfuture_queue.py`  
  放置路径：  
  ```
  vllm/vllm/v1/core/sched/pastfuture_queue.py
  ```

- `pastfuture_scheduler.py`  
  放置路径：  
  ```
  vllm/vllm/v1/core/sched/pastfuture_scheduler.py
  ```

---

## 3. 文件功能说明

### 3.1 arg_utils.py

该脚本新增参数控制是否开启pastfuture调度

### 3.2 scheduler.py

添加了pastfuture调度相关配置参数

### 3.3 pastfuture_queue.py

在原来queue基础上实现pastfuture_queue

### 3.4 pastfuture_scheduler.py

在原来scheduler基础上实现pastfuture_scheduler

---

## 4. 当前实验数据集

当前版本使用 **测试数据集** 进行验证，数据集路径如下：

```
predict-schedule/baseline_experiment/pastfuture/dataset/lmsys-50k-filtered-with-sample
```


