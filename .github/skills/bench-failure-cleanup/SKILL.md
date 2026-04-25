---
name: bench-failure-cleanup
description: '检查 benchmark 测试目录是否成功（Successful requests 是否等于目录名中的 nXXXX），并按失败原因安全删除失败目录。用于 experiment_result 下批量审计与清理失败测试数据。关键词: bench log, Successful requests, nXXXX, failed, missing, delete failed dirs.'
argument-hint: 'base-dir=<结果目录> [delete=true] [dry-run=true] [reasons=...]'
user-invocable: true
---

# Benchmark 失败目录检查与清理

## 适用场景
- 你要批量检查某个结果目录（例如 `experiment_result/2p6d`）里的测试是否成功。
- 成功判定规则是：目录名里的 `nXXXX` 与 `bench*.log` 中最后一次 `Successful requests` 数值相同。
- 你要安全删除失败目录，并保留成功目录。

## 判定规则
1. 从目录名提取期望数据量：`_n(\d+)_`。
2. 在 `<run_dir>/log/` 下查找 `bench*.log`（默认取最新修改时间的日志文件）。
3. 从日志中提取最后一次 `Successful requests: <num>`。
4. 若 `<num> == expected_n` 判定 `PASS`，否则判定 `FAIL`。

## 常见失败原因
- `bench_log_not_found`: 目录下未找到 bench 日志。
- `successful_requests_not_found`: 日志中无 `Successful requests` 行。
- `successful_not_equal_expected_n`: 数值不等于目录名中的 `nXXXX`。
- `dir_name_n_not_found`: 目录名无法解析 `nXXXX`。

## 使用步骤
1. 仅检查并输出结果（不删除）：
```bash
python ./.github/skills/bench-failure-cleanup/scripts/check_and_cleanup_failed_bench_dirs.py \
  --base-dir <result_dir> \
  --out-tsv /tmp/bench_check.tsv
```

2. 预演删除（dry-run）：
```bash
python ./.github/skills/bench-failure-cleanup/scripts/check_and_cleanup_failed_bench_dirs.py \
  --base-dir <result_dir> \
  --out-tsv /tmp/bench_check.tsv \
  --delete-fail \
  --dry-run
```

3. 实际删除全部失败目录：
```bash
python ./.github/skills/bench-failure-cleanup/scripts/check_and_cleanup_failed_bench_dirs.py \
  --base-dir <result_dir> \
  --out-tsv /tmp/bench_check.tsv \
  --delete-fail
```

4. 按失败原因删除（可选）：
```bash
python ./.github/skills/bench-failure-cleanup/scripts/check_and_cleanup_failed_bench_dirs.py \
  --base-dir <result_dir> \
  --out-tsv /tmp/bench_check.tsv \
  --delete-fail \
  --fail-reasons successful_not_equal_expected_n,successful_requests_not_found
```

## 安全约束
- 严禁对 `/root/.cache/huggingface/hub` 及其所有子目录执行删除、移动、修改操作（包括但不限于 `rm`、`mv`、覆盖写入、重命名目录）。
- 删除前会做路径安全检查：仅允许删除 `--base-dir` 内的子目录。
- 会输出删除统计：删除数量、未找到数量、跳过数量。

## 输出文件
- TSV 字段：`dir_name\texpected_n\tsuccessful\tstatus\treason\tlog_path`
- 默认输出路径：`/tmp/bench_check.tsv`
