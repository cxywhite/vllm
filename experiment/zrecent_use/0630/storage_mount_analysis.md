# Docker 容器存储挂载问题梳理

## 架构

```
宿主机
├─ /dev/nvme0n1p2 (3.3T NVMe) → /home
│   └─ 容器: /root/myshare ← bind mount from /home/wt/myshare (rw)
│
└─ /dev/sdc1 (15T HDD) → /home/share
    └─ /home/share/huggingface-cache  (HuggingFace 模型缓存, 55%)
        └─ 容器: /root/.cache/huggingface/hub ← bind mount (ro)
           Docker: -v /home/share/huggingface-cache:/root/.cache/huggingface/hub:ro
```

## 故障时间

**2026-07-26 约 13 小时前**（uptime ~448968s 处首次报错），不是 7 月 21 日。

## 根因

**Docker 容器的 bind mount 变 stale，不是磁盘损坏。**

宿主机 `/dev/sdc1` 盘和文件系统一切正常，`/home/share` 下文件完好。问题出在容器内的挂载点：

- 宿主机 sdc1 短暂断开/复位 → 宿主机自动重连恢复
- 但容器 mount propagation = `rprivate`（Docker 默认），宿主机恢复后容器内挂载未更新
- 容器内的挂载变成"僵尸挂载点"：指向已失效的设备会话，任何 I/O 返回 EIO

关键证据：
```
# cat /proc/1/mountinfo | grep huggingface
3906 3881 8:33 /huggingface-cache /root/.cache/huggingface/hub rw,relatime - ext4 /dev/sdc1 ro
#                                          ↑ 无 shared/slave 标记 → 默认 private 传播
```

```
# dmesg | grep -c "sdc1.*error -5"
3631   ← 持续增长的目录块读取错误
```

## 症状

| 操作 | 结果 | 原因 |
|------|------|------|
| `stat /root/.cache/huggingface/hub/` | ✅ | inode 元数据缓存在 dentry cache |
| `ls /root/.cache/huggingface/hub/` | ❌ I/O error | 实际读目录块触发 EIO |
| 宿主机 `ls /home/share/huggingface-cache/` | ✅ 正常 | 宿主机设备会话完好 |

## 应对措施（7 月 24 日）

因 HF 缓存不可用，模型被**重新下载**到 `/root/myshare/models/`（NVMe），以扁平目录存放：

```
/root/myshare/models/
├── Meta-Llama-3-8B-Instruct/       24G  真实 .safetensors
├── Qwen-2.5-7b-Instruct/           15G  真实 .safetensors
├── wt_predictor/                    25M
└── models--*/...                        空壳 stub（blobs 均为 0 字节）
```

下载工具：`/root/myshare/download_hf_model_via_local_proxy.py`

## "空间没变"的解释

| 位置 | 情况 |
|------|------|
| 宿主机 `/home/share` (sdc1, 15T) | 空间未减 — 原缓存数据完好，未曾删除 |
| 宿主机 `/home` (NVMe, 3.3T) | 新增 ~44G — 重新下载的模型。44G/3.3T≈1.3%，不易察觉 |

模型是**重新下载**的，不是从 sdc1 搬运的，所以源盘空间不释放。

## 当前状态

```
                    宿主机                        容器内

/dev/sdc1      ✅ 正常，文件完好            ❌ 僵尸挂载，I/O error
15T, 55%                                   指向已失效的设备会话

/dev/nvme0n1p2 ✅ 正常                      ✅ 正常
3.3T, 100%                                 /root/myshare (含 44G 模型)
```

## 修复

**重启容器**即可。Docker 重新解析 bind mount，指向当前正常工作的设备，`/root/.cache/huggingface/hub` 恢复访问。

或者不重启容器的手动修复（privileged 容器）：

```bash
umount -l /root/.cache/huggingface/hub
mount --bind /proc/1/root/home/share/huggingface-cache /root/.cache/huggingface/hub
```
