# [WT]prometheus 2025-12-31 15:21:02
import pynvml

_pynvml_inited = False

def _init_nvml():
    global _pynvml_inited
    if not _pynvml_inited:
        pynvml.nvmlInit()
        _pynvml_inited = True

def collect_decode_gpu_stats(device_id: int):
    _init_nvml()
    handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)

    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    print(f'util: {util.gpu}%, mem: {util.memory}%, used: {mem_info.used}, total: {mem_info.total}',flush=True)
    return {
        "compute_util": util.gpu / 100.0,
        "mem_bw_util": util.memory / 100.0,
        "mem_used_bytes": mem_info.used,
        "mem_total_bytes": mem_info.total,
    }
