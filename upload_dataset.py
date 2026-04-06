# upload_folder_to_hf.py
# 放在你的容器工作目录下运行，例如 python upload_folder_to_hf.py

import os
from huggingface_hub import HfApi, whoami, login
from pathlib import Path

# ---------------- 配置区（根据你的实际情况修改） ----------------
LOCAL_FOLDER = "/root/predict-schedule/vllm/examples/online_serving/disaggregated_serving_p2p_nccl_xpyd/wt_experiment/dataset"          # 你要上传的本地文件夹（绝对路径）
REPO_ID = "greatrock/predictor"          # 你的 Hugging Face dataset repo，例如 greatrock/xxx
REPO_TYPE = "dataset"                           # 固定为 "dataset"
PATH_IN_REPO = "dataset/dataset"                               # 留空 = 上传到仓库根目录，保留完整子目录结构
                                                # 如果想放到子目录：PATH_IN_REPO = "data/v1/raw"

COMMIT_MESSAGE = "Upload mixed dataset folder (csv + json + subdirs) from remote container"
PRIVATE = True                                  # 先私有，测试OK再公开

# 可选：忽略某些文件/文件夹（glob 模式）
IGNORE_PATTERNS = ["__pycache__/*", "*.log", ".DS_Store", "temp/*"]

# 可选：只上传特定类型文件（如果需要过滤）
ALLOW_PATTERNS = ["*.csv", "*.json", "*.jsonl", "*.txt"]  # 留空 = 上传所有

# ---------------- 登录检查（已登录通常可跳过，但加保险） ----------------
# 如果之前 hf auth login 成功过，这里一般不需要再跑
# login()  # 会自动读 ~/.cache/huggingface/token

# 验证当前登录用户（可选，调试用）
user_info = whoami()
print(f"当前登录用户: {user_info['name']}")
print(f"Token 有写权限: {'write' in user_info.get('auth', {}).get('accessToken', {}).get('role', '')}")

# ---------------- 核心上传部分 ----------------
api = HfApi()

# 替换原来的 try 块里的打印部分
try:
    commit_info = api.upload_folder(
        folder_path=LOCAL_FOLDER,
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        path_in_repo=PATH_IN_REPO,
        commit_message=COMMIT_MESSAGE,
        # commit_description=COMMIT_DESCRIPTION,
        ignore_patterns=IGNORE_PATTERNS if IGNORE_PATTERNS else None,
        allow_patterns=ALLOW_PATTERNS if ALLOW_PATTERNS else None,
    )
    
    print("\n上传成功！")
    
    # 安全打印 commit 信息（兼容新旧版本）
    if hasattr(commit_info, 'commit_id'):
        print(f"Commit ID: {commit_info.commit_id}")
    elif hasattr(commit_info, 'oid'):
        print(f"Commit OID: {commit_info.oid}")
    else:
        print("Commit 信息：", commit_info)  # 直接打印对象
    
    print(f"仓库地址: https://huggingface.co/datasets/{REPO_ID}")
    if PRIVATE:
        print("当前为私有仓库，可在网页上改为公开。")

except Exception as e:
    print("发生异常：", e)
    # ... 其他排查信息不变
    print("\n常见原因排查：")
    print("1. 网络问题 → ping huggingface.co")
    print("2. Token 无写权限 → 重新 hf auth login --token hf_xxx")
    print("3. Repo 不存在 → 先在网页创建 https://huggingface.co/new-dataset")
    print("4. 文件夹路径错误 → ls -R", LOCAL_FOLDER)