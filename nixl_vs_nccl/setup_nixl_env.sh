#!/bin/bash
#=============================================================================
# NixlConnector 环境一键安装脚本
# 用法: bash setup_nixl_env.sh
#=============================================================================

set -e

echo "============================================"
echo "  NixlConnector 环境安装脚本"
echo "============================================"
echo ""

# ---------------------------------------------
# 1. 检查 CUDA 环境
# ---------------------------------------------
echo "[1/4] 检查 CUDA 环境..."
if ! command -v nvidia-smi &> /dev/null; then
    echo "❌ nvidia-smi 未找到，请确保 CUDA 环境已正确配置"
    exit 1
fi

NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
if [ "$NUM_GPUS" -lt 2 ]; then
    echo "❌ 需要至少 2 块 GPU，当前只有 $NUM_GPUS 块"
    exit 1
fi
echo "✅ 检测到 $NUM_GPUS 块 GPU"

# ---------------------------------------------
# 2. 检查 Python 版本
# ---------------------------------------------
echo ""
echo "[2/4] 检查 Python 环境..."
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "✅ Python 版本: $PYTHON_VERSION"

# ---------------------------------------------
# 3. 检查/安装 nixl（CUDA 12 版本）
# ---------------------------------------------
echo ""
echo "[3/4] 检查/安装 nixl..."

# 检查 CUDA 版本
CUDA_VERSION=$(python3 -c "import torch; print(torch.version.cuda)" 2>/dev/null)
if [ -z "$CUDA_VERSION" ]; then
    echo "⚠️ 无法检测 PyTorch CUDA 版本，尝试安装 cu12 版本"
    CUDA_MAJOR="12"
else
    CUDA_MAJOR=$(echo "$CUDA_VERSION" | cut -d'.' -f1)
    echo "✅ PyTorch CUDA 版本: $CUDA_VERSION"
fi

# nixl（包含 UCX，不需要单独装 ucx-py）
if python3 -c "from nixl._api import nixl_agent" &> /dev/null; then
    echo "✅ nixl 已安装，跳过"
else
    echo "📦 安装 nixl[cu${CUDA_MAJOR}]..."
    pip install "nixl[cu${CUDA_MAJOR}]" -q
    if python3 -c "from nixl._api import nixl_agent" &> /dev/null; then
        echo "✅ nixl 安装成功"
    else
        echo "❌ nixl 安装失败，请参考: https://github.com/ai-dynamo/nixl"
        exit 1
    fi
fi

# ---------------------------------------------
# 4. 检查 mlx5 网卡（RDMA 通信）
# ---------------------------------------------
echo ""
echo "[4/4] 检查网络设备..."
if ip link show | grep -q "mlx5"; then
    echo "✅ 检测到 mlx5 网卡（支持 RDMA，推荐用于跨机通信）"
    echo "   UCX 会自动使用 RDMA 传输"
else
    echo "⚠️ 未检测到 mlx5 网卡，将使用 TCP 模式"
fi

# ---------------------------------------------
# 完成
# ---------------------------------------------
echo ""
echo "============================================"
echo "  ✅ 环境安装完成！"
echo "============================================"
echo ""
echo "nixl 包含 UCX，无需单独安装 ucx-py"
echo ""
echo "下一步："
echo "  1. 确认 toy_proxy_server.py 在当前目录下"
echo "  2. 运行 launch_nixl_disagg.sh 启动服务"
echo ""
