#!/bin/bash
# =============================================================================
# 训练配置 —— 改这里后保存，在 motus_yue 根目录执行:  bash scripts/train.sh
# （不会改写 yaml；仅通过命令行把参数传入 train.py）
# 注意：训练流程/启动参数尽量与 /data/250010061/Motus/scripts/train.sh 保持一致
# =============================================================================

TASK="robotwin"
CONFIG_FILE="configs/robotwin_astribot.yaml"

# Robotwin 采样/训练三选一: strict | tail_padding | random_start_strict_fill
ROBOTWIN_SAMPLING_MODE="random_start_strict_fill"

NPROC_PER_NODE="4"
MASTER_ADDR="127.0.0.1"
MASTER_PORT="29500"

DEEPSPEED_CONFIG="configs/zero1.json"
REPORT_TO="tensorboard"

# 训练过程附属输出根目录（TensorBoard 等仍以 yaml / train.py 里的 checkpoint 为准）
OUTPUT_DIR="outputs/motus-${TASK}"

if [ ! -d "$OUTPUT_DIR" ]; then
    mkdir -p "$OUTPUT_DIR"
    echo "Folder '$OUTPUT_DIR' created"
else
    echo "Folder '$OUTPUT_DIR' already exists"
fi

RSM="${ROBOTWIN_SAMPLING_MODE}"
RSM="${RSM,,}"
RSM="${RSM//-/_}"
case "$RSM" in
  strict|tail_padding|random_start_strict_fill) ;;
  *)
    echo "Invalid ROBOTWIN_SAMPLING_MODE: ${ROBOTWIN_SAMPLING_MODE} (need strict | tail_padding | random_start_strict_fill)" >&2
    exit 1
    ;;
esac

echo "Robotwin sampling mode: ${RSM}"

torchrun \
    --nnodes=1 \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --node_rank=0 \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    train/train.py \
    --deepspeed "${DEEPSPEED_CONFIG}" \
    --config "${CONFIG_FILE}" \
    --run_name "${TASK}" \
    --report_to "${REPORT_TO}" \
    --robotwin_sampling_mode "${RSM}"
