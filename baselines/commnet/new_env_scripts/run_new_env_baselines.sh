#!/bin/bash
# ============================================================
# 一键运行新环境 Baseline 测试脚本
# 支持环境：SMAC v2 / MPE / MAMuJoCo
# 支持 baseline：commnet / magic / gacomm / tarcomm / ic3net
#
# 用法：
#   bash run_new_env_baselines.sh [ENV] [BASELINE]
#
#   ENV 可选：smac_v2 | mpe | mamujoco | all
#   BASELINE 可选：commnet | magic | gacomm | tarcomm | ic3net | all
#
# 示例：
#   bash run_new_env_baselines.sh smac_v2 commnet     # 运行单个
#   bash run_new_env_baselines.sh mpe all             # 运行 MPE 所有 baseline
#   bash run_new_env_baselines.sh all commnet         # 所有环境只跑 commnet
#   bash run_new_env_baselines.sh all all             # 全量运行（顺序执行）
# ============================================================

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "${SCRIPT_DIR}")"

ENV_ARG="${1:-all}"
BASELINE_ARG="${2:-all}"

ENVS=()
BASELINES=()

# 解析环境参数
if [[ "${ENV_ARG}" == "all" ]]; then
    ENVS=(smac_v2 mpe mamujoco)
else
    ENVS=("${ENV_ARG}")
fi

# 解析 baseline 参数
if [[ "${BASELINE_ARG}" == "all" ]]; then
    BASELINES=(commnet magic gacomm tarcomm ic3net)
else
    BASELINES=("${BASELINE_ARG}")
fi

echo "=============================================="
echo "  新环境 Baseline 训练任务启动"
echo "  环境: ${ENVS[*]}"
echo "  Baselines: ${BASELINES[*]}"
echo "  工作目录: ${PROJECT_DIR}"
echo "=============================================="

run_script() {
    local env="$1"
    local baseline="$2"
    local script="${SCRIPT_DIR}/${env}/train_${env}_${baseline}_ctap_v2.sh"

    if [[ ! -f "${script}" ]]; then
        echo "[WARN] 脚本不存在，跳过: ${script}"
        return
    fi

    echo ""
    echo "----------------------------------------------"
    echo "  启动: ENV=${env}  BASELINE=${baseline}"
    echo "  脚本: ${script}"
    echo "----------------------------------------------"

    cd "${PROJECT_DIR}"
    if bash -o pipefail "${script}"; then
        echo "[OK] ${env} / ${baseline} 训练完成"
    else
        local exit_code=$?
        echo "[ERROR] ${env} / ${baseline} 训练失败 (exit=${exit_code})"
    fi
}

# 顺序执行
for env in "${ENVS[@]}"; do
    for baseline in "${BASELINES[@]}"; do
        run_script "${env}" "${baseline}"
    done
done

echo ""
echo "=============================================="
echo "  全部任务执行完毕"
echo "=============================================="
