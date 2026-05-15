#! /bin/bash

set -euo pipefail

EMBODIED_PATH="$( cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_PATH=$(dirname $(dirname "$EMBODIED_PATH"))

CONFIG_NAME=${CONFIG_NAME:-behavior_ppo_openpi05_eval_skill_chain}
ROBOT_PLATFORM=${ROBOT_PLATFORM:-LIBERO}
DATASET_ROOT=${DATASET_ROOT:-/mnt/public/mjwei/download_models/2025-challenge-demos}
NUM_ENVS=${NUM_ENVS:-8}
EVAL_EPOCHS=${EVAL_EPOCHS:-2}
MAX_STEPS=${MAX_STEPS:-4096}
LOG_ROOT=${LOG_ROOT:-"${REPO_PATH}/logs/skill_chain_eval_$(date +'%Y%m%d-%H%M%S')"}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-behavior_ppo_openpi05_eval_skill_chain}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

# Keep Ray / Torch / Isaac temporary files off the shared /mnt/public volume by
# default. The shared filesystem is often near capacity and can make startup look
# hung while workers wait on pycache, compile artifacts, or Ray session files.
LOCAL_TMP_ROOT=${RLINF_LOCAL_TMP_ROOT:-"/tmp/rlinf_${USER:-user}"}
mkdir -p "${LOCAL_TMP_ROOT}/ray" "${LOCAL_TMP_ROOT}/tmp"
if [[ -z "${RAY_TMPDIR:-}" || "${RAY_TMPDIR}" == /mnt/public/* ]]; then
    export RAY_TMPDIR="${LOCAL_TMP_ROOT}/ray"
fi
if [[ -z "${TMPDIR:-}" || "${TMPDIR}" == /mnt/public/* ]]; then
    export TMPDIR="${LOCAL_TMP_ROOT}/tmp"
fi

# Demo-backed turning_on_radio instance ids. Override with e.g.
# INSTANCE_IDS="[1,2,3,4]" bash eval_behavior_skill_chain.sh
INSTANCE_IDS=${INSTANCE_IDS:-"[1,2,3,4,5,6,7,8,10,11,12,13,14,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52]"}

mkdir -p "${LOG_ROOT}"

echo "=== Evaluating BEHAVIOR skill chain ==="
echo "config=${CONFIG_NAME}"
echo "robot_platform=${ROBOT_PLATFORM}"
echo "num_envs=${NUM_ENVS}, eval_rollout_epoch=${EVAL_EPOCHS}, max_steps=${MAX_STEPS}"
echo "logs=${LOG_ROOT}"
echo "RAY_TMPDIR=${RAY_TMPDIR}, TMPDIR=${TMPDIR}"

bash "${EMBODIED_PATH}/eval_embodiment.sh" \
    "${CONFIG_NAME}" \
    "${ROBOT_PLATFORM}" \
    "env.eval.replay_init.dataset_root=${DATASET_ROOT}" \
    "env.eval.omni_config.task.activity_instance_id=${INSTANCE_IDS}" \
    "env.eval.total_num_envs=${NUM_ENVS}" \
    "env.eval.max_episode_steps=${MAX_STEPS}" \
    "env.eval.max_steps_per_rollout_epoch=${MAX_STEPS}" \
    "algorithm.eval_rollout_epoch=${EVAL_EPOCHS}" \
    "runner.logger.log_path=${LOG_ROOT}" \
    "runner.logger.experiment_name=${EXPERIMENT_NAME}" \
    "$@"
