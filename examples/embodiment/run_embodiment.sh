#! /bin/bash

export EMBODIED_PATH="$( cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd )"
export REPO_PATH=$(dirname $(dirname "$EMBODIED_PATH"))
export SRC_FILE="${EMBODIED_PATH}/train_embodied_agent.py"

export MUJOCO_GL="egl"
export PYOPENGL_PLATFORM="egl"
export ROBOTWIN_PATH=${ROBOTWIN_PATH:-"/path/to/RoboTwin"}
export PYTHONPATH=${REPO_PATH}:${ROBOTWIN_PATH}:$PYTHONPATH

# Base path to the BEHAVIOR dataset, which is the BEHAVIOR-1k repo's dataset folder
# Only required when running the behavior experiment.
export OMNIGIBSON_NO_OMNI_LOGS=${OMNIGIBSON_NO_OMNI_LOGS:-1}
export OMNIGIBSON_DEBUG=${OMNIGIBSON_DEBUG:-0}
export OMNIGIBSON_DATA_PATH=/mnt/public/zhuchunyang_rl/hf_datasets/BEHAVIOR-1K-datasets-372
export OMNIGIBSON_DATASET_PATH=${OMNIGIBSON_DATASET_PATH:-$OMNIGIBSON_DATA_PATH/behavior-1k-assets/}
export OMNIGIBSON_KEY_PATH=${OMNIGIBSON_KEY_PATH:-$OMNIGIBSON_DATA_PATH/omnigibson.key}
export OMNIGIBSON_ASSET_PATH=${OMNIGIBSON_ASSET_PATH:-$OMNIGIBSON_DATA_PATH/omnigibson-robot-assets/}
export OMNIGIBSON_HEADLESS=${OMNIGIBSON_HEADLESS:-1}
# Base path to Isaac Sim, only required when running the behavior experiment.
export ISAAC_PATH=${ISAAC_PATH:-/mnt/public/zhuchunyang_rl/sim_envs/isaac-sim}
export EXP_PATH=${EXP_PATH:-$ISAAC_PATH/apps}
export CARB_APP_PATH=${CARB_APP_PATH:-$ISAAC_PATH/kit}
if [ -f "$ISAAC_PATH/setup_python_env.sh" ]; then
    source "$ISAAC_PATH/setup_python_env.sh"
    if [ -n "$VIRTUAL_ENV" ]; then
        PYTHON_VERSION=$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
        export PYTHONPATH="$VIRTUAL_ENV/lib/python${PYTHON_VERSION}/site-packages:$PYTHONPATH"
    fi
    export PYTHONPATH=$(python - <<'PY'
import os

paths = [p for p in os.environ.get("PYTHONPATH", "").split(":") if p]
# Isaac Sim's bundled coverage package is incompatible with this venv's numba.
paths = [p for p in paths if "extscache/omni.kit.pip_archive" not in p]
isaac_python_lib = os.path.realpath(
    os.path.join(os.environ.get("ISAAC_PATH", ""), "kit", "python", "lib")
)


def is_isaac_stdlib_path(path):
    real_path = os.path.realpath(path)
    try:
        rel_path = os.path.relpath(real_path, isaac_python_lib)
    except ValueError:
        return False
    parts = rel_path.split(os.sep)
    if len(parts) < 1 or not parts[0].startswith("python"):
        return False
    return len(parts) == 1 or parts[1] == "lib-dynload"


# Do not let the active conda/venv Python import Isaac Sim's bundled stdlib.
paths = [p for p in paths if not is_isaac_stdlib_path(p)]
print(":".join(paths))
PY
)
else
    echo "Warning: Isaac Sim setup script not found at $ISAAC_PATH/setup_python_env.sh"
fi

if [ -z "$1" ]; then
    CONFIG_NAME="maniskill_ppo_openvlaoft"
else
    CONFIG_NAME=$1
fi

# NOTE: Set the active robot platform (required for correct action dimension and normalization), supported platforms are LIBERO, ALOHA, BRIDGE, default is LIBERO
ROBOT_PLATFORM=${2:-${ROBOT_PLATFORM:-"LIBERO"}}

export ROBOT_PLATFORM
export CUDA_LAUNCH_BLOCKING=1  # for debugging
# Libero variant: standard, pro, plus
export LIBERO_TYPE=${LIBERO_TYPE:-"standard"}
if [ "$LIBERO_TYPE" == "pro" ]; then
    export LIBERO_PERTURBATION="all"  # all,swap,object,lan
    echo "Evaluation Mode: LIBERO-PRO | Perturbation: $LIBERO_PERTURBATION"
elif [ "$LIBERO_TYPE" == "plus" ]; then
    export LIBERO_SUFFIX="all"
    echo "Evaluation Mode: LIBERO-PLUS | Suffix: $LIBERO_SUFFIX"
else
    echo "Evaluation Mode: Standard LIBERO"
fi

echo "Using ROBOT_PLATFORM=$ROBOT_PLATFORM"

echo "Using Python at $(which python)"
LOG_DIR="${REPO_PATH}/logs/$(date +'%Y%m%d-%H:%M:%S')-${CONFIG_NAME}" #/$(date +'%Y%m%d-%H:%M:%S')"
MEGA_LOG_FILE="${LOG_DIR}/run_embodiment.log"
mkdir -p "${LOG_DIR}"
CMD="python ${SRC_FILE} --config-path ${EMBODIED_PATH}/config/ --config-name ${CONFIG_NAME} runner.logger.log_path=${LOG_DIR}"
echo ${CMD} > ${MEGA_LOG_FILE}
${CMD} 2>&1 | tee -a ${MEGA_LOG_FILE}
