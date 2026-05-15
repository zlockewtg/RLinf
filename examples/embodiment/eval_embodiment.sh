#! /bin/bash

export EMBODIED_PATH="$( cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd )"
export REPO_PATH=$(dirname $(dirname "$EMBODIED_PATH"))
export SRC_FILE="${EMBODIED_PATH}/eval_embodied_agent.py"

export MUJOCO_GL="osmesa"
export PYOPENGL_PLATFORM="osmesa"
export PYTHONPATH=${REPO_PATH}:$PYTHONPATH

# Base path to the BEHAVIOR dataset, which is the BEHAVIOR-1k repo's dataset folder
# Only required when running the behavior experiment.
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
    # Do not let the active conda/venv Python import Isaac Sim's bundled stdlib.
    # This must be done in shell because Python cannot start if PYTHONPATH is
    # already polluted with a different Python version's stdlib.
    ISAAC_PYTHON_LIB="$(readlink -f "$ISAAC_PATH/kit/python/lib" 2>/dev/null || true)"
    FILTERED_PYTHONPATH=()
    IFS=':' read -r -a PYTHONPATH_ENTRIES <<< "${PYTHONPATH:-}"
    for PYTHONPATH_ENTRY in "${PYTHONPATH_ENTRIES[@]}"; do
        [ -n "$PYTHONPATH_ENTRY" ] || continue
        case "$PYTHONPATH_ENTRY" in
            *extscache/omni.kit.pip_archive*) continue ;;
        esac
        REAL_PYTHONPATH_ENTRY="$(readlink -f "$PYTHONPATH_ENTRY" 2>/dev/null || printf '%s' "$PYTHONPATH_ENTRY")"
        if [ -n "$ISAAC_PYTHON_LIB" ]; then
            case "$REAL_PYTHONPATH_ENTRY" in
                "$ISAAC_PYTHON_LIB"/python* | "$ISAAC_PYTHON_LIB"/python*/lib-dynload) continue ;;
            esac
        fi
        FILTERED_PYTHONPATH+=("$PYTHONPATH_ENTRY")
    done
    export PYTHONPATH="$(IFS=:; echo "${FILTERED_PYTHONPATH[*]}")"
    if [ -n "$VIRTUAL_ENV" ]; then
        for VENV_SITE_PACKAGES in "$VIRTUAL_ENV"/lib/python*/site-packages; do
            [ -d "$VENV_SITE_PACKAGES" ] || continue
            export PYTHONPATH="$VENV_SITE_PACKAGES${PYTHONPATH:+:$PYTHONPATH}"
        done
    fi
else
    echo "Warning: Isaac Sim setup script not found at $ISAAC_PATH/setup_python_env.sh"
fi

export ROBOTWIN_PATH=${ROBOTWIN_PATH:-"/path/to/RoboTwin"}
export PYTHONPATH=${REPO_PATH}:${ROBOTWIN_PATH}:$PYTHONPATH

export DREAMZERO_PATH=${DREAMZERO_PATH:-"/path/to/DreamZero"}
export PYTHONPATH=${DREAMZERO_PATH}:$PYTHONPATH

export HYDRA_FULL_ERROR=1

if [ -z "$1" ]; then
    CONFIG_NAME="maniskill_ppo_openvlaoft"
else
    CONFIG_NAME=$1
fi

# NOTE: Set the active robot platform (required for correct action dimension and normalization), supported platforms are LIBERO, ALOHA, BRIDGE, default is LIBERO
ROBOT_PLATFORM=${2:-${ROBOT_PLATFORM:-"LIBERO"}}
EXTRA_ARGS=("${@:3}")

export ROBOT_PLATFORM

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

LOG_DIR="${REPO_PATH}/logs/$(date +'%Y%m%d-%H:%M:%S')-${CONFIG_NAME}" #/$(date +'%Y%m%d-%H:%M:%S')"
MEGA_LOG_FILE="${LOG_DIR}/eval_embodiment.log"
mkdir -p "${LOG_DIR}"
CMD=(
    python "${SRC_FILE}"
    --config-path "${EMBODIED_PATH}/config/"
    --config-name "${CONFIG_NAME}"
    "runner.logger.log_path=${LOG_DIR}"
    "${EXTRA_ARGS[@]}"
)
printf '%q ' "${CMD[@]}"
echo
"${CMD[@]}" 2>&1 | tee "${MEGA_LOG_FILE}"
