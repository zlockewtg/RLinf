#! /bin/bash

set -euo pipefail

EMBODIED_PATH="$( cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_PATH=$(dirname $(dirname "$EMBODIED_PATH"))
CONFIG_NAME=${CONFIG_NAME:-behavior_ppo_openpi05_eval_move_to_skill}
ROBOT_PLATFORM=${ROBOT_PLATFORM:-LIBERO}
MAX_PARALLEL=${MAX_PARALLEL:-1}
THRESHOLD=${THRESHOLD:-0.8}
DATASET_ROOT=${DATASET_ROOT:-/mnt/public/mjwei/download_models/2025-challenge-demos}
ENVS_PER_TASK=${ENVS_PER_TASK:-16}
LOG_ROOT=${LOG_ROOT:-"${REPO_PATH}/logs/move_to_eval_$(date +'%Y%m%d-%H%M%S')"}

declare -A TASK_IDS=(
    [turning_on_radio]=0
    [hanging_pictures]=34
    [attach_a_camera_to_a_tripod]=35
    [clean_a_trumpet]=37
    [cook_cabbage]=41
    [chop_an_onion]=42
    [cook_hot_dogs]=45
    [cook_bacon]=46
)

declare -A TASK_SCENES=(
    [turning_on_radio]=house_double_floor_lower
    [hanging_pictures]=house_double_floor_lower
    [attach_a_camera_to_a_tripod]=house_double_floor_upper
    [clean_a_trumpet]=house_double_floor_upper
    [cook_cabbage]=house_single_floor
    [chop_an_onion]=house_double_floor_lower
    [cook_hot_dogs]=house_single_floor
    [cook_bacon]=house_single_floor
)

declare -A TASK_INSTANCE_IDS=(
    [turning_on_radio]=1,2,3,4,5,6,7,8
    [hanging_pictures]=2,3,4,5,6,9,10,11
    [attach_a_camera_to_a_tripod]=1,2,3,4,5,6,7,8
    [clean_a_trumpet]=1,2,3,4,5,6,7,8
    [cook_cabbage]=3,4,5,6,7,8,9,10
    [chop_an_onion]=1,2,3,4,6,8,9,10
    [cook_hot_dogs]=1,2,3,4,5,6,7,8
    [cook_bacon]=1,2,3,4,5,6,7,8
)

if [ "$#" -gt 0 ]; then
    TASKS=("$@")
else
    TASKS=(
        turning_on_radio
        hanging_pictures
        attach_a_camera_to_a_tripod
        clean_a_trumpet
        cook_cabbage
        chop_an_onion
        cook_hot_dogs
        cook_bacon
    )
fi

mkdir -p "${LOG_ROOT}"

count_move_to_fragments() {
    local task_id="$1"
    local instance_ids="$2"
    "${REPO_PATH}/.venv/bin/python" - "$DATASET_ROOT" "$task_id" "$instance_ids" <<'PY'
import json
import re
import sys
from pathlib import Path

dataset_root = Path(sys.argv[1])
task_id = int(sys.argv[2])
instance_ids = [int(x) for x in sys.argv[3].split(",") if x]
annotation_dir = dataset_root / "annotations" / f"task-{task_id:04d}"

def normalize(text):
    text = "" if text is None else str(text).lower().replace("_", " ")
    return re.sub(r"[^a-z0-9]+", " ", text).strip()

total = 0
missing = []
for instance_id in instance_ids:
    episode_index = task_id * 10000 + instance_id * 10
    path = annotation_dir / f"episode_{episode_index:08d}.json"
    if not path.is_file():
        missing.append(str(instance_id))
        continue
    with path.open("r", encoding="utf-8") as f:
        annotation = json.load(f)
    for skill in annotation.get("skill_annotation", []):
        descriptions = skill.get("skill_description", [])
        if isinstance(descriptions, str):
            descriptions = [descriptions]
        if any(
            normalize(description) == "move to"
            or normalize(description).startswith("move to ")
            for description in descriptions
        ):
            total += 1

if missing:
    raise SystemExit(
        "Missing annotation files for instance ids: " + ",".join(missing)
    )
print(total)
PY
}

choose_envs_and_epochs() {
    local fragment_count="$1"
    local envs
    local epochs

    if [ "${fragment_count}" -le 0 ]; then
        echo "No move_to fragments found." >&2
        return 1
    fi

    if [ "${ENVS_PER_TASK}" = "all" ]; then
        envs="${fragment_count}"
        epochs=1
    else
        envs="${ENVS_PER_TASK}"
        if [ "${envs}" -le 0 ]; then
            echo "ENVS_PER_TASK must be positive or 'all', got ${ENVS_PER_TASK}" >&2
            return 1
        fi
        if [ "${envs}" -gt "${fragment_count}" ]; then
            envs="${fragment_count}"
        fi
        epochs=$(( (fragment_count + envs - 1) / envs ))
    fi

    echo "${envs} ${epochs}"
}

run_task() {
    local task="$1"
    if [ -z "${TASK_IDS[$task]+x}" ]; then
        echo "Unknown task '${task}'. Supported tasks: ${!TASK_IDS[*]}" >&2
        return 1
    fi

    local task_id="${TASK_IDS[$task]}"
    local scene_model="${TASK_SCENES[$task]}"
    local instance_ids="${TASK_INSTANCE_IDS[$task]}"
    local instance_ids_list="[${instance_ids}]"
    local fragment_count
    fragment_count="$(count_move_to_fragments "${task_id}" "${instance_ids}")"
    local envs_and_epochs
    envs_and_epochs="$(choose_envs_and_epochs "${fragment_count}")"
    read -r num_envs eval_epochs <<< "${envs_and_epochs}"

    local task_log_dir="${LOG_ROOT}/${task}"
    mkdir -p "${task_log_dir}"
    echo "=== Evaluating all move_to fragments on ${task}; fragments=${fragment_count}, envs=${num_envs}, eval_rollout_epoch=${eval_epochs}; logs: ${task_log_dir} ==="
    bash "${EMBODIED_PATH}/eval_embodiment.sh" \
        "${CONFIG_NAME}" \
        "${ROBOT_PLATFORM}" \
        "skill_eval.task_id=${task_id}" \
        "skill_eval.activity_name=${task}" \
        "skill_eval.activity_definition_id=0" \
        "skill_eval.scene_model=${scene_model}" \
        "skill_eval.activity_instance_id=${instance_ids_list}" \
        "skill_eval.dataset_root=${DATASET_ROOT}" \
        "env.eval.move_to_eval.activity_name=${task}" \
        "env.eval.move_to_eval.activity_instance_id=${instance_ids_list}" \
        "env.eval.move_to_eval.dataset_root=${DATASET_ROOT}" \
        "env.eval.move_to_eval.success_distance_threshold=${THRESHOLD}" \
        "env.eval.replay_init.skill_occurrence=all" \
        "env.eval.replay_init.sample_mode=sequential" \
        "env.eval.total_num_envs=${num_envs}" \
        "algorithm.eval_rollout_epoch=${eval_epochs}" \
        "runner.logger.log_path=${task_log_dir}" \
        "runner.logger.experiment_name=behavior_ppo_openpi05_eval_move_to_all_${task}"
}

running=0
for task in "${TASKS[@]}"; do
    if [ "${MAX_PARALLEL}" -le 1 ]; then
        run_task "${task}"
    else
        run_task "${task}" &
        running=$((running + 1))
        if [ "${running}" -ge "${MAX_PARALLEL}" ]; then
            wait -n
            running=$((running - 1))
        fi
    fi
done

if [ "${MAX_PARALLEL}" -gt 1 ]; then
    wait
fi

echo "All move_to evals finished. Logs are under ${LOG_ROOT}"
