# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import gc
import inspect
import json
import os
import re
import traceback
from multiprocessing import get_context
from threading import Thread
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from rlinf.envs.behavior.instance_loader import ActivityInstanceLoader
from rlinf.envs.behavior.replay_initializer import (
    maybe_make_replay_initializer,
    replay_plans_to_infos,
)
from rlinf.envs.behavior.utils import (
    apply_env_wrapper,
    apply_runtime_renderer_settings,
    convert_uint8_rgb,
    setup_omni_cfg,
)
from rlinf.envs.utils import list_of_dict_to_dict_of_list, to_tensor
from rlinf.utils.logging import get_logger

__all__ = ["BehaviorEnv"]

MOVE_TO_EVAL_TASKS = {
    "turning_on_radio": {
        "task_id": 0,
        "scene_model": "house_double_floor_lower",
        "activity_instance_id": [1, 2, 3, 4, 5, 6, 7, 8],
    },
    "hanging_pictures": {
        "task_id": 34,
        "scene_model": "house_double_floor_lower",
        "activity_instance_id": [2, 3, 4, 5, 6, 9, 10, 11],
    },
    "attach_a_camera_to_a_tripod": {
        "task_id": 35,
        "scene_model": "house_double_floor_upper",
        "activity_instance_id": [1, 2, 3, 4, 5, 6, 7, 8],
    },
    "clean_a_trumpet": {
        "task_id": 37,
        "scene_model": "house_double_floor_upper",
        "activity_instance_id": [1, 2, 3, 4, 5, 6, 7, 8],
    },
    "cook_cabbage": {
        "task_id": 41,
        "scene_model": "house_single_floor",
        "activity_instance_id": [3, 4, 5, 6, 7, 8, 9, 10],
    },
    "chop_an_onion": {
        "task_id": 42,
        "scene_model": "house_double_floor_lower",
        "activity_instance_id": [1, 2, 3, 4, 6, 8, 9, 10],
    },
    "cook_hot_dogs": {
        "task_id": 45,
        "scene_model": "house_single_floor",
        "activity_instance_id": [1, 2, 3, 4, 5, 6, 7, 8],
    },
    "cook_bacon": {
        "task_id": 46,
        "scene_model": "house_single_floor",
        "activity_instance_id": [1, 2, 3, 4, 5, 6, 7, 8],
    },
}


def _normalize_object_text(text) -> str:
    text = "" if text is None else str(text).lower().replace("_", " ")
    text = re.sub(r"\.n\.\d+", " ", text)
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def _unwrap_env(env):
    while hasattr(env, "env"):
        env = env.env
    return env


def _object_position(obj):
    if obj is None:
        return None
    wrapped_obj = getattr(obj, "wrapped_obj", obj)
    get_pose = getattr(wrapped_obj, "get_position_orientation", None)
    if callable(get_pose):
        return get_pose()[0]
    get_pose = getattr(obj, "get_position_orientation", None)
    if callable(get_pose):
        return get_pose()[0]
    states = getattr(obj, "states", None)
    if states is not None:
        try:
            from omnigibson.object_states import Pose

            return states[Pose].get_value()[0]
        except Exception:
            return None
    return None


def _robot_position(robot):
    get_pose = getattr(robot, "get_position_orientation", None)
    if callable(get_pose):
        return get_pose()[0]
    links = getattr(robot, "links", {})
    for link_name in ("base_footprint_link", "base_link", "torso_lift_link"):
        link = links.get(link_name) if isinstance(links, dict) else None
        pos = _object_position(link)
        if pos is not None:
            return pos
    return None


def _robot_body_positions(robot) -> list:
    if robot is None:
        return []

    positions = []
    links = getattr(robot, "links", {})
    for link_name in ("base_footprint_link", "base_link", "torso_lift_link"):
        link = links.get(link_name) if isinstance(links, dict) else None
        pos = _object_position(link)
        if pos is not None:
            positions.append(pos)

    root_pos = _robot_position(robot)
    if root_pos is not None:
        positions.append(root_pos)
    return positions


def _robot_eef_positions(robot) -> list:
    if robot is None:
        return []

    positions = []
    arm_names = list(getattr(robot, "arm_names", []) or [])
    default_arm = getattr(robot, "default_arm", None)
    if default_arm is not None and default_arm not in arm_names:
        arm_names.append(default_arm)

    get_eef_position = getattr(robot, "get_eef_position", None)
    if callable(get_eef_position):
        for arm in arm_names or ["default"]:
            try:
                pos = get_eef_position(arm)
            except (AttributeError, KeyError, RuntimeError, TypeError, ValueError):
                continue
            if pos is not None:
                positions.append(pos)

    eef_links = getattr(robot, "eef_links", {})
    if isinstance(eef_links, dict):
        for link in eef_links.values():
            pos = _object_position(link)
            if pos is not None:
                positions.append(pos)
    return positions


def _min_distance_to_target(points: list, target_pos, axis_ids: list[int]) -> float:
    distances = []
    for point in points:
        if point is None:
            continue
        point = torch.as_tensor(point, dtype=torch.float32)
        target = torch.as_tensor(target_pos, dtype=torch.float32, device=point.device)
        distances.append(torch.linalg.norm(point[axis_ids] - target[axis_ids]).item())
    return min(distances) if distances else float("nan")


def _find_distance_reward_target(env, target_name: str | None):
    if not target_name:
        return None
    base_env = _unwrap_env(env)
    task = getattr(base_env, "task", None)
    target_norm = _normalize_object_text(target_name)
    object_scope = getattr(task, "object_scope", {}) or {}
    object_instance_to_category = getattr(task, "object_instance_to_category", {}) or {}

    for bddl_inst, entity in object_scope.items():
        if entity is None or getattr(entity, "is_system", False):
            continue
        names = [
            bddl_inst,
            getattr(entity, "name", None),
            getattr(entity, "category", None),
            getattr(entity, "bddl_inst", None),
            object_instance_to_category.get(bddl_inst),
        ]
        normalized_names = [_normalize_object_text(name) for name in names if name]
        if any(
            name == target_norm
            or name.startswith(f"{target_norm} ")
            or target_norm in name.split()
            for name in normalized_names
        ):
            return entity

    scene = getattr(base_env, "scene", None)
    object_registry = getattr(scene, "object_registry", None)
    if callable(object_registry):
        for key in ("name", "category"):
            try:
                obj = object_registry(key, target_name)
            except Exception:
                obj = None
            if obj is not None:
                return obj
    return None


def _distance_reward_for_env(env, cfg, target_override: str | None = None):
    reward_cfg = OmegaConf.select(cfg, "distance_reward")
    if reward_cfg is None or not bool(
        OmegaConf.select(reward_cfg, "enabled", default=False)
    ):
        return None

    configured_target = OmegaConf.select(reward_cfg, "target_object", default=None)
    if configured_target is not None and str(configured_target).strip().lower() == "auto":
        configured_target = None
    target_name = target_override or configured_target
    target = _find_distance_reward_target(env, target_name)
    base_env = _unwrap_env(env)
    robots = getattr(base_env, "robots", None) or getattr(
        getattr(base_env, "scene", None), "robots", None
    )
    robot = robots[0] if robots else None
    target_pos = _object_position(target)

    info = {
        "target_object": str(target_name) if target_name else "",
        "target_object_found": target is not None,
        "target_distance": float("nan"),
        "body_target_distance": float("nan"),
        "eef_target_distance": float("nan"),
        "target_distance_source": "",
        "distance_reward": 0.0,
    }
    if robot is None or target_pos is None:
        return 0.0, info

    axes = str(OmegaConf.select(reward_cfg, "distance_axes", default="xy")).lower()
    axis_ids = [idx for idx, axis in enumerate("xyz") if axis in axes]
    if not axis_ids:
        axis_ids = [0, 1]

    body_distance = _min_distance_to_target(
        _robot_body_positions(robot), target_pos, axis_ids
    )
    eef_distance = _min_distance_to_target(
        _robot_eef_positions(robot), target_pos, axis_ids
    )
    distance_candidates = [
        ("body", body_distance),
        ("eef", eef_distance),
    ]
    finite_candidates = [
        (source, distance)
        for source, distance in distance_candidates
        if np.isfinite(distance)
    ]
    if not finite_candidates:
        return 0.0, info
    distance_source, distance = min(finite_candidates, key=lambda item: item[1])

    scale = float(OmegaConf.select(reward_cfg, "scale", default=1.0))
    reward = -distance * scale
    success_threshold = OmegaConf.select(reward_cfg, "success_threshold", default=None)
    if success_threshold is not None and distance <= float(success_threshold):
        reward += float(OmegaConf.select(reward_cfg, "success_bonus", default=0.0))

    info["target_distance"] = float(distance)
    info["body_target_distance"] = float(body_distance)
    info["eef_target_distance"] = float(eef_distance)
    info["target_distance_source"] = distance_source
    info["distance_reward"] = float(reward)
    return reward, info


def _apply_distance_rewards(cfg, env, rewards, infos, target_overrides=None):
    if target_overrides is None:
        target_overrides = [None] * len(env.envs)
    reward_values = []
    any_distance_reward = False
    for child_env, raw_reward, info, target_override in zip(
        env.envs, rewards, infos, target_overrides, strict=True
    ):
        result = _distance_reward_for_env(child_env, cfg, target_override)
        if result is None:
            reward_values.append(raw_reward)
            continue
        any_distance_reward = True
        reward, reward_info = result
        reward_values.append(reward)
        if isinstance(info, dict):
            reward_dict = info.setdefault("reward", {})
            if isinstance(reward_dict, dict):
                reward_dict["distance"] = reward_info

    if not any_distance_reward:
        return rewards, infos
    return torch.as_tensor(reward_values, dtype=torch.float32), infos


TURNING_ON_RADIO_STAGE_NAMES = (
    "move_to_radio",
    "pickup_from_support",
    "press_radio",
    "place_on_support",
)


def _float_value(value, default: float = float("nan")) -> float:
    if value is None:
        return default
    try:
        if torch.is_tensor(value):
            value = value.detach().reshape(-1)[0].cpu().item()
        return float(value)
    except (RuntimeError, TypeError, ValueError, IndexError):
        return default


def _bool_value(value) -> bool:
    if value is None:
        return False
    try:
        if torch.is_tensor(value):
            value = value.detach().reshape(-1)[0].cpu().item()
        return bool(value)
    except (RuntimeError, TypeError, ValueError, IndexError):
        return False


def _turning_on_radio_direct_stage_info(child_env, state: dict, active_stage_idx: int):
    base_env = _unwrap_env(child_env)
    task = getattr(base_env, "task", None)
    if getattr(task, "activity_name", None) != "turning_on_radio":
        return None

    try:
        from omnigibson.object_states.toggle import ToggledOn
        from omnigibson.reward_functions.support_utils import (
            get_min_eef_distance_to_obj,
            get_min_eef_distance_to_toggle,
            get_stage_objects_by_name,
            is_supported_by_surface,
            is_target_in_hand,
        )
        from omnigibson.reward_functions.turning_on_radio_reward import (
            TurningOnRadioReward,
        )
    except Exception:
        return None

    stage_objects = state.get("stage_objects")
    if stage_objects is None:
        stage_objects = {
            name: get_stage_objects_by_name(base_env, object_names)
            for name, object_names in TurningOnRadioReward.STAGE_OBJECT_NAMES.items()
        }
        state["stage_objects"] = stage_objects
    radio_objects = stage_objects.get("move_to_radio", [])
    pickup_objects = stage_objects.get("pickup_from_support", [])
    if not radio_objects:
        return None
    radio_obj = radio_objects[0]
    support_obj = pickup_objects[1] if len(pickup_objects) > 1 else None
    robots = getattr(base_env, "robots", None) or []
    robot = robots[0] if robots else None
    if robot is None:
        return None

    try:
        toggle_state = radio_obj.states[ToggledOn]
    except Exception:
        toggle_state = None

    radio_pos = _object_position(radio_obj)
    initial_pos = state.get("radio_initial_pos")
    if initial_pos is None and radio_pos is not None:
        initial_pos = torch.as_tensor(radio_pos).detach().clone()
        state["radio_initial_pos"] = initial_pos

    try:
        eef_to_obj_distance = _float_value(get_min_eef_distance_to_obj(robot, radio_obj))
    except Exception:
        eef_to_obj_distance = float("nan")
    try:
        in_hand = _bool_value(is_target_in_hand(robot, radio_obj))
    except Exception:
        in_hand = False
    try:
        on_support = _bool_value(is_supported_by_surface(radio_obj, support_obj))
    except Exception:
        on_support = False

    has_left_support = bool(state.get("has_left_support", False)) or (not on_support)
    has_picked_up = bool(state.get("has_picked_up", False)) or (
        has_left_support and in_hand
    )
    state["has_left_support"] = has_left_support
    state["has_picked_up"] = has_picked_up

    radio_displacement = 0.0
    if radio_pos is not None and initial_pos is not None:
        try:
            radio_displacement = _float_value(
                torch.linalg.norm(torch.as_tensor(radio_pos) - torch.as_tensor(initial_pos)),
                default=0.0,
            )
        except Exception:
            radio_displacement = 0.0

    toggle_steps = 0
    toggled_on = False
    eef_to_toggle_distance = float("nan")
    if toggle_state is not None:
        try:
            toggled_on = _bool_value(toggle_state.get_value())
        except Exception:
            toggled_on = False
        toggle_steps = int(getattr(toggle_state, "robot_can_toggle_steps", 0) or 0)
        try:
            eef_to_toggle_distance = _float_value(
                get_min_eef_distance_to_toggle(robot, radio_obj, toggle_state)
            )
        except Exception:
            eef_to_toggle_distance = float("nan")

    active_stage_idx = max(
        0, min(int(active_stage_idx), len(TURNING_ON_RADIO_STAGE_NAMES) - 1)
    )
    current_stage_name = TURNING_ON_RADIO_STAGE_NAMES[active_stage_idx]
    stage_infos = {
        "move_to_radio": {
            "completed": active_stage_idx > 0,
            "reward": 0.0,
            "completion_bonus": 0.0,
            "eef_to_obj_distance": eef_to_obj_distance,
        },
        "pickup_from_support": {
            "completed": active_stage_idx > 1 or (
                active_stage_idx == 1 and has_picked_up
            ),
            "reward": 0.0,
            "completion_bonus": 0.0,
            "eef_to_obj_distance": eef_to_obj_distance,
            "in_hand": in_hand,
            "on_support": on_support,
            "has_left_support": has_left_support,
            "has_picked_up": has_picked_up,
            "radio_displacement_from_initial": radio_displacement,
        },
        "press_radio": {
            "completed": active_stage_idx > 2 or (
                active_stage_idx == 2 and toggled_on
            ),
            "reward": 0.0,
            "completion_bonus": 0.0,
            "eef_to_toggle_distance": eef_to_toggle_distance,
            "toggle_steps": toggle_steps,
        },
        "place_on_support": {
            "completed": active_stage_idx == 3 and on_support and not in_hand,
            "reward": 0.0,
            "completion_bonus": 0.0,
            "eef_to_obj_distance": eef_to_obj_distance,
            "in_hand": in_hand,
            "on_support": on_support,
        },
    }
    completed_stage_count = sum(
        bool(stage_infos[name].get("completed", False))
        for name in TURNING_ON_RADIO_STAGE_NAMES
    )
    return {
        "current_stage_idx": active_stage_idx,
        "current_stage_name": current_stage_name,
        "completed_stage_count": completed_stage_count,
        "total_stage_count": len(TURNING_ON_RADIO_STAGE_NAMES),
        "all_stages_completed": completed_stage_count == len(TURNING_ON_RADIO_STAGE_NAMES),
        "completion_bonus": 0.0,
        "stage_infos": stage_infos,
    }


def _apply_direct_skill_chain_infos(env, infos, direct_states, active_stage_indices):
    for env_idx, (child_env, info) in enumerate(
        zip(env.envs, infos, strict=True)
    ):
        if not isinstance(info, dict):
            continue
        direct_info = _turning_on_radio_direct_stage_info(
            child_env,
            direct_states[env_idx],
            active_stage_indices[env_idx],
        )
        if direct_info is None:
            continue
        reward_dict = info.setdefault("reward", {})
        if isinstance(reward_dict, dict) and not any(
            isinstance(payload, dict)
            and ("stage_infos" in payload or "current_stage_name" in payload)
            for payload in reward_dict.values()
        ):
            reward_dict["skill_chain_direct"] = direct_info
    return infos


def _apply_move_to_eval_preset(cfg: DictConfig) -> DictConfig:
    move_to_cfg = OmegaConf.select(cfg, "move_to_eval")
    if move_to_cfg is None or not bool(OmegaConf.select(move_to_cfg, "enabled", default=False)):
        return cfg

    activity_name = str(OmegaConf.select(move_to_cfg, "activity_name")).strip()
    if activity_name not in MOVE_TO_EVAL_TASKS:
        supported = ", ".join(sorted(MOVE_TO_EVAL_TASKS))
        raise ValueError(
            f"Unsupported move_to_eval.activity_name={activity_name!r}. "
            f"Supported tasks: {supported}"
        )

    task_info = MOVE_TO_EVAL_TASKS[activity_name]
    scene_model = task_info["scene_model"]
    task_id = int(task_info["task_id"])
    activity_definition_id = int(task_info.get("activity_definition_id", 0))
    instance_ids = OmegaConf.select(move_to_cfg, "activity_instance_id", default=None)
    if instance_ids is None:
        instance_ids = list(task_info["activity_instance_id"])
    else:
        instance_ids = OmegaConf.to_container(instance_ids, resolve=True)
    dataset_root = OmegaConf.select(
        move_to_cfg,
        "dataset_root",
        default="/mnt/public/mjwei/download_models/2025-challenge-demos",
    )
    threshold = float(OmegaConf.select(move_to_cfg, "success_distance_threshold", default=0.8))

    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
    OmegaConf.update(cfg, "success_metric", "target_distance", force_add=True)
    OmegaConf.update(cfg, "success_distance_threshold", threshold, force_add=True)
    OmegaConf.update(cfg, "replay_init.enabled", True, force_add=True)
    OmegaConf.update(cfg, "replay_init.dataset_root", dataset_root, force_add=True)
    OmegaConf.update(cfg, "replay_init.task_id", task_id, force_add=True)
    OmegaConf.update(cfg, "replay_init.skill_name", "move to", force_add=True)
    OmegaConf.update(cfg, "replay_init.stage_boundary", "start", force_add=True)
    OmegaConf.update(cfg, "distance_reward.enabled", True, force_add=True)
    OmegaConf.update(cfg, "distance_reward.target_object", "auto", force_add=True)
    OmegaConf.update(cfg, "distance_reward.distance_axes", "xy", force_add=True)
    OmegaConf.update(cfg, "distance_reward.success_threshold", threshold, force_add=True)
    OmegaConf.update(cfg, "distance_reward.success_bonus", 0.0, force_add=True)
    OmegaConf.update(cfg, "omni_config.task.activity_name", activity_name, force_add=True)
    OmegaConf.update(
        cfg,
        "omni_config.task.activity_definition_id",
        activity_definition_id,
        force_add=True,
    )
    OmegaConf.update(
        cfg,
        "omni_config.task.activity_instance_dir",
        (
            "${oc.env:OMNIGIBSON_DATA_PATH}/2025-challenge-task-instances/"
            f"scenes/{scene_model}/json/{scene_model}_task_{activity_name}_instances"
        ),
        force_add=True,
    )
    OmegaConf.update(cfg, "omni_config.task.activity_instance_id", instance_ids, force_add=True)
    OmegaConf.update(cfg, "omni_config.scene.scene_model", scene_model, force_add=True)
    OmegaConf.update(cfg, "omni_config.task.reward_config.reward_mode", "potential", force_add=True)
    OmegaConf.update(
        cfg,
        "omni_config.task.reward_config.task_specific_reward_name",
        activity_name,
        force_add=True,
    )
    return cfg


def _behavior_env_worker(
    cfg: DictConfig, conn, num_envs: int, replay_seed_offset: int = 0
):
    env = None
    try:
        from omnigibson.envs import VectorEnvironment

        cfg = _apply_move_to_eval_preset(cfg)
        omni_cfg = setup_omni_cfg(cfg)
        instance_loader = ActivityInstanceLoader.from_omni_cfg(omni_cfg)
        replay_initializer = maybe_make_replay_initializer(
            cfg, seed_offset=replay_seed_offset
        )
        group_size = int(OmegaConf.select(cfg, "group_size", default=1))
        if group_size <= 0:
            raise ValueError(f"env.group_size must be positive, got {group_size}.")
        if num_envs % group_size != 0:
            raise ValueError(
                f"Behavior env shard num_envs={num_envs} must be divisible by "
                f"group_size={group_size}. If using num_env_subprocess, make each "
                "subprocess shard contain a whole number of GRPO groups."
            )

        # create env and apply env wrapper if enabled
        omni_cfg_dict = OmegaConf.to_container(
            omni_cfg,
            resolve=True,
            throw_on_missing=True,
        )
        env = VectorEnvironment(num_envs, omni_cfg_dict)
        wrapper_name = OmegaConf.select(omni_cfg, "env.env_wrapper")
        env = apply_env_wrapper(env, wrapper_name)
        apply_runtime_renderer_settings()

        # Isaac Sim's `omni.kit.app` calls ``gc.disable()`` at startup.
        # OmniGibson has self-referential cycles and leaks memory when
        # cyclic GC is disabled. Since we do not need real-time performance,
        # enable cyclic GC here so that we do not encounter OOMs in long runs.
        gc.enable()

        step_signature = inspect.signature(env.step)
        step_params = step_signature.parameters.values()
        step_supports_kwargs = any(
            param.kind == inspect.Parameter.VAR_KEYWORD for param in step_params
        )
        step_supports_get_obs = step_supports_kwargs or "get_obs" in step_signature.parameters
        step_supports_render = step_supports_kwargs or "render" in step_signature.parameters
        skip_intermediate_obs_in_chunk = bool(
            OmegaConf.select(cfg, "skip_intermediate_obs_in_chunk", default=False)
        )

        def _step_env(actions, need_obs: bool):
            if step_supports_get_obs and step_supports_render:
                return env.step(actions, get_obs=need_obs, render=need_obs)
            return env.step(actions)

        def _reset_replay_episode_counters():
            for child_env in getattr(env, "envs", []):
                if hasattr(child_env, "_current_step"):
                    child_env._current_step = 0

        direct_stage_states = []
        direct_active_stage_indices = []

        def _reset_direct_skill_chain_state():
            direct_stage_states[:] = [{} for _ in getattr(env, "envs", [])]
            direct_active_stage_indices[:] = [0 for _ in getattr(env, "envs", [])]

        def _annotate_direct_skill_chain_infos(infos):
            if not bool(OmegaConf.select(cfg, "skill_chain.enabled", default=False)):
                return infos
            return _apply_direct_skill_chain_infos(
                env,
                infos,
                direct_stage_states,
                direct_active_stage_indices,
            )

        def _set_active_stage_indices(stage_indices):
            child_envs = getattr(env, "envs", [])
            if not child_envs:
                return
            for env_idx, (child_env, stage_index) in enumerate(
                zip(child_envs, stage_indices, strict=True)
            ):
                if stage_index is None:
                    continue
                if env_idx < len(direct_active_stage_indices):
                    direct_active_stage_indices[env_idx] = int(stage_index)
                task = getattr(_unwrap_env(child_env), "task", None)
                reward_functions = getattr(task, "_reward_functions", {})
                task_reward = (
                    reward_functions.get("task_specific")
                    if isinstance(reward_functions, dict)
                    else None
                )
                set_stage = getattr(task_reward, "set_active_stage_index", None)
                if callable(set_stage):
                    set_stage(int(stage_index))

        def _prime_replay_reward_stages(replay_plans):
            _set_active_stage_indices(
                [getattr(plan, "target_stage_index", None) for plan in replay_plans]
            )

        def _annotate_replay_infos(infos, replay_infos):
            for info, replay_info in zip(infos, replay_infos, strict=True):
                if isinstance(info, dict):
                    info["replay_init"] = replay_info
            return infos

        def _replay_after_reset(raw_obs, infos, replay_plans):
            if not replay_plans:
                return raw_obs, infos
            max_steps = max(plan.replay_steps for plan in replay_plans)
            replay_infos = replay_plans_to_infos(replay_plans)
            if max_steps <= 0:
                _prime_replay_reward_stages(replay_plans)
                return raw_obs, _annotate_replay_infos(infos, replay_infos)

            action_dim = replay_plans[0].actions.shape[-1]
            action_batch = np.zeros((len(replay_plans), action_dim), dtype=np.float32)

            for step_idx in range(max_steps):
                for env_idx, plan in enumerate(replay_plans):
                    if plan.replay_steps <= 0:
                        action_batch[env_idx] = 0.0
                    elif step_idx < plan.replay_steps:
                        action_batch[env_idx] = plan.actions[step_idx]
                    else:
                        action_batch[env_idx] = plan.actions[-1]
                need_obs = step_idx == max_steps - 1
                raw_obs, _rewards, _terminations, _truncations, infos = _step_env(
                    action_batch, need_obs=need_obs
                )

            _reset_replay_episode_counters()
            _prime_replay_reward_stages(replay_plans)
            return raw_obs, _annotate_replay_infos(infos, replay_infos)

        conn.send(
            {
                "type": "ready",
                "activity_name": instance_loader.activity_name,
            }
        )
        distance_reward_targets = [None] * len(env.envs)
        _reset_direct_skill_chain_state()

        while True:
            cmd, payload = conn.recv()

            if cmd == "reset":
                replay_plans = (
                    replay_initializer.sample_grouped_plans(len(env.envs), group_size)
                    if replay_initializer is not None
                    else None
                )
                replay_instance_ids = (
                    [plan.instance_id for plan in replay_plans]
                    if replay_plans is not None
                    else None
                )
                distance_reward_targets = (
                    [
                        plan.target_object_id or plan.target_object_name
                        for plan in replay_plans
                    ]
                    if replay_plans is not None
                    else [None] * len(env.envs)
                )
                instance_loader.prepare_reset(
                    env,
                    instance_ids=replay_instance_ids,
                    group_size=group_size,
                )
                _reset_direct_skill_chain_state()
                raw_obs, infos = env.reset()
                if replay_plans is not None:
                    raw_obs, infos = _replay_after_reset(raw_obs, infos, replay_plans)
                infos = _annotate_direct_skill_chain_infos(infos)
                reset_rewards = torch.zeros(len(env.envs), dtype=torch.float32)
                _, infos = _apply_distance_rewards(
                    cfg, env, reset_rewards, infos, distance_reward_targets
                )
                conn.send({"type": "ok", "result": (raw_obs, infos)})

            elif cmd == "step":
                result = env.step(payload)
                raw_obs, rewards, terminations, truncations, infos = result
                infos = _annotate_direct_skill_chain_infos(infos)
                rewards, infos = _apply_distance_rewards(
                    cfg, env, rewards, infos, distance_reward_targets
                )
                result = (raw_obs, rewards, terminations, truncations, infos)
                conn.send({"type": "ok", "result": result})

            elif cmd == "chunk_step":
                chunk_actions = payload["chunk_actions"]
                chunk_size = chunk_actions.shape[1]

                raw_obs_list = []
                chunk_rewards = []
                raw_chunk_terminations = []
                raw_chunk_truncations = []
                infos_list = []

                for i in range(chunk_size):
                    actions = chunk_actions[:, i]
                    is_last = i == chunk_size - 1
                    need_obs = not skip_intermediate_obs_in_chunk or is_last
                    raw_obs, step_rewards, terminations, truncations, infos = _step_env(
                        actions, need_obs=need_obs
                    )
                    infos = _annotate_direct_skill_chain_infos(infos)
                    step_rewards, infos = _apply_distance_rewards(
                        cfg, env, step_rewards, infos, distance_reward_targets
                    )
                    if not need_obs:
                        # Normalize intermediate-step observations to None so downstream
                        # code can skip parsing cleanly.
                        raw_obs = None

                    raw_obs_list.append(raw_obs)
                    chunk_rewards.append(to_tensor(step_rewards))
                    raw_chunk_terminations.append(to_tensor(terminations))
                    raw_chunk_truncations.append(to_tensor(truncations))
                    infos_list.append(infos)

                conn.send(
                    {
                        "type": "ok",
                        "result": (
                            raw_obs_list,
                            chunk_rewards,
                            raw_chunk_terminations,
                            raw_chunk_truncations,
                            infos_list,
                        ),
                    }
                )

            elif cmd == "set_active_stage_indices":
                _set_active_stage_indices(payload)
                conn.send({"type": "ok", "result": None})

            elif cmd == "close":
                env.close()
                env = None
                conn.send({"type": "ok", "result": None})
                break
            else:
                raise NotImplementedError(f"Unknown command: {cmd}")

    except Exception:
        conn.send({"type": "error", "traceback": traceback.format_exc()})

    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
        conn.close()


class BehaviorEnv(gym.Env):
    def __init__(
        self,
        cfg,
        num_envs,
        seed_offset,
        total_num_processes,
        worker_info,
        record_metrics=True,
    ):
        self.cfg = cfg
        self.reward_coef = cfg.get("reward_coef", 1)

        self.num_envs = num_envs
        self.ignore_terminations = cfg.ignore_terminations
        self.use_rel_reward = cfg.use_rel_reward
        self.seed_offset = seed_offset
        self.seed = self.cfg.seed + seed_offset
        self.total_num_processes = total_num_processes
        self.worker_info = worker_info
        self.record_metrics = record_metrics
        self._is_start = True
        self.use_thread_worker = self.cfg.get("use_thread_worker", False)
        self.num_env_subprocess = int(self.cfg.get("num_env_subprocess", 1))
        self.env_shard_size = self._split_num_envs(self.num_envs, self.num_env_subprocess)
        self.env_process_list = []
        self.parent_conn_list = []
        self.child_conn_list = []
        self.use_subtask_prompt = bool(self.cfg.get("use_subtask_prompt", False))
        self.subtask_prompt_override = self.cfg.get("subtask_prompt_override", None)
        if self.subtask_prompt_override is not None:
            self.subtask_prompt_override = str(self.subtask_prompt_override).strip()
        self.subtask_prompt_only = bool(self.cfg.get("subtask_prompt_only", False))
        move_to_eval_enabled = bool(
            OmegaConf.select(self.cfg, "move_to_eval.enabled", default=False)
        )
        self.success_metric = str(self.cfg.get("success_metric", "env")).strip().lower()
        if move_to_eval_enabled and self.success_metric == "env":
            self.success_metric = "target_distance"
        self.success_stage_name = self.cfg.get("success_stage_name", None)
        if self.success_stage_name is not None:
            self.success_stage_name = str(self.success_stage_name).strip()
        self.success_distance_threshold = OmegaConf.select(
            self.cfg, "move_to_eval.success_distance_threshold", default=None
        )
        if self.success_distance_threshold is None:
            self.success_distance_threshold = self.cfg.get(
                "success_distance_threshold",
                OmegaConf.select(self.cfg, "distance_reward.success_threshold", default=None),
            )
        if self.success_distance_threshold is not None:
            self.success_distance_threshold = float(self.success_distance_threshold)
        self._stage_prompt_lists: list[list[str] | None] = [None] * self.num_envs
        self._current_stage_prompts: list[str | None] = [None] * self.num_envs
        self.skill_chain_enabled = bool(
            OmegaConf.select(self.cfg, "skill_chain.enabled", default=False)
        )
        self.skill_chain_move_to_threshold = float(
            OmegaConf.select(
                self.cfg,
                "skill_chain.move_to_success_distance_threshold",
                default=self.success_distance_threshold
                if self.success_distance_threshold is not None
                else 0.8,
            )
        )
        self.skill_chain_max_steps_per_subtask = int(
            OmegaConf.select(self.cfg, "skill_chain.max_steps_per_subtask", default=1024)
        )
        self.skill_chain_use_annotation_step_limits = bool(
            OmegaConf.select(
                self.cfg,
                "skill_chain.use_annotation_step_limits",
                default=False,
            )
        )
        self.skill_chain_step_limits_by_skill = {}
        step_limits_cfg = OmegaConf.select(
            self.cfg,
            "skill_chain.step_limits_by_skill",
            default={},
        )
        if step_limits_cfg:
            for key, value in OmegaConf.to_container(step_limits_cfg, resolve=True).items():
                try:
                    self.skill_chain_step_limits_by_skill[
                        self._normalize_skill_text(key)
                    ] = max(1, int(value))
                except (TypeError, ValueError):
                    continue
        self.skill_chain_post_success_steps_by_skill = {}
        post_success_steps_cfg = OmegaConf.select(
            self.cfg,
            "skill_chain.post_success_steps_by_skill",
            default={},
        )
        if post_success_steps_cfg:
            for key, value in OmegaConf.to_container(
                post_success_steps_cfg, resolve=True
            ).items():
                try:
                    self.skill_chain_post_success_steps_by_skill[
                        self._normalize_skill_text(key)
                    ] = max(0, int(value))
                except (TypeError, ValueError):
                    continue
        self.skill_chain_prompts_by_skill = {}
        prompts_cfg = OmegaConf.select(
            self.cfg,
            "skill_chain.prompts_by_skill",
            default={},
        )
        if prompts_cfg:
            for key, value in OmegaConf.to_container(prompts_cfg, resolve=True).items():
                prompt = str(value).strip()
                if prompt:
                    self.skill_chain_prompts_by_skill[
                        self._normalize_skill_text(key)
                    ] = prompt
        self.skill_chain_config_stage_names = self._string_list_from_cfg(
            OmegaConf.select(self.cfg, "skill_chain.stage_names", default=None)
        )
        self.skill_chain_config_stage_prompts = self._string_list_from_cfg(
            OmegaConf.select(self.cfg, "skill_chain.stage_prompts", default=None)
        )
        self.skill_chain_config_stage_limits = self._int_list_from_cfg(
            OmegaConf.select(self.cfg, "skill_chain.stage_limits", default=None)
        )
        self.skill_chain_config_reward_stage_indices = self._int_list_from_cfg(
            OmegaConf.select(self.cfg, "skill_chain.reward_stage_indices", default=None),
            allow_none=True,
        )
        self.skill_chain_config_reward_stage_names = self._string_list_from_cfg(
            OmegaConf.select(self.cfg, "skill_chain.reward_stage_names", default=None)
        )
        self._skill_chain_stage_names: list[list[str]] = [[] for _ in range(self.num_envs)]
        self._skill_chain_stage_prompts: list[list[str]] = [[] for _ in range(self.num_envs)]
        self._skill_chain_stage_limits: list[list[int]] = [[] for _ in range(self.num_envs)]
        self._skill_chain_reward_stage_indices: list[list[int | None]] = [
            [] for _ in range(self.num_envs)
        ]
        self._skill_chain_reward_stage_names: list[list[str | None]] = [
            [] for _ in range(self.num_envs)
        ]
        self._skill_chain_stage_idx = [0 for _ in range(self.num_envs)]
        self._skill_chain_stage_elapsed = [0 for _ in range(self.num_envs)]
        self._skill_chain_completed: list[list[bool]] = [[] for _ in range(self.num_envs)]
        self._skill_chain_failed = [False for _ in range(self.num_envs)]
        self._skill_chain_done = [False for _ in range(self.num_envs)]
        self._skill_chain_failed_stage = [-1 for _ in range(self.num_envs)]
        self._skill_chain_terminal_reported = [False for _ in range(self.num_envs)]
        self._skill_chain_pending_success = [False for _ in range(self.num_envs)]
        self._skill_chain_post_success_remaining = [0 for _ in range(self.num_envs)]

        self.logger = get_logger()

        self.auto_reset = cfg.auto_reset
        self.max_episode_steps = cfg.max_episode_steps
        self.use_fixed_reset_state_ids = cfg.use_fixed_reset_state_ids
        self._step_count = torch.zeros(
            self.num_envs,
            device=self.device,
            dtype=torch.int32,
        )
        if self.record_metrics:
            self._init_metrics()
        self._init_env()

    def _split_num_envs(self, num_envs: int, num_processes: int) -> int:
        """Split ``num_envs`` across ``num_processes`` shards as evenly as possible."""
        assert num_processes > 0, f"num_processes({num_processes}) must be positive"
        if self.use_thread_worker and num_processes > 1:
            self.logger.warning(f"assign num_processes({num_processes}) to 1 when use_thread_worker is True")
            num_processes = 1
        assert num_envs % num_processes == 0, f"num_envs({num_envs}) must be divisible by num_processes({num_processes})"
        return num_envs // num_processes

    def _load_tasks_cfg(self, activity_name: str):
        # Read task description

        task_description_path = os.path.join(
            os.path.dirname(__file__), "behavior_task.jsonl"
        )
        with open(task_description_path, "r") as f:
            text = f.read()
            task_description = [json.loads(x) for x in text.strip().split("\n") if x]
        task_description_map = {
            task_description[i]["task_name"]: task_description[i]["task"]
            for i in range(len(task_description))
        }
        self.task_description = task_description_map[activity_name]

    def _init_env(self):
        self._ctx = get_context("spawn")
        if self.use_thread_worker:
            process_cls = Thread
        else:
            process_cls = self._ctx.Process

        activity_name = None
        for _ in range(self.num_env_subprocess):
            parent_conn, child_conn = self._ctx.Pipe()
            env_process = process_cls(
                target=_behavior_env_worker,
                args=(
                    self.cfg,
                    child_conn,
                    self.env_shard_size,
                    self.seed_offset + len(self.env_process_list),
                ),
                daemon=True,
            )
            env_process.start()
            if not self.use_thread_worker:
                child_conn.close()
                child_conn = None

            msg = parent_conn.recv()
            if msg.get("type") != "ready":
                raise RuntimeError(
                    f"Failed to initialize behavior subprocess env: {msg.get('traceback', msg)}"
                )
            if activity_name is None:
                activity_name = msg["activity_name"]
            elif msg["activity_name"] != activity_name:
                raise RuntimeError(
                    "Behavior env subprocesses reported different activity_name: "
                    f"{activity_name!r} vs {msg['activity_name']!r}"
                )
            self.parent_conn_list.append(parent_conn)
            self.child_conn_list.append(child_conn)
            self.env_process_list.append(env_process)

        self._load_tasks_cfg(activity_name)

    @staticmethod
    def _extend_per_env(dst: list, batch):
        if isinstance(batch, (list, tuple)):
            dst.extend(batch)
        else:
            dst.append(batch)

    @staticmethod
    def _cat_batch_dim(parts: list):
        if not parts:
            return parts
        tensors = [p if torch.is_tensor(p) else torch.as_tensor(p) for p in parts]
        return torch.cat(tensors, dim=0)

    @staticmethod
    def _list_from_cfg(value) -> list:
        if value is None:
            return []
        if OmegaConf.is_config(value):
            value = OmegaConf.to_container(value, resolve=True)
        if isinstance(value, (list, tuple)):
            return list(value)
        return [value]

    @classmethod
    def _string_list_from_cfg(cls, value) -> list[str]:
        return [
            item
            for item in (str(item).strip() for item in cls._list_from_cfg(value))
            if item
        ]

    @classmethod
    def _int_list_from_cfg(cls, value, allow_none: bool = False) -> list[int | None]:
        parsed = []
        for item in cls._list_from_cfg(value):
            if item is None and allow_none:
                parsed.append(None)
                continue
            try:
                parsed.append(int(item))
            except (TypeError, ValueError):
                continue
        return parsed

    @staticmethod
    def _normalize_skill_text(text: str | None) -> str:
        text = "" if text is None else str(text).lower().replace("_", " ")
        return re.sub(r"[^a-z0-9]+", " ", text).strip()

    @staticmethod
    def _metric_truthy(value, threshold: float = 0.5) -> bool | None:
        if value is None:
            return None
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return bool(value)
        if not np.isfinite(numeric):
            return None
        return numeric >= threshold

    def _pick_up_stage_success(self, stage_info: dict, fallback_completed: bool) -> bool:
        metric_success = self._metric_truthy(stage_info.get("has_picked_up"))
        if metric_success is not None:
            return metric_success
        completed = self._metric_truthy(stage_info.get("completed"))
        if completed is not None:
            return completed
        return fallback_completed

    def _reset_skill_chain_state(self, infos: list[dict] | None):
        if not self.skill_chain_enabled:
            return
        infos = infos or [{} for _ in range(self.num_envs)]
        for env_idx in range(self.num_envs):
            info = infos[env_idx] if env_idx < len(infos) and isinstance(infos[env_idx], dict) else {}
            replay_info = info.get("replay_init", {}) if isinstance(info, dict) else {}
            if not isinstance(replay_info, dict):
                replay_info = {}
            names = [
                str(name).strip()
                for name in replay_info.get("replay_stage_skill_names", [])
                if str(name).strip()
            ]
            if not names and self.skill_chain_config_stage_names:
                names = list(self.skill_chain_config_stage_names)
            reward_stage_indices = []
            for value in replay_info.get("replay_stage_indices", []):
                try:
                    reward_stage_indices.append(int(value))
                except (TypeError, ValueError):
                    reward_stage_indices.append(None)
            if not reward_stage_indices and self.skill_chain_config_reward_stage_indices:
                reward_stage_indices = list(self.skill_chain_config_reward_stage_indices)
            reward_stage_names = list(self.skill_chain_config_reward_stage_names)
            prompts = [
                str(prompt).strip()
                for prompt in replay_info.get("replay_stage_prompts", [])
                if str(prompt).strip()
            ]
            if not prompts and self.skill_chain_config_stage_prompts:
                prompts = list(self.skill_chain_config_stage_prompts)
            annotation_limits = []
            for value in replay_info.get("replay_stage_step_limits", []):
                try:
                    annotation_limits.append(max(1, int(value)))
                except (TypeError, ValueError):
                    annotation_limits.append(1)
            default_limit = max(1, int(self.skill_chain_max_steps_per_subtask))
            if self.skill_chain_use_annotation_step_limits and annotation_limits:
                limits = annotation_limits
            elif self.skill_chain_config_stage_limits:
                limits = [max(1, int(limit)) for limit in self.skill_chain_config_stage_limits]
            else:
                limits = [
                    self.skill_chain_step_limits_by_skill.get(
                        self._normalize_skill_text(name),
                        default_limit,
                    )
                    for name in names
                ]
            if len(prompts) < len(names):
                prompts.extend(names[len(prompts) :])
            prompts = [
                self.skill_chain_prompts_by_skill.get(
                    self._normalize_skill_text(name),
                    prompt,
                )
                for name, prompt in zip(names, prompts, strict=False)
            ]
            if len(limits) < len(names):
                limits.extend(
                    [
                        self.skill_chain_step_limits_by_skill.get(
                            self._normalize_skill_text(name),
                            default_limit,
                        )
                        for name in names[len(limits) :]
                    ]
                )
            if len(reward_stage_indices) < len(names):
                reward_stage_indices.extend(
                    range(len(reward_stage_indices), len(names))
                )
            if len(reward_stage_names) < len(names):
                reward_stage_names.extend([None] * (len(names) - len(reward_stage_names)))

            self._skill_chain_stage_names[env_idx] = names
            self._skill_chain_stage_prompts[env_idx] = prompts[: len(names)]
            self._skill_chain_stage_limits[env_idx] = limits[: len(names)]
            self._skill_chain_reward_stage_indices[env_idx] = reward_stage_indices[
                : len(names)
            ]
            self._skill_chain_reward_stage_names[env_idx] = reward_stage_names[
                : len(names)
            ]
            self._skill_chain_stage_idx[env_idx] = 0
            self._skill_chain_stage_elapsed[env_idx] = 0
            self._skill_chain_completed[env_idx] = [False] * len(names)
            self._skill_chain_failed[env_idx] = False
            self._skill_chain_done[env_idx] = len(names) == 0
            self._skill_chain_failed_stage[env_idx] = -1
            self._skill_chain_terminal_reported[env_idx] = False
            self._skill_chain_pending_success[env_idx] = False
            self._skill_chain_post_success_remaining[env_idx] = 0

    def _skill_chain_current_name(self, env_idx: int) -> str | None:
        names = self._skill_chain_stage_names[env_idx]
        idx = self._skill_chain_stage_idx[env_idx]
        if idx < 0 or idx >= len(names):
            return None
        return names[idx]

    def _skill_chain_current_prompt(self, env_idx: int) -> str | None:
        prompts = self._skill_chain_stage_prompts[env_idx]
        idx = self._skill_chain_stage_idx[env_idx]
        if idx < 0 or idx >= len(prompts):
            return self._skill_chain_current_name(env_idx)
        return prompts[idx]

    def _skill_chain_current_policy(self, env_idx: int) -> str:
        if self._skill_chain_done[env_idx]:
            return "__done__"
        if self._skill_chain_failed[env_idx]:
            return "__failed__"
        return self._skill_chain_current_name(env_idx) or "__unknown__"

    def _skill_chain_current_reward_stage_index(self, env_idx: int) -> int | None:
        indices = self._skill_chain_reward_stage_indices[env_idx]
        idx = self._skill_chain_stage_idx[env_idx]
        if idx < 0 or idx >= len(indices):
            return None
        return indices[idx]

    def _skill_chain_current_reward_stage_name(self, env_idx: int) -> str | None:
        names = self._skill_chain_reward_stage_names[env_idx]
        idx = self._skill_chain_stage_idx[env_idx]
        if idx < 0 or idx >= len(names):
            return None
        return names[idx]

    def _skill_name_to_reward_stage_name(self, skill_name: str | None) -> str | None:
        normalized_name = self._normalize_skill_text(skill_name)
        if normalized_name == "move to":
            return "move_to_radio"
        if normalized_name.startswith("pick up"):
            return "pickup_from_support"
        if normalized_name.startswith("press"):
            return "press_radio"
        if normalized_name.startswith("place on"):
            return "place_on_support"
        return None

    def _skill_chain_current_post_success_steps(self, env_idx: int) -> int:
        name = self._skill_chain_current_name(env_idx)
        normalized_name = self._normalize_skill_text(name)
        if normalized_name in self.skill_chain_post_success_steps_by_skill:
            return int(self.skill_chain_post_success_steps_by_skill[normalized_name])
        for configured_name, steps in self.skill_chain_post_success_steps_by_skill.items():
            if normalized_name.startswith(configured_name) or configured_name.startswith(
                normalized_name
            ):
                return int(steps)
        return 0

    def _sync_skill_chain_reward_stages(self):
        if not self.skill_chain_enabled:
            return
        self._call_subproc(
            "set_active_stage_indices",
            [
                self._skill_chain_current_reward_stage_index(env_idx)
                for env_idx in range(self.num_envs)
            ],
        )

    def _skill_chain_completed_count(self, env_idx: int) -> int:
        return int(sum(bool(done) for done in self._skill_chain_completed[env_idx]))

    def _skill_chain_prior_stage_completed(self, env_idx: int, prefix: str) -> bool:
        target_prefix = self._normalize_skill_text(prefix)
        current_idx = self._skill_chain_stage_idx[env_idx]
        for idx, name in enumerate(self._skill_chain_stage_names[env_idx][:current_idx]):
            if self._normalize_skill_text(name).startswith(target_prefix) and bool(
                self._skill_chain_completed[env_idx][idx]
            ):
                return True
        return False

    def _skill_chain_new_terminal_tensor(self) -> torch.Tensor:
        terminal = []
        for env_idx, (done, failed) in enumerate(
            zip(self._skill_chain_done, self._skill_chain_failed, strict=True)
        ):
            is_terminal = bool(done or failed)
            report_terminal = is_terminal and not self._skill_chain_terminal_reported[env_idx]
            terminal.append(report_terminal)
            if report_terminal:
                self._skill_chain_terminal_reported[env_idx] = True
        return torch.tensor(terminal, dtype=torch.bool)

    def _attach_skill_chain_obs_fields(self, obs: dict | None) -> dict | None:
        if not self.skill_chain_enabled or obs is None:
            return obs
        obs["task_descriptions"] = [
            self._compose_task_description(self._skill_chain_current_prompt(env_idx))
            for env_idx in range(self.num_envs)
        ]
        obs["skill_chain_policy"] = [
            self._skill_chain_current_policy(env_idx) for env_idx in range(self.num_envs)
        ]
        obs["skill_chain_completed_count"] = [
            self._skill_chain_completed_count(env_idx) for env_idx in range(self.num_envs)
        ]
        obs["skill_chain_stage_idx"] = [
            self._skill_chain_stage_idx[env_idx] for env_idx in range(self.num_envs)
        ]
        obs["skill_chain_stage_count"] = [
            len(self._skill_chain_stage_names[env_idx]) for env_idx in range(self.num_envs)
        ]
        obs["skill_chain_stage_elapsed"] = [
            self._skill_chain_stage_elapsed[env_idx] for env_idx in range(self.num_envs)
        ]
        obs["skill_chain_stage_limit"] = [
            (
                self._skill_chain_stage_limits[env_idx][self._skill_chain_stage_idx[env_idx]]
                if self._skill_chain_stage_idx[env_idx]
                < len(self._skill_chain_stage_limits[env_idx])
                else 0
            )
            for env_idx in range(self.num_envs)
        ]
        obs["skill_chain_post_success_remaining"] = [
            self._skill_chain_post_success_remaining[env_idx]
            for env_idx in range(self.num_envs)
        ]
        return obs

    def _slice_actions_for_shards(self, actions):
        if actions is None:
            return [None] * self.num_env_subprocess
        s = self.env_shard_size
        return [
            actions[i * s : (i + 1) * s] for i in range(self.num_env_subprocess)
        ]

    def _merge_step_results(self, shard_results: list):
        raw_obs = []
        rewards_parts = []
        term_parts = []
        trunc_parts = []
        infos = []
        for raw_obs_s, rew_s, term_s, trunc_s, infos_s in shard_results:
            self._extend_per_env(raw_obs, raw_obs_s)
            rewards_parts.append(rew_s)
            term_parts.append(term_s)
            trunc_parts.append(trunc_s)
            self._extend_per_env(infos, infos_s)
        return (
            raw_obs,
            self._cat_batch_dim(rewards_parts),
            self._cat_batch_dim(term_parts),
            self._cat_batch_dim(trunc_parts),
            infos,
        )

    def _merge_chunk_results(self, shard_results: list):
        chunk_size = len(shard_results[0][0])
        merged_obs_lists = []
        merged_rewards = []
        merged_terms = []
        merged_trunc = []
        merged_infos = []
        for t in range(chunk_size):
            obs_t = []
            r_t = []
            term_t = []
            trunc_t = []
            info_t = []
            for (
                raw_obs_list,
                raw_rewards_list,
                raw_terminations_list,
                raw_truncations_list,
                raw_infos_list,
            ) in shard_results:
                self._extend_per_env(obs_t, raw_obs_list[t])
                r_t.append(raw_rewards_list[t])
                term_t.append(raw_terminations_list[t])
                trunc_t.append(raw_truncations_list[t])
                self._extend_per_env(info_t, raw_infos_list[t])
            merged_obs_lists.append(obs_t)
            merged_rewards.append(self._cat_batch_dim(r_t))
            merged_terms.append(self._cat_batch_dim(term_t))
            merged_trunc.append(self._cat_batch_dim(trunc_t))
            merged_infos.append(info_t)
        return (
            merged_obs_lists,
            merged_rewards,
            merged_terms,
            merged_trunc,
            merged_infos,
        )

    def _call_all_subprocs(self, cmd: str, payloads: list) -> list:
        """Send the same command to every shard; recv in parallel to avoid pipe backpressure."""
        n = len(self.parent_conn_list)
        assert len(payloads) == n, f"payloads length {len(payloads)} != num subprocesses {n}"
        for conn, payload in zip(self.parent_conn_list, payloads):
            conn.send((cmd, payload))

        results: list = [None] * n
        errors: list[str | None] = [None] * n

        def _recv_shard(i: int):
            msg = self.parent_conn_list[i].recv()
            if msg.get("type") == "error":
                errors[i] = msg["traceback"]
            else:
                results[i] = msg["result"]

        recv_threads = [
            Thread(target=_recv_shard, args=(i,), daemon=True) for i in range(n)
        ]
        for t in recv_threads:
            t.start()
        for t in recv_threads:
            t.join()

        err = next((e for e in errors if e is not None), None)
        if err is not None:
            raise RuntimeError(
                f"Behavior subprocess env failed on command '{cmd}':\n{err}"
            )
        return results

    def _call_subproc(self, cmd: str, payload=None):
        n = len(self.parent_conn_list)

        if cmd == "reset":
            shard_results = self._call_all_subprocs("reset", [None] * n)
            raw_obs = []
            infos = []
            for ro, inf in shard_results:
                self._extend_per_env(raw_obs, ro)
                self._extend_per_env(infos, inf)
            return (raw_obs, infos)
        if cmd == "step":
            payloads = self._slice_actions_for_shards(payload)
            shard_results = self._call_all_subprocs("step", payloads)
            return self._merge_step_results(shard_results)
        if cmd == "chunk_step":
            chunk_actions = payload["chunk_actions"]
            payloads = [
                {"chunk_actions": ca}
                for ca in self._slice_actions_for_shards(chunk_actions)
            ]
            shard_results = self._call_all_subprocs("chunk_step", payloads)
            return self._merge_chunk_results(shard_results)
        if cmd == "set_active_stage_indices":
            payloads = [
                payload[i * self.env_shard_size : (i + 1) * self.env_shard_size]
                for i in range(n)
            ]
            self._call_all_subprocs("set_active_stage_indices", payloads)
            return None
        if cmd == "close":
            self._call_all_subprocs("close", [None] * n)
            return None
        raise NotImplementedError(f"Unknown command: {cmd}")

    def _extract_obs_image(self, raw_obs):
        state = None
        for sensor_data in raw_obs.values():
            assert isinstance(sensor_data, dict)
            for k, v in sensor_data.items():
                if "left_realsense_link:Camera:0" in k:
                    left_image = convert_uint8_rgb(v["rgb"])
                elif "right_realsense_link:Camera:0" in k:
                    right_image = convert_uint8_rgb(v["rgb"])
                elif "zed_link:Camera:0" in k:
                    zed_image = convert_uint8_rgb(v["rgb"])
                elif "proprio" in k:
                    state = v
        assert state is not None, (
            "state is not found in the observation which is required for the behavior training."
        )

        return {
            "main_images": zed_image,  # [H, W, C]
            "wrist_images": torch.stack(
                [left_image, right_image], axis=0
            ),  # [N_IMG, H, W, C]
            "state": state,
        }

    def _update_stage_prompts_from_info(self, env_idx: int, info: dict | None) -> str | None:
        if not isinstance(info, dict):
            return self._current_stage_prompts[env_idx]

        stage_info = info
        reward_info = info.get("reward")
        if isinstance(reward_info, dict):
            task_specific_info = reward_info.get("task_specific")
            if isinstance(task_specific_info, dict):
                stage_info = task_specific_info

        replay_info = info.get("replay_init")
        if isinstance(replay_info, dict):
            replay_skill_prompt = replay_info.get("replay_skill_prompt")
            if replay_skill_prompt:
                prompt = str(replay_skill_prompt).strip()
                self._current_stage_prompts[env_idx] = prompt
                return prompt
            replay_prompts = replay_info.get("replay_stage_prompts")
            if isinstance(replay_prompts, (list, tuple)):
                self._stage_prompt_lists[env_idx] = [
                    str(prompt).strip()
                    for prompt in replay_prompts
                    if str(prompt).strip()
                ]

        explicit_prompt = (
            stage_info.get("current_stage_prompt")
            or stage_info.get("stage_prompt")
            or stage_info.get("subtask_prompt")
        )
        if explicit_prompt:
            prompt = str(explicit_prompt).strip()
            self._current_stage_prompts[env_idx] = prompt
            return prompt

        stage_idx = stage_info.get("current_stage_idx")
        if stage_idx is None:
            return self._current_stage_prompts[env_idx]
        try:
            stage_idx = int(stage_idx)
        except (TypeError, ValueError):
            return self._current_stage_prompts[env_idx]

        stage_prompts = self._stage_prompt_lists[env_idx]
        if not stage_prompts or stage_idx < 0 or stage_idx >= len(stage_prompts):
            return self._current_stage_prompts[env_idx]
        prompt = stage_prompts[stage_idx]
        self._current_stage_prompts[env_idx] = prompt
        return prompt

    def _compose_task_description(self, stage_prompt: str | None) -> str:
        if self.subtask_prompt_override is not None:
            return self.subtask_prompt_override
        if self.subtask_prompt_only and stage_prompt:
            return stage_prompt
        if not stage_prompt:
            return self.task_description
        return f"{self.task_description}\nCurrent stage: {stage_prompt}"

    def _task_descriptions_from_infos(self, infos=None) -> list[str]:
        if self.skill_chain_enabled:
            return [
                self._compose_task_description(self._skill_chain_current_prompt(env_idx))
                for env_idx in range(self.num_envs)
            ]
        if not self.use_subtask_prompt or infos is None:
            return [self.task_description for _ in range(self.num_envs)]
        return [
            self._compose_task_description(
                self._update_stage_prompts_from_info(env_idx, info)
            )
            for env_idx, info in enumerate(infos)
        ]

    def _wrap_obs(self, obs_list, infos=None):
        extracted_obs_list = []
        for obs in obs_list:
            extracted_obs = self._extract_obs_image(obs)
            extracted_obs_list.append(extracted_obs)

        obs = {
            "main_images": torch.stack(
                [obs["main_images"] for obs in extracted_obs_list], axis=0
            ),  # [N_ENV, H, W, C]
            "wrist_images": torch.stack(
                [obs["wrist_images"] for obs in extracted_obs_list], axis=0
            ),  # [N_ENV, N_IMG, H, W, C]
            "task_descriptions": self._task_descriptions_from_infos(infos),
            "states": torch.stack(
                [obs["state"] for obs in extracted_obs_list], axis=0
            ),  # [N_ENV, 32]
        }
        return self._attach_skill_chain_obs_fields(obs)

    def reset(self):
        raw_obs, infos = self._call_subproc("reset")
        self._reset_skill_chain_state(infos)
        self._sync_skill_chain_reward_stages()
        obs = self._wrap_obs(raw_obs, infos)
        rewards = torch.zeros(self.num_envs, dtype=bool)
        infos = self._record_metrics(rewards, infos)
        self._reset_metrics()
        return obs, infos

    def step(
        self, actions=None
    ) -> tuple[dict, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        if isinstance(actions, torch.Tensor):
            actions = actions.detach().cpu()
        raw_obs, rewards, terminations, truncations, infos = self._call_subproc(
            "step", actions
        )
        obs = self._wrap_obs(raw_obs, infos)
        rewards = self._calc_step_reward(rewards, infos)
        infos = self._record_metrics(rewards, infos)
        if self.ignore_terminations:
            terminations[:] = False

        return (
            obs,
            to_tensor(rewards),
            to_tensor(terminations),
            to_tensor(truncations),
            infos,
        )

    def chunk_step(self, chunk_actions):
        # chunk_actions: [num_envs, chunk_step, action_dim]
        if isinstance(chunk_actions, torch.Tensor):
            chunk_actions = chunk_actions.detach().cpu()
        (
            raw_obs_list,
            raw_rewards_list,
            raw_terminations_list,
            raw_truncations_list,
            raw_infos_list,
        ) = self._call_subproc(
            "chunk_step",
            {"chunk_actions": chunk_actions},
        )

        chunk_size = len(raw_obs_list)
        obs_list = []
        infos_list = []
        scaled_rewards_list = []
        for i in range(chunk_size):
            step_rewards = self._calc_step_reward(
                raw_rewards_list[i], raw_infos_list[i]
            )
            scaled_rewards_list.append(step_rewards)
            infos = self._record_metrics(
                step_rewards,
                raw_infos_list[i],
                allow_skill_chain_stage_transition=i == chunk_size - 1,
            )
            if self.ignore_terminations:
                raw_terminations_list[i] = torch.zeros_like(raw_terminations_list[i])
            raw_obs = raw_obs_list[i]
            if raw_obs is None or (
                isinstance(raw_obs, (list, tuple))
                and all(obs is None for obs in raw_obs)
            ):
                obs_list.append(None)
            else:
                obs_list.append(self._wrap_obs(raw_obs, raw_infos_list[i]))
            infos_list.append(infos)
            if self.skill_chain_enabled:
                obs_list[-1] = self._attach_skill_chain_obs_fields(obs_list[-1])
                raw_terminations_list[i] = torch.logical_or(
                    raw_terminations_list[i].bool(),
                    self._skill_chain_new_terminal_tensor(),
                )

        chunk_rewards = torch.stack(scaled_rewards_list, dim=1)  # [num_envs, chunk_steps]
        raw_terminations = torch.stack(
            raw_terminations_list, dim=1
        )  # [num_envs, chunk_steps]
        raw_truncations = torch.stack(
            raw_truncations_list, dim=1
        )  # [num_envs, chunk_steps]

        past_terminations = raw_terminations.any(dim=1)
        past_truncations = raw_truncations.any(dim=1)

        # Some OmniGibson builds may report episode completion primarily via
        # `info["done"]` while leaving `terminations`/`truncations` booleans
        # as all-False for the whole chunk. RLinf's evaluation metrics gate on
        # `terminations|truncations`, so we fall back to info-done here.
        #
        # `raw_infos_list[i]` is a list of per-env info dicts for chunk step i.
        info_done_flags = []
        for i in range(chunk_size):
            step_infos = raw_infos_list[i]
            step_done = [
                self._extract_info_done(info) if isinstance(info, dict) else False
                for info in step_infos
            ]
            info_done_flags.append(torch.tensor(step_done, dtype=torch.bool))
        past_info_dones = torch.stack(info_done_flags, dim=1).any(dim=1)

        # If the config asks to ignore terminations, map info-done into
        # truncations; otherwise map it into terminations.
        if self.ignore_terminations:
            past_truncations = torch.logical_or(past_truncations, past_info_dones)
        else:
            past_terminations = torch.logical_or(past_terminations, past_info_dones)
        past_dones = torch.logical_or(past_terminations, past_truncations)

        if past_dones.any() and self.auto_reset:
            obs_list[-1], infos_list[-1] = self._handle_auto_reset(
                past_dones, obs_list[-1], infos_list[-1]
            )

        chunk_terminations = torch.zeros_like(raw_terminations)
        chunk_terminations[:, -1] = past_terminations

        chunk_truncations = torch.zeros_like(raw_truncations)
        chunk_truncations[:, -1] = past_truncations
        return (
            obs_list,
            chunk_rewards,
            chunk_terminations,
            chunk_truncations,
            infos_list,
        )

    @property
    def device(self):
        return "cuda"

    @property
    def elapsed_steps(self):
        return torch.tensor(self.cfg.max_episode_steps)

    @property
    def is_start(self):
        return self._is_start

    @is_start.setter
    def is_start(self, value):
        self._is_start = value

    def _init_metrics(self):
        self.success_once = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )
        self.success_step = torch.full(
            (self.num_envs,), -1, device=self.device, dtype=torch.int32
        )
        self.returns = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float32
        )
        self.prev_step_reward = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float32
        )

    def _reset_metrics(self, env_idx=None):
        if env_idx is not None:
            mask = torch.zeros(self.num_envs, dtype=bool, device=self.device)
            mask[env_idx] = True
        else:
            mask = torch.ones(self.num_envs, dtype=bool, device=self.device)
        self.prev_step_reward[mask] = 0.0
        if self.record_metrics:
            self.success_once[mask] = False
            self.success_step[mask] = -1
            self.returns[mask] = 0

    def _update_skill_chain_metrics(
        self,
        env_idx: int,
        episode_length: int,
        skill_success: bool,
        allow_stage_transition: bool = True,
    ) -> dict[str, Any]:
        if not self.skill_chain_enabled:
            return {}

        names = self._skill_chain_stage_names[env_idx]
        stage_count = len(names)
        stage_changed = False
        if stage_count == 0:
            self._skill_chain_done[env_idx] = True
        elif (
            episode_length > 0
            and not self._skill_chain_done[env_idx]
            and not self._skill_chain_failed[env_idx]
        ):
            idx = self._skill_chain_stage_idx[env_idx]
            success_just_detected = False
            if skill_success and not self._skill_chain_pending_success[env_idx]:
                self._skill_chain_pending_success[env_idx] = True
                self._skill_chain_post_success_remaining[env_idx] = (
                    self._skill_chain_current_post_success_steps(env_idx)
                )
                success_just_detected = True

            if self._skill_chain_pending_success[env_idx] and not success_just_detected:
                self._skill_chain_post_success_remaining[env_idx] = max(
                    0,
                    int(self._skill_chain_post_success_remaining[env_idx]) - 1,
                )

            if (
                self._skill_chain_pending_success[env_idx]
                and self._skill_chain_post_success_remaining[env_idx] <= 0
                and allow_stage_transition
            ):
                self._skill_chain_completed[env_idx][idx] = True
                self._skill_chain_pending_success[env_idx] = False
                self._skill_chain_post_success_remaining[env_idx] = 0
                if idx + 1 >= stage_count:
                    self._skill_chain_done[env_idx] = True
                else:
                    self._skill_chain_stage_idx[env_idx] = idx + 1
                    self._skill_chain_stage_elapsed[env_idx] = 0
                    stage_changed = True
            elif not self._skill_chain_pending_success[env_idx]:
                self._skill_chain_stage_elapsed[env_idx] += 1
                limit = self._skill_chain_stage_limits[env_idx][idx]
                if self._skill_chain_stage_elapsed[env_idx] >= limit:
                    self._skill_chain_failed[env_idx] = True
                    self._skill_chain_failed_stage[env_idx] = idx
            else:
                self._skill_chain_stage_elapsed[env_idx] += 1

        current_idx = self._skill_chain_stage_idx[env_idx]
        chain_success = (
            self._skill_chain_done[env_idx] and not self._skill_chain_failed[env_idx]
        )
        return {
            "skill_chain_enabled": True,
            "skill_chain_success": chain_success,
            "skill_chain_failed": self._skill_chain_failed[env_idx],
            "skill_chain_done": self._skill_chain_done[env_idx],
            "skill_chain_stage_idx": current_idx,
            "skill_chain_stage_count": stage_count,
            "skill_chain_stage_elapsed": self._skill_chain_stage_elapsed[env_idx],
            "skill_chain_stage_limit": (
                self._skill_chain_stage_limits[env_idx][current_idx]
                if current_idx < stage_count
                else 0
            ),
            "skill_chain_completed_count": self._skill_chain_completed_count(env_idx),
            "skill_chain_failed_stage": self._skill_chain_failed_stage[env_idx],
            "skill_chain_post_success_remaining": self._skill_chain_post_success_remaining[
                env_idx
            ],
            "skill_chain_stage_changed": stage_changed,
        }

    def _record_metrics(
        self,
        rewards,
        infos,
        allow_skill_chain_stage_transition: bool = True,
    ):
        info_lists = []
        replay_info_lists = []
        sync_reward_stage = False
        for env_idx, (reward, info) in enumerate(zip(rewards, infos)):
            task_reward = self._get_task_specific_reward_info(info)
            distance_reward = self._get_distance_reward_info(info)
            completion_bonus = float(task_reward.get("completion_bonus", 0.0) or 0.0)
            stage_infos = task_reward.get("stage_infos", {})
            if not isinstance(stage_infos, dict):
                stage_infos = {}
            success_stage_info = self._get_success_stage_info(task_reward, stage_infos)
            success_stage_completed = bool(success_stage_info.get("completed", False))
            current_stage_name = task_reward.get("current_stage_name")
            stage_info = stage_infos.get("press_radio", {})
            if not isinstance(stage_info, dict) or (
                "button_normal_up" not in stage_info and "button_up_alignment" not in stage_info
            ):
                current_stage_info = (
                    stage_infos.get(current_stage_name, {}) if current_stage_name else {}
                )
                if (
                    isinstance(current_stage_info, dict)
                    and "press" in self._normalize_skill_text(current_stage_name)
                    and (
                        "button_normal_up" in current_stage_info
                        or "button_up_alignment" in current_stage_info
                    )
                ):
                    stage_info = current_stage_info
            radio_button_up = None
            radio_button_align = None
            if isinstance(stage_info, dict):
                button_normal_up = stage_info.get("button_normal_up")
                button_up_alignment = stage_info.get("button_up_alignment")
                if button_normal_up is not None:
                    radio_button_up = self._metric_truthy(button_normal_up)
                if button_up_alignment is not None:
                    radio_button_align = float(button_up_alignment)
                    if radio_button_up is None:
                        radio_button_up = self._metric_truthy(radio_button_align)
            target_distance = float(distance_reward.get("target_distance", float("nan")))
            body_target_distance = float(
                distance_reward.get("body_target_distance", float("nan"))
            )
            eef_target_distance = float(
                distance_reward.get("eef_target_distance", float("nan"))
            )
            target_object_found = bool(distance_reward.get("target_object_found", False))
            distance_success = (
                self.success_distance_threshold is not None
                and target_object_found
                and np.isfinite(target_distance)
                and target_distance <= self.success_distance_threshold
            )
            done_dict = info.get("done", {})
            step_success = done_dict.get("success", False)
            end_success = info.get("success", step_success)
            if self.success_metric in {"stage", "stage_completion"}:
                step_success = success_stage_completed
                end_success = success_stage_completed
            elif self.success_metric in {"distance", "target_distance", "distance_threshold"}:
                step_success = distance_success
                end_success = distance_success
            current_stage_idx = task_reward.get("current_stage_idx", -1)
            try:
                current_stage_idx_int = int(current_stage_idx)
            except (TypeError, ValueError):
                current_stage_idx_int = -1
            episode_length = info.get("episode_length", 0)
            try:
                episode_length_int = int(episode_length)
            except (TypeError, ValueError):
                episode_length_int = 0
            skill_chain_info = {}
            if self.skill_chain_enabled:
                skill_name = self._skill_chain_current_name(env_idx)
                normalized_skill_name = self._normalize_skill_text(skill_name)
                expected_stage_idx = self._skill_chain_current_reward_stage_index(env_idx)
                expected_stage_name = (
                    self._skill_chain_current_reward_stage_name(env_idx)
                    or self._skill_name_to_reward_stage_name(skill_name)
                )
                expected_stage_info = success_stage_info
                if expected_stage_name:
                    candidate_stage_info = stage_infos.get(expected_stage_name, {})
                    if isinstance(candidate_stage_info, dict):
                        expected_stage_info = candidate_stage_info
                elif expected_stage_idx is not None:
                    ordered_stage_infos = list(stage_infos.values())
                    expected_stage_idx_int = int(expected_stage_idx)
                    if 0 <= expected_stage_idx_int < len(ordered_stage_infos):
                        candidate_stage_info = ordered_stage_infos[expected_stage_idx_int]
                        if isinstance(candidate_stage_info, dict):
                            expected_stage_info = candidate_stage_info
                expected_stage_completed = bool(
                    expected_stage_info.get("completed", False)
                )
                if expected_stage_name:
                    active_expected_stage = (
                        current_stage_name == expected_stage_name
                        or expected_stage_completed
                    )
                else:
                    active_expected_stage = (
                        expected_stage_idx is None
                        or current_stage_idx_int == int(expected_stage_idx)
                        or expected_stage_completed
                    )
                if normalized_skill_name == "move to":
                    skill_success = distance_success
                elif normalized_skill_name.startswith("pick up"):
                    skill_success = self._pick_up_stage_success(
                        expected_stage_info,
                        expected_stage_completed,
                    )
                elif normalized_skill_name.startswith("press"):
                    skill_success = expected_stage_completed
                elif normalized_skill_name.startswith("place on"):
                    on_support = self._metric_truthy(expected_stage_info.get("on_support"))
                    skill_success = (
                        active_expected_stage
                        and self._skill_chain_prior_stage_completed(env_idx, "press")
                        and (on_support if on_support is not None else expected_stage_completed)
                    )
                else:
                    skill_success = active_expected_stage and expected_stage_completed
                skill_chain_info = self._update_skill_chain_metrics(
                    env_idx,
                    episode_length_int,
                    bool(skill_success),
                    allow_stage_transition=allow_skill_chain_stage_transition,
                )
                sync_reward_stage = sync_reward_stage or bool(
                    skill_chain_info.get("skill_chain_stage_changed", False)
                )
                skill_chain_info["skill_chain_skill_success"] = bool(skill_success)
                step_success = bool(skill_chain_info.get("skill_chain_success", False))
                end_success = step_success
            total_stage_count = task_reward.get("total_stage_count", 0)
            success_stage_in_hand = success_stage_info.get("in_hand", None)
            success_stage_on_support = success_stage_info.get("on_support", None)
            success_stage_distance = success_stage_info.get(
                "eef_to_obj_distance",
                success_stage_info.get(
                    "eef_to_target_distance",
                    success_stage_info.get("eef_to_container_distance", float("nan")),
                ),
            )
            episode_info = {
                "episode_length": episode_length,
                "completion_bonus": completion_bonus,
                "current_stage_idx": int(current_stage_idx if current_stage_idx is not None else -1),
                "total_stage_count": int(total_stage_count if total_stage_count is not None else 0),
                "success_stage_completed": success_stage_completed,
                "success_stage_in_hand": (
                    float(success_stage_in_hand)
                    if success_stage_in_hand is not None
                    else float("nan")
                ),
                "success_stage_on_support": (
                    float(success_stage_on_support)
                    if success_stage_on_support is not None
                    else float("nan")
                ),
                "success_stage_distance": float(success_stage_distance),
                "success_stage_threshold": float(
                    success_stage_info.get("success_threshold", float("nan"))
                ),
                "success_distance_threshold": float(
                    self.success_distance_threshold
                    if self.success_distance_threshold is not None
                    else float("nan")
                ),
                "radio_button_up": False,
                "radio_button_align": -1.0,
                "target_distance": target_distance,
                "body_target_distance": body_target_distance,
                "eef_target_distance": eef_target_distance,
                "distance_reward": float(
                    distance_reward.get("distance_reward", 0.0)
                ),
                "target_object_found": target_object_found,
            }
            episode_info.update(skill_chain_info)
            if radio_button_up is not None:
                episode_info["radio_button_up"] = bool(radio_button_up)
            if radio_button_align is not None:
                episode_info["radio_button_align"] = float(radio_button_align)
            self.returns[env_idx] += reward
            step_success_bool = bool(to_tensor(step_success).reshape(-1)[0].item())
            if step_success_bool and not bool(self.success_once[env_idx].item()):
                self.success_step[env_idx] = int(episode_length)
            self.success_once[env_idx] = self.success_once[env_idx] | step_success
            episode_info["success_once"] = self.success_once[env_idx].clone()
            episode_info["success_step"] = self.success_step[env_idx].clone()
            episode_info["success_at_end"] = end_success

            episode_info["return"] = self.returns[env_idx].clone()
            episode_info["episode_len"] = episode_length
            episode_info["reward"] = (
                episode_info["return"]
                / torch.clamp(to_tensor(episode_length), min=1).to(self.device)
            )

            info_lists.append(episode_info)
            if "replay_init" in info:
                replay_info_lists.append(
                    {
                        key: value
                        for key, value in info["replay_init"].items()
                        if isinstance(value, (bool, int, float))
                    }
                )

        if sync_reward_stage:
            self._sync_skill_chain_reward_stages()

        infos = {"episode": to_tensor(list_of_dict_to_dict_of_list(info_lists))}
        if replay_info_lists:
            infos["replay_init"] = to_tensor(
                list_of_dict_to_dict_of_list(replay_info_lists)
            )
        return infos

    def _get_success_stage_info(self, task_reward: dict | None, stage_infos: dict) -> dict:
        if self.success_stage_name:
            stage_info = stage_infos.get(self.success_stage_name, {})
            return stage_info if isinstance(stage_info, dict) else {}

        current_stage_name = (
            task_reward.get("current_stage_name") if isinstance(task_reward, dict) else None
        )
        stage_info = stage_infos.get(current_stage_name, {}) if current_stage_name else {}
        return stage_info if isinstance(stage_info, dict) else {}

    @staticmethod
    def _get_task_specific_reward_info(info: dict | None) -> dict:
        if not isinstance(info, dict):
            return {}
        if "stage_infos" in info or "current_stage_name" in info:
            return info
        reward_info = info.get("reward", {})
        if not isinstance(reward_info, dict):
            return {}
        task_reward = reward_info.get("task_specific")
        if isinstance(task_reward, dict) and (
            "stage_infos" in task_reward or "current_stage_name" in task_reward
        ):
            return task_reward
        for reward_payload in reward_info.values():
            if isinstance(reward_payload, dict) and (
                "stage_infos" in reward_payload
                or "current_stage_name" in reward_payload
            ):
                return reward_payload
        return {}

    @staticmethod
    def _get_distance_reward_info(info: dict | None) -> dict:
        if not isinstance(info, dict):
            return {}
        reward_info = info.get("reward", {})
        if not isinstance(reward_info, dict):
            return {}
        distance_reward = reward_info.get("distance", {})
        return distance_reward if isinstance(distance_reward, dict) else {}

    @staticmethod
    def _extract_info_done(info: dict) -> bool:
        tc = info["done"]["termination_conditions"]
        return any(v["done"] for v in tc.values())

    def _handle_auto_reset(self, dones, extracted_obs, infos):
        final_obs = extracted_obs.copy()
        env_idx = torch.arange(0, self.num_envs, device=self.device)[dones]
        options = {"env_idx": env_idx}
        final_info = infos.copy()
        if self.use_fixed_reset_state_ids:
            options.update(episode_id=self.reset_state_ids[env_idx])
        extracted_obs, infos = self.reset()
        # gymnasium calls it final observation but it really is just o_{t+1} or the true next observation
        infos["final_observation"] = final_obs
        infos["final_info"] = final_info
        infos["_final_info"] = dones
        infos["_final_observation"] = dones
        infos["_elapsed_steps"] = dones
        return extracted_obs, infos

    def update_reset_state_ids(self):
        # use for multi task training
        pass

    def close(self):
        if not self.parent_conn_list:
            return
        try:
            self._call_subproc("close")
        except Exception:
            pass
        finally:
            for proc in self.env_process_list:
                if getattr(proc, "is_alive", lambda: False)():
                    proc.join(timeout=2)
                    if proc.is_alive() and not self.use_thread_worker:
                        proc.terminate()
            for ch in self.child_conn_list:
                if ch is not None:
                    try:
                        ch.close()
                    except Exception:
                        pass
            for pc in self.parent_conn_list:
                try:
                    pc.close()
                except Exception:
                    pass
            self.child_conn_list.clear()
            self.parent_conn_list.clear()
            self.env_process_list.clear()

    def _completion_bonus_tensor(self, infos, reward):
        bonuses = []
        for info in infos or [{} for _ in range(self.num_envs)]:
            reward_info = info.get("reward", {}) if isinstance(info, dict) else {}
            task_reward = reward_info.get("task_specific", {})
            bonuses.append(float(task_reward.get("completion_bonus", 0.0) or 0.0))
        return self.reward_coef * torch.as_tensor(
            bonuses, dtype=reward.dtype, device=reward.device
        )

    def _calc_step_reward(self, rewards, infos=None):
        reward = self.reward_coef * rewards
        if not self.use_rel_reward:
            return reward

        completion_bonus = self._completion_bonus_tensor(infos, reward)
        dense_reward = reward - completion_bonus
        reward_diff = dense_reward - self.prev_step_reward.to(dense_reward.device)
        self.prev_step_reward = dense_reward.to(self.prev_step_reward.device)
        return reward_diff + completion_bonus
