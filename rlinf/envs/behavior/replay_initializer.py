# Copyright 2026 The RLinf Authors.
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

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import DictConfig, ListConfig, OmegaConf


@dataclass(frozen=True)
class ReplayEpisode:
    episode_index: int
    instance_id: int
    parquet_path: Path
    annotation_path: Path | None
    orchestrator_path: Path | None
    length: int | None = None
    skill_occurrence: int | None = None


@dataclass(frozen=True)
class ReplayPlan:
    episode_index: int
    instance_id: int
    skill_occurrence: int | None
    target_stage_index: int | None
    actions: np.ndarray
    replay_steps: int
    target_step: int
    stage_prompts: tuple[str, ...] = ()
    stage_skill_names: tuple[str, ...] = ()
    stage_indices: tuple[int, ...] = ()
    stage_step_limits: tuple[int, ...] = ()
    stage_object_ids: tuple[str | None, ...] = ()
    stage_object_names: tuple[str | None, ...] = ()
    skill_prompt: str | None = None
    target_object_id: str | None = None
    target_object_name: str | None = None


class BehaviorReplayInitializer:
    """Replay BEHAVIOR demonstration prefixes before policy rollout starts."""

    def __init__(self, cfg: DictConfig, seed_offset: int = 0):
        replay_cfg = OmegaConf.select(cfg, "replay_init")
        self.enabled = bool(
            replay_cfg is not None and OmegaConf.select(replay_cfg, "enabled", default=False)
        )
        if not self.enabled:
            return

        self.use_subtask_prompt = bool(
            OmegaConf.select(cfg, "use_subtask_prompt", default=False)
        )
        self.dataset_root = Path(
            OmegaConf.select(replay_cfg, "dataset_root", default="")
        ).expanduser()
        if not self.dataset_root.is_dir():
            raise ValueError(
                f"env.replay_init.dataset_root must be an existing directory, got {self.dataset_root}"
            )

        self.task_id = int(OmegaConf.select(replay_cfg, "task_id", default=0))
        self.task_dir = f"task-{self.task_id:04d}"
        self.action_column = str(
            OmegaConf.select(replay_cfg, "action_column", default="action")
        )
        self.stage_index = OmegaConf.select(replay_cfg, "stage_index", default=None)
        if self.stage_index is not None:
            self.stage_index = int(self.stage_index)
        self.skill_name = OmegaConf.select(replay_cfg, "skill_name", default=None)
        if self.skill_name is not None:
            self.skill_name = str(self.skill_name).strip()
        raw_skill_occurrence = OmegaConf.select(
            replay_cfg, "skill_occurrence", default=0
        )
        self.all_skill_occurrences = (
            isinstance(raw_skill_occurrence, str)
            and raw_skill_occurrence.strip().lower() in ("all", "*")
        )
        if self.all_skill_occurrences:
            if not self.skill_name:
                raise ValueError(
                    "env.replay_init.skill_occurrence='all' requires skill_name."
                )
            self.skill_occurrence = 0
        else:
            self.skill_occurrence = int(raw_skill_occurrence)
            if self.skill_occurrence < 0:
                raise ValueError("env.replay_init.skill_occurrence must be non-negative.")
        if self.stage_index is not None and self.skill_name:
            raise ValueError(
                "env.replay_init should set only one of stage_index or skill_name."
            )
        self.sample_mode = str(
            OmegaConf.select(
                replay_cfg,
                "sample_mode",
                default="sequential" if self.all_skill_occurrences else "random",
            )
        ).lower()
        if self.sample_mode not in ("random", "sequential"):
            raise ValueError(
                "env.replay_init.sample_mode must be 'random' or 'sequential', "
                f"got {self.sample_mode!r}."
            )
        self._next_episode_idx = 0
        self.stage_boundary = str(
            OmegaConf.select(replay_cfg, "stage_boundary", default="start")
        ).lower()
        if self.stage_boundary not in ("start", "end", "previous_end"):
            raise ValueError(
                "env.replay_init.stage_boundary must be 'start', 'end', or 'previous_end'."
            )
        self.target_step = OmegaConf.select(replay_cfg, "target_step", default=None)
        if self.target_step is not None:
            self.target_step = int(self.target_step)

        self.replay_ratio = float(
            OmegaConf.select(replay_cfg, "replay_ratio", default=1.0)
        )
        if self.replay_ratio < 0:
            raise ValueError("env.replay_init.replay_ratio must be non-negative.")
        self.min_replay_steps = int(
            OmegaConf.select(replay_cfg, "min_replay_steps", default=0)
        )
        self.max_replay_steps = OmegaConf.select(
            replay_cfg, "max_replay_steps", default=None
        )
        if self.max_replay_steps is not None:
            self.max_replay_steps = int(self.max_replay_steps)
        self.jitter_steps = int(OmegaConf.select(replay_cfg, "jitter_steps", default=0))

        self.noise_std = float(OmegaConf.select(replay_cfg, "noise_std", default=0.0))
        self.noise_clip = OmegaConf.select(replay_cfg, "noise_clip", default=None)
        if self.noise_clip is not None:
            self.noise_clip = float(self.noise_clip)
        self.action_clip = OmegaConf.select(replay_cfg, "action_clip", default=None)
        if isinstance(self.action_clip, ListConfig):
            self.action_clip = OmegaConf.to_container(self.action_clip)
        if self.action_clip is not None:
            if len(self.action_clip) != 2:
                raise ValueError("env.replay_init.action_clip must be [low, high].")
            self.action_clip = (float(self.action_clip[0]), float(self.action_clip[1]))

        try:
            seed_value = OmegaConf.select(replay_cfg, "seed", default=None)
        except Exception:
            seed_value = None
        if seed_value is None:
            seed_value = cfg.get("seed", 0)
        seed = int(seed_value) + int(seed_offset)
        self._py_rng = random.Random(seed)
        self._np_rng = np.random.default_rng(seed)
        self._action_cache: dict[int, np.ndarray] = {}
        self.allowed_instance_ids = self._resolve_allowed_instance_ids(cfg)

        requested_episode_ids = OmegaConf.select(
            replay_cfg, "episode_ids", default=None
        )
        if isinstance(requested_episode_ids, ListConfig):
            requested_episode_ids = OmegaConf.to_container(requested_episode_ids)
        if requested_episode_ids is not None:
            requested_episode_ids = [int(x) for x in requested_episode_ids]

        self.episodes = self._discover_episodes(requested_episode_ids)
        if not self.episodes:
            raise ValueError(
                f"No replay episodes found under {self.dataset_root / 'data' / self.task_dir}."
            )

    def sample_plans(self, num_envs: int) -> list[ReplayPlan]:
        if self.sample_mode == "sequential":
            episodes = [
                self.episodes[(self._next_episode_idx + idx) % len(self.episodes)]
                for idx in range(num_envs)
            ]
            self._next_episode_idx = (
                self._next_episode_idx + num_envs
            ) % len(self.episodes)
        else:
            episodes = [self._py_rng.choice(self.episodes) for _ in range(num_envs)]
        return [self._build_plan(episode) for episode in episodes]

    def sample_grouped_plans(self, num_envs: int, group_size: int) -> list[ReplayPlan]:
        if group_size <= 1:
            return self.sample_plans(num_envs)
        if num_envs % group_size != 0:
            raise ValueError(
                f"num_envs={num_envs} must be divisible by group_size={group_size} "
                "for grouped BEHAVIOR replay initialization."
            )
        group_plans = self.sample_plans(num_envs // group_size)
        plans: list[ReplayPlan] = []
        for plan in group_plans:
            plans.extend([plan] * group_size)
        return plans

    def _discover_episodes(
        self, requested_episode_ids: list[int] | None
    ) -> list[ReplayEpisode]:
        data_dir = self.dataset_root / "data" / self.task_dir
        annotation_dir = self.dataset_root / "annotations" / self.task_dir
        orchestrator_dir = self.dataset_root / "orchestrators" / self.task_dir
        meta_by_episode = self._load_episode_lengths()

        episodes = []
        requested = set(requested_episode_ids) if requested_episode_ids else None
        for parquet_path in sorted(data_dir.glob("episode_*.parquet")):
            episode_index = self._parse_episode_index(parquet_path.stem)
            if requested is not None and episode_index not in requested:
                continue
            instance_id = self._episode_index_to_instance_id(episode_index)
            if (
                self.allowed_instance_ids is not None
                and instance_id not in self.allowed_instance_ids
            ):
                continue
            annotation_path = annotation_dir / f"episode_{episode_index:08d}.json"
            orchestrator_path = (
                orchestrator_dir
                / f"episode_{episode_index:08d}"
                / "task_annotated.json"
            )
            episode = ReplayEpisode(
                episode_index=episode_index,
                instance_id=instance_id,
                parquet_path=parquet_path,
                annotation_path=annotation_path if annotation_path.is_file() else None,
                orchestrator_path=orchestrator_path if orchestrator_path.is_file() else None,
                length=meta_by_episode.get(episode_index),
            )
            if self.all_skill_occurrences:
                for occurrence in self._matching_skill_occurrences(episode):
                    episodes.append(
                        ReplayEpisode(
                            episode_index=episode.episode_index,
                            instance_id=episode.instance_id,
                            parquet_path=episode.parquet_path,
                            annotation_path=episode.annotation_path,
                            orchestrator_path=episode.orchestrator_path,
                            length=episode.length,
                            skill_occurrence=occurrence,
                        )
                    )
            else:
                episodes.append(episode)
        return episodes

    def _matching_skill_occurrences(self, episode: ReplayEpisode) -> list[int]:
        if episode.annotation_path is None:
            return []
        with episode.annotation_path.open("r", encoding="utf-8") as f:
            annotation = json.load(f)
        target = self._normalize_skill_text(self.skill_name)
        occurrences = []
        occurrence = 0
        for skill in annotation.get("skill_annotation", []):
            descriptions = skill.get("skill_description", [])
            if isinstance(descriptions, str):
                descriptions = [descriptions]
            if any(
                self._skill_name_matches(target, str(description))
                for description in descriptions
            ):
                occurrences.append(occurrence)
                occurrence += 1
        return occurrences

    @staticmethod
    def _resolve_allowed_instance_ids(cfg: DictConfig) -> set[int] | None:
        instance_ids = OmegaConf.select(
            cfg, "omni_config.task.activity_instance_id", default=None
        )
        if isinstance(instance_ids, ListConfig):
            instance_ids = OmegaConf.to_container(instance_ids, resolve=True)
        if isinstance(instance_ids, int):
            return {int(instance_ids)}
        if isinstance(instance_ids, (list, tuple)):
            allowed = set()
            for item in instance_ids:
                if isinstance(item, int):
                    allowed.add(int(item))
            return allowed or None
        return None

    def _load_episode_lengths(self) -> dict[int, int]:
        path = self.dataset_root / "meta" / "episodes.jsonl"
        if not path.is_file():
            return {}
        lengths = {}
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                episode_index = int(item["episode_index"])
                if episode_index // 10000 == self.task_id:
                    lengths[episode_index] = int(item.get("length", 0))
        return lengths

    def _build_plan(self, episode: ReplayEpisode) -> ReplayPlan:
        actions = self._load_actions(episode)
        target_stage_index = self._resolve_plan_stage_index(episode)
        target_step = self._resolve_target_step(episode, len(actions))
        replay_steps = int(round(target_step * self.replay_ratio))
        if self.jitter_steps > 0:
            replay_steps += self._py_rng.randint(-self.jitter_steps, self.jitter_steps)
        replay_steps = max(self.min_replay_steps, replay_steps)
        if self.max_replay_steps is not None:
            replay_steps = min(self.max_replay_steps, replay_steps)
        replay_steps = min(max(replay_steps, 0), len(actions))

        replay_actions = actions[:replay_steps].copy()
        if replay_actions.size and self.noise_std > 0:
            noise = self._np_rng.normal(
                loc=0.0, scale=self.noise_std, size=replay_actions.shape
            ).astype(replay_actions.dtype, copy=False)
            if self.noise_clip is not None:
                noise = np.clip(noise, -self.noise_clip, self.noise_clip)
            replay_actions += noise
        if replay_actions.size and self.action_clip is not None:
            replay_actions = np.clip(
                replay_actions, self.action_clip[0], self.action_clip[1]
            )

        stage_prompts = self._load_stage_prompts(episode) if self.use_subtask_prompt else ()
        stage_summaries = self._load_stage_summaries(episode, stage_prompts)
        target_object_id = self._load_skill_target_object_id(episode)
        target_object_name = self._load_skill_target_object_name(episode)
        if target_object_id is None and stage_summaries["object_ids"]:
            target_object_id = stage_summaries["object_ids"][0]
            target_object_name = stage_summaries["object_names"][0]

        return ReplayPlan(
            episode_index=episode.episode_index,
            instance_id=episode.instance_id,
            skill_occurrence=episode.skill_occurrence,
            target_stage_index=target_stage_index,
            actions=replay_actions,
            replay_steps=replay_steps,
            target_step=target_step,
            stage_prompts=stage_prompts,
            stage_skill_names=stage_summaries["skill_names"],
            stage_indices=stage_summaries["stage_indices"],
            stage_step_limits=stage_summaries["step_limits"],
            stage_object_ids=stage_summaries["object_ids"],
            stage_object_names=stage_summaries["object_names"],
            skill_prompt=self._load_skill_prompt(episode) if self.skill_name else None,
            target_object_id=target_object_id,
            target_object_name=target_object_name,
        )

    def _load_actions(self, episode: ReplayEpisode) -> np.ndarray:
        cached = self._action_cache.get(episode.episode_index)
        if cached is not None:
            return cached

        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise ImportError(
                "env.replay_init requires pyarrow to read BEHAVIOR parquet files."
            ) from exc

        table = pq.read_table(episode.parquet_path, columns=[self.action_column])
        values = table[self.action_column].to_pylist()
        actions = np.asarray(values, dtype=np.float32)
        if actions.ndim != 2:
            raise ValueError(
                f"Replay action column {self.action_column!r} in {episode.parquet_path} "
                f"must be 2-D after loading, got shape {actions.shape}."
            )
        self._action_cache[episode.episode_index] = actions
        return actions

    def _resolve_target_step(self, episode: ReplayEpisode, action_count: int) -> int:
        if self.target_step is not None:
            return min(max(self.target_step, 0), action_count)
        if self.stage_index is not None or self.skill_name:
            return min(max(self._load_stage_step(episode), 0), action_count)
        if episode.length is not None and episode.length > 0:
            return min(episode.length, action_count)
        return action_count

    def _resolve_plan_stage_index(self, episode: ReplayEpisode) -> int | None:
        if not (self.stage_index is not None or self.skill_name):
            return None
        _, idx = self._load_stage_entry_with_index(episode)
        return idx

    def _load_stage_step(self, episode: ReplayEpisode) -> int:
        skill, idx = self._load_stage_entry_with_index(episode)
        duration = skill.get("frame_duration")
        if not isinstance(duration, list) or len(duration) != 2:
            raise ValueError(
                f"Invalid frame_duration for stage {self.stage_index} in {episode.annotation_path}."
            )
        if self.stage_boundary == "previous_end":
            if idx <= 0:
                return int(duration[0])
            previous_skill = self._load_stage_entry_by_index(episode, idx - 1)
            previous_duration = previous_skill.get("frame_duration")
            if not isinstance(previous_duration, list) or len(previous_duration) != 2:
                raise ValueError(
                    f"Invalid frame_duration for previous stage {idx - 1} in {episode.annotation_path}."
                )
            return int(previous_duration[1])
        return int(duration[0] if self.stage_boundary == "start" else duration[1])

    def _load_stage_entry_by_index(
        self, episode: ReplayEpisode, idx: int
    ) -> dict[str, Any]:
        if episode.annotation_path is None:
            raise ValueError(
                "stage_index/skill_name was set but no annotation file exists for "
                f"episode {episode.episode_index}."
            )
        with episode.annotation_path.open("r", encoding="utf-8") as f:
            annotation = json.load(f)
        skills = annotation.get("skill_annotation", [])
        if not skills:
            raise ValueError(f"No skill_annotation entries in {episode.annotation_path}.")
        if idx < 0:
            idx = len(skills) + idx
        if idx < 0 or idx >= len(skills):
            raise ValueError(
                f"stage_index={idx} is out of range for "
                f"{episode.annotation_path} with {len(skills)} stages."
            )
        return skills[idx]

    def _load_stage_entry(self, episode: ReplayEpisode) -> dict[str, Any]:
        if episode.annotation_path is None:
            raise ValueError(
                "stage_index/skill_name was set but no annotation file exists for "
                f"episode {episode.episode_index}."
            )
        with episode.annotation_path.open("r", encoding="utf-8") as f:
            annotation = json.load(f)
        skills = annotation.get("skill_annotation", [])
        if not skills:
            raise ValueError(f"No skill_annotation entries in {episode.annotation_path}.")
        idx = self._resolve_stage_index(skills, episode)
        if idx < 0:
            idx = len(skills) + idx
        if idx < 0 or idx >= len(skills):
            raise ValueError(
                f"stage_index={self.stage_index} is out of range for "
                f"{episode.annotation_path} with {len(skills)} stages."
            )
        return skills[idx]

    def _load_skill_prompt(self, episode: ReplayEpisode) -> str | None:
        skill, idx = self._load_stage_entry_with_index(episode)
        stage_prompts = self._load_stage_prompts(episode)
        if 0 <= idx < len(stage_prompts):
            return self._skill_prompt_with_object(stage_prompts[idx], skill)
        return self._build_skill_prompt(skill)

    def _load_stage_entry_with_index(
        self, episode: ReplayEpisode
    ) -> tuple[dict[str, Any], int]:
        if episode.annotation_path is None:
            raise ValueError(
                "stage_index/skill_name was set but no annotation file exists for "
                f"episode {episode.episode_index}."
            )
        with episode.annotation_path.open("r", encoding="utf-8") as f:
            annotation = json.load(f)
        skills = annotation.get("skill_annotation", [])
        if not skills:
            raise ValueError(f"No skill_annotation entries in {episode.annotation_path}.")
        idx = self._resolve_stage_index(skills, episode)
        if idx < 0:
            idx = len(skills) + idx
        if idx < 0 or idx >= len(skills):
            raise ValueError(
                f"stage_index={self.stage_index} is out of range for "
                f"{episode.annotation_path} with {len(skills)} stages."
            )
        return skills[idx], idx

    def _build_skill_prompt(self, skill: dict[str, Any]) -> str | None:
        descriptions = skill.get("skill_description", [])
        if isinstance(descriptions, str):
            descriptions = [descriptions]
        description = str(descriptions[0]).strip() if descriptions else self.skill_name
        return self._skill_prompt_with_object(description, skill)

    def _skill_prompt_with_object(
        self, description: str | None, skill: dict[str, Any]
    ) -> str | None:
        description = str(description).strip() if description else ""
        if not description:
            return None

        object_name = self._first_object_name(skill.get("object_id"))
        normalized_description = self._normalize_skill_text(description)
        normalized_object_name = self._normalize_skill_text(object_name)
        if object_name and normalized_object_name not in normalized_description:
            return f"{description} {object_name}"
        return description

    def _load_skill_target_object_id(self, episode: ReplayEpisode) -> str | None:
        if not (self.stage_index is not None or self.skill_name):
            return None
        skill = self._load_stage_entry(episode)
        return self._first_object_id(skill.get("object_id"))

    def _load_skill_target_object_name(self, episode: ReplayEpisode) -> str | None:
        object_id = self._load_skill_target_object_id(episode)
        return self._object_id_to_name(object_id) if object_id else None

    def _first_object_id(self, object_ids: Any) -> str | None:
        if isinstance(object_ids, str):
            return str(object_ids).strip() or None
        if not isinstance(object_ids, (list, tuple)):
            return None
        for item in object_ids:
            object_id = self._first_object_id(item)
            if object_id:
                return object_id
        return None

    def _first_object_name(self, object_ids: Any) -> str | None:
        if isinstance(object_ids, str):
            return self._object_id_to_name(object_ids)
        if not isinstance(object_ids, (list, tuple)):
            return None
        for item in object_ids:
            name = self._first_object_name(item)
            if name:
                return name
        return None

    @staticmethod
    def _object_id_to_name(object_id: str) -> str | None:
        parts = [part for part in str(object_id).split("_") if part]
        numeric_suffixes = []
        while parts and parts[-1].isdigit():
            numeric_suffixes.append(parts[-1])
            parts.pop()

        # Some BEHAVIOR scene object ids include a random model token before the
        # instance id, e.g. fridge_dszchb_0. Keep real category words such as
        # digital_camera_87, scrub_brush_86, and cutting_board_76 intact.
        if (
            numeric_suffixes
            and numeric_suffixes[0] == "0"
            and len(parts) > 1
            and parts[-1].isalpha()
            and len(parts[-1]) == 6
        ):
            parts.pop()
        return " ".join(parts) if parts else None

    def _resolve_stage_index(self, skills: list[dict[str, Any]], episode: ReplayEpisode) -> int:
        if self.stage_index is not None:
            return self.stage_index

        target = self._normalize_skill_text(self.skill_name)
        selected_occurrence = (
            self.skill_occurrence
            if episode.skill_occurrence is None
            else episode.skill_occurrence
        )
        occurrence = 0
        for idx, skill in enumerate(skills):
            descriptions = skill.get("skill_description", [])
            if isinstance(descriptions, str):
                descriptions = [descriptions]
            if any(
                self._skill_name_matches(target, str(description))
                for description in descriptions
            ):
                if occurrence == selected_occurrence:
                    return idx
                occurrence += 1

        available = []
        for skill in skills:
            descriptions = skill.get("skill_description", [])
            if isinstance(descriptions, str):
                descriptions = [descriptions]
            available.extend(str(desc) for desc in descriptions)
        raise ValueError(
            f"Could not find skill_name={self.skill_name!r} in "
            f"{episode.annotation_path}. Available skills: {available}"
        )

    @staticmethod
    def _normalize_skill_text(text: str | None) -> str:
        text = "" if text is None else str(text).lower().replace("_", " ")
        return re.sub(r"[^a-z0-9]+", " ", text).strip()

    @classmethod
    def _skill_name_matches(cls, target: str, candidate: str) -> bool:
        candidate = cls._normalize_skill_text(candidate)
        return (
            candidate == target
            or candidate.startswith(f"{target} ")
            or target in candidate.split()
        )

    @staticmethod
    def _load_stage_prompts(episode: ReplayEpisode) -> tuple[str, ...]:
        if episode.orchestrator_path is None:
            return ()
        with episode.orchestrator_path.open("r", encoding="utf-8") as f:
            orchestrator = json.load(f)
        prompts = orchestrator.get("cot_subtask_description_list", [])
        if not isinstance(prompts, list):
            return ()
        return tuple(str(prompt).strip() for prompt in prompts if str(prompt).strip())

    def _load_stage_summaries(
        self, episode: ReplayEpisode, stage_prompts: tuple[str, ...]
    ) -> dict[str, tuple]:
        if episode.annotation_path is None:
            return {
                "skill_names": (),
                "stage_indices": (),
                "step_limits": (),
                "object_ids": (),
                "object_names": (),
            }
        with episode.annotation_path.open("r", encoding="utf-8") as f:
            annotation = json.load(f)
        skills = annotation.get("skill_annotation", [])
        if not isinstance(skills, list):
            skills = []

        skill_names = []
        stage_indices = []
        step_limits = []
        object_ids = []
        object_names = []
        for stage_idx, skill in enumerate(skills):
            descriptions = skill.get("skill_description", [])
            if isinstance(descriptions, str):
                descriptions = [descriptions]
            description = str(descriptions[0]).strip() if descriptions else ""
            skill_names.append(description)
            stage_indices.append(stage_idx)

            duration = skill.get("frame_duration")
            if isinstance(duration, list) and len(duration) == 2:
                step_limits.append(max(1, int(duration[1]) - int(duration[0])))
            else:
                step_limits.append(1)

            object_id = self._first_object_id(skill.get("object_id"))
            object_ids.append(object_id)
            object_names.append(self._object_id_to_name(object_id) if object_id else None)

        return {
            "skill_names": tuple(skill_names),
            "stage_indices": tuple(stage_indices),
            "step_limits": tuple(step_limits),
            "object_ids": tuple(object_ids),
            "object_names": tuple(object_names),
        }

    def _episode_index_to_instance_id(self, episode_index: int) -> int:
        task_offset = self.task_id * 10000
        within_task_index = episode_index - task_offset
        if within_task_index <= 0:
            raise ValueError(
                f"episode_index={episode_index} does not belong to task_id={self.task_id}."
            )
        if within_task_index % 10 == 0:
            return within_task_index // 10
        return within_task_index

    @staticmethod
    def _parse_episode_index(stem: str) -> int:
        prefix = "episode_"
        if not stem.startswith(prefix):
            raise ValueError(f"Invalid BEHAVIOR episode filename stem: {stem}")
        return int(stem[len(prefix) :])


def maybe_make_replay_initializer(
    cfg: DictConfig, seed_offset: int = 0
) -> BehaviorReplayInitializer | None:
    initializer = BehaviorReplayInitializer(cfg, seed_offset=seed_offset)
    return initializer if initializer.enabled else None


def replay_plans_to_infos(plans: list[ReplayPlan]) -> list[dict[str, Any]]:
    return [
        {
            "replay_episode_index": plan.episode_index,
            "replay_instance_id": plan.instance_id,
            "replay_skill_occurrence": plan.skill_occurrence,
            "replay_target_stage_index": plan.target_stage_index,
            "replay_steps": plan.replay_steps,
            "replay_target_step": plan.target_step,
            "replay_stage_prompts": list(plan.stage_prompts),
            "replay_stage_skill_names": list(plan.stage_skill_names),
            "replay_stage_indices": list(plan.stage_indices),
            "replay_stage_step_limits": list(plan.stage_step_limits),
            "replay_stage_object_ids": list(plan.stage_object_ids),
            "replay_stage_object_names": list(plan.stage_object_names),
            "replay_skill_prompt": plan.skill_prompt,
            "replay_target_object_id": plan.target_object_id,
            "replay_target_object_name": plan.target_object_name,
        }
        for plan in plans
    ]
