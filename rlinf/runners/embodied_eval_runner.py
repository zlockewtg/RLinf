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

import json
import os
import re
import typing

import numpy as np
import torch
from omegaconf import OmegaConf

from rlinf.scheduler import Channel
from rlinf.scheduler import WorkerGroupFuncResult as Handle
from rlinf.utils.distributed import ScopedTimer
from rlinf.utils.logging import get_logger
from rlinf.utils.metric_logger import MetricLogger
from rlinf.utils.metric_utils import compute_evaluate_metrics

if typing.TYPE_CHECKING:
    from omegaconf.dictconfig import DictConfig

    from rlinf.workers.env.env_worker import EnvWorker
    from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker


class EmbodiedEvalRunner:
    def __init__(
        self,
        cfg: "DictConfig",
        rollout: "MultiStepRolloutWorker",
        env: "EnvWorker",
        run_timer=None,
    ):
        self.cfg = cfg
        self.rollout = rollout
        self.env = env

        # Data channels
        self.env_channel = Channel.create("Env")
        self.rollout_channel = Channel.create("Rollout")

        # this timer checks if we should stop training
        self.run_timer = run_timer

        self.timer = ScopedTimer(reduction="max", sync_cuda=False)
        self.metric_logger = MetricLogger(cfg)

        self.logger = get_logger()
        self._last_eval_metrics_list = []

    def init_workers(self):
        rollout_handle = self.rollout.init_worker()
        env_handle = self.env.init_worker()

        rollout_handle.wait()
        env_handle.wait()

    def evaluate(self):
        env_handle: Handle = self.env.evaluate(
            input_channel=self.env_channel,
            rollout_channel=self.rollout_channel,
        )
        rollout_handle: Handle = self.rollout.evaluate(
            input_channel=self.rollout_channel,
            output_channel=self.env_channel,
        )
        env_results = env_handle.wait()
        rollout_handle.wait()
        eval_metrics_list = [results for results in env_results if results is not None]
        self._last_eval_metrics_list = eval_metrics_list
        eval_metrics = compute_evaluate_metrics(eval_metrics_list)
        return eval_metrics

    @staticmethod
    def _as_scalar(value):
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu()
            return value.item() if value.numel() == 1 else value.tolist()
        if isinstance(value, np.ndarray):
            return value.item() if value.size == 1 else value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        return value

    @staticmethod
    def _concat_metric(eval_metrics_list, key: str) -> list:
        values = []
        for metrics in eval_metrics_list:
            if key not in metrics:
                continue
            value = metrics[key]
            if isinstance(value, torch.Tensor):
                values.extend(value.detach().cpu().reshape(-1).tolist())
            else:
                values.extend(torch.as_tensor(value).reshape(-1).tolist())
        return values

    def _load_behavior_task_prompt(self, activity_name: str | None) -> str | None:
        if not activity_name:
            return None
        task_path = os.path.normpath(
            os.path.join(
                os.path.dirname(__file__),
                "..",
                "envs",
                "behavior",
                "behavior_task.jsonl",
            )
        )
        try:
            with open(task_path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    item = json.loads(line)
                    if item.get("task_name") == activity_name:
                        return item.get("task")
        except OSError:
            return None
        return None

    @classmethod
    def _load_behavior_skill_prompt(
        cls,
        dataset_root: str | None,
        task_id: int | None,
        skill_name: str | None,
        skill_occurrence: int = 0,
    ) -> str | None:
        if dataset_root is None or task_id is None or skill_name is None:
            return None
        dataset_root = os.path.expanduser(str(dataset_root))
        annotation_dir = os.path.join(
            dataset_root,
            "annotations",
            f"task-{int(task_id):04d}",
        )
        if not os.path.isdir(annotation_dir):
            return None
        target = cls._normalize_skill_text(skill_name)
        for filename in sorted(os.listdir(annotation_dir)):
            if not filename.endswith(".json"):
                continue
            path = os.path.join(annotation_dir, filename)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    annotation = json.load(f)
            except OSError:
                continue
            occurrence = 0
            for skill in annotation.get("skill_annotation", []):
                descriptions = skill.get("skill_description", [])
                if isinstance(descriptions, str):
                    descriptions = [descriptions]
                if any(
                    cls._skill_name_matches(target, str(description))
                    for description in descriptions
                ):
                    if occurrence == skill_occurrence:
                        prompt = cls._load_orchestrator_prompt_for_skill(
                            dataset_root,
                            int(task_id),
                            filename,
                            int(skill.get("skill_idx", occurrence)),
                        )
                        return prompt or cls._build_skill_prompt(skill, skill_name)
                    occurrence += 1
        return None

    @staticmethod
    def _load_orchestrator_prompt_for_skill(
        dataset_root: str, task_id: int, annotation_filename: str, skill_idx: int
    ) -> str | None:
        episode_name = os.path.splitext(annotation_filename)[0]
        path = os.path.join(
            dataset_root,
            "orchestrators",
            f"task-{task_id:04d}",
            episode_name,
            "task_annotated.json",
        )
        try:
            with open(path, "r", encoding="utf-8") as f:
                orchestrator = json.load(f)
        except OSError:
            return None
        prompts = orchestrator.get("cot_subtask_description_list", [])
        if isinstance(prompts, list) and 0 <= skill_idx < len(prompts):
            prompt = str(prompts[skill_idx]).strip()
            return prompt or None
        return None

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

    @classmethod
    def _build_skill_prompt(cls, skill: dict, fallback_skill_name: str | None) -> str | None:
        descriptions = skill.get("skill_description", [])
        if isinstance(descriptions, str):
            descriptions = [descriptions]
        description = str(descriptions[0]).strip() if descriptions else fallback_skill_name
        if not description:
            return None
        object_name = cls._first_object_name(skill.get("object_id"))
        if object_name and object_name not in cls._normalize_skill_text(description).split():
            return f"{description} {object_name}"
        return description

    @classmethod
    def _first_object_name(cls, object_ids) -> str | None:
        if isinstance(object_ids, str):
            return cls._object_id_to_name(object_ids)
        if not isinstance(object_ids, (list, tuple)):
            return None
        for item in object_ids:
            name = cls._first_object_name(item)
            if name:
                return name
        return None

    @staticmethod
    def _object_id_to_name(object_id: str) -> str | None:
        parts = [part for part in str(object_id).split("_") if part]
        while parts and parts[-1].isdigit():
            parts.pop()
        if len(parts) > 1 and parts[-1].isalpha() and len(parts[-1]) >= 4:
            parts.pop()
        return " ".join(parts) if parts else None

    def _prompt_summary(self) -> dict:
        activity_name = OmegaConf.select(
            self.cfg, "env.eval.omni_config.task.activity_name", default=None
        )
        if activity_name is None:
            activity_name = OmegaConf.select(
                self.cfg, "env.eval.move_to_eval.activity_name", default=None
            )
        task_prompt = OmegaConf.select(self.cfg, "env.eval.task_prompt", default=None)
        if task_prompt is None:
            task_prompt = self._load_behavior_task_prompt(activity_name)
        if task_prompt is None:
            task_prompt = activity_name

        subtask_prompt = OmegaConf.select(
            self.cfg, "env.eval.subtask_prompt_override", default=None
        )
        subtask_source = "override" if subtask_prompt else None
        if subtask_prompt is None and OmegaConf.select(
            self.cfg, "env.eval.move_to_eval.enabled", default=False
        ):
            subtask_prompt = "move to"
            subtask_source = "move_to_eval"
        if subtask_prompt is None and OmegaConf.select(
            self.cfg, "env.eval.use_subtask_prompt", default=False
        ):
            skill_name = OmegaConf.select(
                self.cfg, "env.eval.replay_init.skill_name", default=None
            )
            if skill_name is not None:
                subtask_prompt = self._load_behavior_skill_prompt(
                    OmegaConf.select(
                        self.cfg, "env.eval.replay_init.dataset_root", default=None
                    ),
                    OmegaConf.select(
                        self.cfg, "env.eval.replay_init.task_id", default=None
                    ),
                    skill_name,
                    int(
                        OmegaConf.select(
                            self.cfg,
                            "env.eval.replay_init.skill_occurrence",
                            default=0,
                        )
                    ),
                ) or skill_name
                subtask_source = "replay_init.skill_name"
            stage_index = OmegaConf.select(
                self.cfg, "env.eval.replay_init.stage_index", default=None
            )
            if subtask_prompt is None and stage_index is not None:
                subtask_prompt = f"stage_index={stage_index}"
                subtask_source = "replay_init.stage_index"

        prompt = task_prompt
        if subtask_source == "override" and subtask_prompt:
            prompt = subtask_prompt
        elif (
            OmegaConf.select(self.cfg, "env.eval.subtask_prompt_only", default=False)
            and subtask_prompt
        ):
            prompt = subtask_prompt
        elif task_prompt and subtask_prompt:
            prompt = f"{task_prompt}\nCurrent stage: {subtask_prompt}"

        return {
            "activity_name": activity_name,
            "task_prompt": task_prompt,
            "subtask": subtask_prompt,
            "subtask_source": subtask_source,
            "prompt": prompt,
        }

    def _write_eval_summary_json(self, eval_metrics: dict):
        eval_metrics_list = self._last_eval_metrics_list
        success_once = [bool(x) for x in self._concat_metric(eval_metrics_list, "success_once")]
        success_at_end = [
            bool(x) for x in self._concat_metric(eval_metrics_list, "success_at_end")
        ]
        episode_lens = [int(x) for x in self._concat_metric(eval_metrics_list, "episode_len")]
        success_steps_raw = self._concat_metric(eval_metrics_list, "success_step")
        success_steps = [int(x) for x in success_steps_raw]

        num_trajectories = int(self._as_scalar(eval_metrics.get("num_trajectories", 0)))
        if num_trajectories == 0:
            num_trajectories = max(len(success_once), len(success_at_end), len(episode_lens))
        success_once_count = sum(success_once)
        success_at_end_count = sum(success_at_end)
        valid_success_steps = [
            step
            for ok, step in zip(success_once, success_steps)
            if ok and step >= 0
        ]
        prompt_info = self._prompt_summary()

        episodes = []
        for idx in range(num_trajectories):
            success_step = success_steps[idx] if idx < len(success_steps) else -1
            episodes.append(
                {
                    "index": idx,
                    "success_once": success_once[idx] if idx < len(success_once) else None,
                    "success_at_end": (
                        success_at_end[idx] if idx < len(success_at_end) else None
                    ),
                    "episode_len": episode_lens[idx] if idx < len(episode_lens) else None,
                    "success_step": success_step if success_step >= 0 else None,
                    "current_stage_idx": self._metric_value_at(
                        eval_metrics_list, "current_stage_idx", idx
                    ),
                    "total_stage_count": self._metric_value_at(
                        eval_metrics_list, "total_stage_count", idx
                    ),
                    "target_distance": self._metric_value_at(
                        eval_metrics_list, "target_distance", idx
                    ),
                    "body_target_distance": self._metric_value_at(
                        eval_metrics_list, "body_target_distance", idx
                    ),
                    "eef_target_distance": self._metric_value_at(
                        eval_metrics_list, "eef_target_distance", idx
                    ),
                    "target_object_found": self._metric_value_at(
                        eval_metrics_list, "target_object_found", idx
                    ),
                    "success_distance_threshold": self._metric_value_at(
                        eval_metrics_list, "success_distance_threshold", idx
                    ),
                    "skill_chain_stage_idx": self._metric_value_at(
                        eval_metrics_list, "skill_chain_stage_idx", idx
                    ),
                    "skill_chain_stage_count": self._metric_value_at(
                        eval_metrics_list, "skill_chain_stage_count", idx
                    ),
                    "skill_chain_stage_elapsed": self._metric_value_at(
                        eval_metrics_list, "skill_chain_stage_elapsed", idx
                    ),
                    "skill_chain_stage_limit": self._metric_value_at(
                        eval_metrics_list, "skill_chain_stage_limit", idx
                    ),
                    "skill_chain_completed_count": self._metric_value_at(
                        eval_metrics_list, "skill_chain_completed_count", idx
                    ),
                    "skill_chain_skill_success": self._metric_value_at(
                        eval_metrics_list, "skill_chain_skill_success", idx
                    ),
                    "skill_chain_failed_stage": self._metric_value_at(
                        eval_metrics_list, "skill_chain_failed_stage", idx
                    ),
                    "skill_chain_failed": self._metric_value_at(
                        eval_metrics_list, "skill_chain_failed", idx
                    ),
                    "skill_chain_done": self._metric_value_at(
                        eval_metrics_list, "skill_chain_done", idx
                    ),
                    "return": self._metric_value_at(eval_metrics_list, "return", idx),
                    "reward": self._metric_value_at(eval_metrics_list, "reward", idx),
                    "subtask": prompt_info["subtask"],
                    "prompt": prompt_info["prompt"],
                }
            )

        summary = {
            **prompt_info,
            "num_trajectories": num_trajectories,
            "success_once_count": success_once_count,
            "success_at_end_count": success_at_end_count,
            "success_rate_once": (
                success_once_count / num_trajectories if num_trajectories else None
            ),
            "success_rate_at_end": (
                success_at_end_count / num_trajectories if num_trajectories else None
            ),
            "avg_episode_len": (
                sum(episode_lens) / len(episode_lens) if episode_lens else None
            ),
            "avg_success_step": (
                sum(valid_success_steps) / len(valid_success_steps)
                if valid_success_steps
                else None
            ),
            "metrics": {key: self._as_scalar(value) for key, value in eval_metrics.items()},
            "episodes": episodes,
        }

        output_path = os.path.join(self.cfg.runner.logger.log_path, "eval_summary.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        self.logger.info(f"Wrote eval summary JSON to {output_path}")

    def _metric_value_at(self, eval_metrics_list, key: str, idx: int):
        values = self._concat_metric(eval_metrics_list, key)
        if idx >= len(values):
            return None
        value = values[idx]
        if isinstance(value, float) and value.is_integer():
            return int(value)
        return value

    def run(self):
        eval_metrics = self.evaluate()
        self._write_eval_summary_json(eval_metrics)
        eval_metrics = {f"eval/{k}": v for k, v in eval_metrics.items()}
        self.logger.info(eval_metrics)
        self.metric_logger.log(step=0, data=eval_metrics)

        self.metric_logger.finish()
