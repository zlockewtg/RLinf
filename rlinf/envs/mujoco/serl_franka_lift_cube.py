import os
import copy
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
import gym

from rlinf.envs.utils import put_info_on_image, save_rollout_video, tile_images


# ==========================================================
# Obs utils
# ==========================================================
def _unwrap_object_scalar(x, max_unwrap: int = 10):
    cur = x
    for _ in range(max_unwrap):
        arr = np.asarray(cur)
        if arr.dtype == object and arr.shape == ():
            try:
                cur = arr.item()
            except Exception:
                break
        else:
            break
    return cur


def _flatten_any_safe(x, visited=None, depth=0, max_depth=20) -> np.ndarray:
    if visited is None:
        visited = set()
    if depth > max_depth:
        raise RuntimeError("flatten exceeded max_depth; obs too deep/cyclic")

    obj_id = id(x)
    if obj_id in visited:
        return np.zeros((0,), dtype=np.float32)
    visited.add(obj_id)

    if isinstance(x, dict):
        return np.concatenate(
            [_flatten_any_safe(x[k], visited, depth + 1, max_depth) for k in sorted(x.keys())],
            axis=0,
        ) if x else np.zeros((0,), dtype=np.float32)

    if isinstance(x, (list, tuple)):
        return np.concatenate(
            [_flatten_any_safe(v, visited, depth + 1, max_depth) for v in x],
            axis=0,
        ) if x else np.zeros((0,), dtype=np.float32)

    arr = np.asarray(x)
    if arr.dtype == object and arr.shape == ():
        y = _unwrap_object_scalar(x)
        return _flatten_any_safe(y, visited, depth + 1, max_depth) if id(y) != id(x) else np.zeros((0,), np.float32)

    if arr.dtype == object:
        return np.concatenate(
            [_flatten_any_safe(v, visited, depth + 1, max_depth) for v in arr.ravel().tolist()],
            axis=0,
        ) if arr.size else np.zeros((0,), np.float32)

    return arr.astype(np.float32, copy=False).reshape(-1)


def extract_serl_state(raw_obs, state_key="states") -> np.ndarray:
    if isinstance(raw_obs, tuple) and len(raw_obs) == 2:
        raw_obs = raw_obs[0]
    if isinstance(raw_obs, dict):
        raw_obs = raw_obs[state_key]
    raw_obs = _unwrap_object_scalar(raw_obs)
    return _flatten_any_safe(raw_obs)


def extract_serl_images_dict(raw_obs, image_key="images") -> Dict[str, np.ndarray]:
    if isinstance(raw_obs, tuple) and len(raw_obs) == 2:
        raw_obs = raw_obs[0]
    assert isinstance(raw_obs, dict), f"Expected dict obs, got {type(raw_obs)}"

    images_dict = _unwrap_object_scalar(raw_obs[image_key])
    if not isinstance(images_dict, dict):
        raise TypeError(f"Expected images_dict to be dict, got {type(images_dict)}")

    return {k: np.asarray(v, dtype=np.uint8) for k, v in images_dict.items()}


# ==========================================================
# Generic helpers
# ==========================================================
def _cfg_get(cfg, key: str, default=None):
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def torch_clone(x):
    if isinstance(x, torch.Tensor):
        return x.clone()
    if isinstance(x, dict):
        return {k: torch_clone(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [torch_clone(v) for v in x]
    return copy.deepcopy(x)


def _to_scalar(v):
    if isinstance(v, np.ndarray) and v.shape == ():
        return v.item()
    if isinstance(v, torch.Tensor) and v.numel() == 1:
        return v.item()
    return v


# ==========================================================
# SERLFrankaEnv
# ==========================================================
class SERLFrankaEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, cfg, num_envs: int, seed_offset: int, total_num_processes: int, worker_info=None, record_metrics=True):
        super().__init__()
        self.cfg = cfg
        self.seed = int(_cfg_get(cfg, "seed", 0)) + int(seed_offset)
        self.total_num_processes = int(total_num_processes)
        self.worker_info = worker_info

        # basic flags
        self.auto_reset = bool(_cfg_get(cfg, "auto_reset", True))
        self.use_rel_reward = bool(_cfg_get(cfg, "use_rel_reward", False))
        self.ignore_terminations = bool(_cfg_get(cfg, "ignore_terminations", False))

        # group (compat only)
        self.group_size = int(_cfg_get(cfg, "group_size", 1))
        self.num_group = int(num_envs) // max(self.group_size, 1)
        self.use_fixed_reset_state_ids = bool(_cfg_get(cfg, "use_fixed_reset_state_ids", False))

        # obs config
        wrap_obs_mode = _cfg_get(cfg, "wrap_obs_mode", None)
        if wrap_obs_mode is None:
            wrap_obs_mode = "openvla" if str(_cfg_get(cfg, "obs_format", "openvla")).lower() == "openvla" else "simple"
        self.wrap_obs_mode = str(wrap_obs_mode).lower()

        obs_mode = str(_cfg_get(cfg, "obs_mode", "state")).lower()
        self.obs_mode = "rgb" if obs_mode in ("rgb", "image", "vision") else "state"

        self.state_key = _cfg_get(cfg, "state_key", "states")
        self.image_key = _cfg_get(cfg, "image_key", "images")
        self.main_camera = _cfg_get(cfg, "main_camera", _cfg_get(cfg, "camera", "front"))
        self.extra_camera = _cfg_get(cfg, "extra_camera", "wrist")
        self.use_wrist_as_extra_view = bool(_cfg_get(cfg, "use_wrist_as_extra_view", True))

        self.task_prompt = _cfg_get(cfg, "task_prompt", "Pick up the cube.")
        self.record_metrics = bool(record_metrics)

        # device
        self._device = torch.device(str(_cfg_get(cfg, "device", "cpu")))
        if self._device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(f"cfg.device={self._device} but CUDA not available")

        # video config
        self.video_cfg = _cfg_get(cfg, "video_cfg", None)
        self.video_cnt = 0
        self.render_images: List[np.ndarray] = []

        # Mujoco backend
        if self.obs_mode == "rgb":
            os.environ.setdefault("MUJOCO_GL", str(_cfg_get(cfg, "mujoco_gl", "egl")))
            os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
            os.environ.setdefault("NVIDIA_DRIVER_CAPABILITIES", "all")
        else:
            os.environ.setdefault("MUJOCO_GL", str(_cfg_get(cfg, "mujoco_gl", "disable")))
            self.video_cfg = None

        # build envs
        self.env_id = _cfg_get(cfg, "gym_id", "PandaPickCube-v0")
        self._num_envs = int(num_envs)
        self.envs = [self._make_env(i) for i in range(self._num_envs)]

        self.single_action_space = self.envs[0].action_space
        self.action_space = self.single_action_space

        # infer obs space
        raw0, _ = self._gym_reset(self.envs[0], seed=self.seed)
        self._state_dim = int(extract_serl_state(raw0, self.state_key).shape[0])
        self._init_observation_space(raw0)

        # tracking
        self.prev_step_reward = torch.zeros(self._num_envs, device=self.device)
        self._elapsed_steps = torch.zeros(self._num_envs, dtype=torch.int32, device=self.device)
        self._needs_reset = torch.zeros(self._num_envs, dtype=torch.bool, device=self.device)
        self._is_start = True

        if self.record_metrics:
            self._init_metrics()

        self._last_obs = None
        self._last_info = {}

        self._init_reset_state_ids()

    # -------------------- basic properties --------------------
    @property
    def num_envs(self):
        return self._num_envs

    @property
    def device(self):
        return self._device

    @property
    def elapsed_steps(self):
        return self._elapsed_steps

    @property
    def is_start(self):
        return self._is_start

    @is_start.setter
    def is_start(self, value):
        self._is_start = bool(value)

    @property
    def instruction(self):
        return [str(self.task_prompt)] * self.num_envs

    @property
    def total_num_group_envs(self):
        return np.iinfo(np.uint8).max // 2

    # -------------------- init helpers --------------------
    def _make_env(self, i: int):
        kwargs = {}
        if self.obs_mode == "rgb":
            kwargs.update(image_obs=True, render_mode="rgb_array")
        e = gym.make(self.env_id, disable_env_checker=True, **kwargs)
        try:
            e.reset(seed=self.seed + i)
        except Exception:
            pass
        return e

    def _gym_reset(self, env, seed=None, options=None):
        try:
            return env.reset(seed=seed, options=options)
        except TypeError:
            out = env.reset(seed=seed)
            return out if isinstance(out, tuple) else (out, {})
        except Exception:
            out = env.reset()
            return out if isinstance(out, tuple) else (out, {})

    def _init_observation_space(self, raw0):
        if self.obs_mode != "rgb":
            self.observation_space = gym.spaces.Dict(
                {"states": gym.spaces.Box(-np.inf, np.inf, shape=(self._num_envs, self._state_dim), dtype=np.float32)}
            )
            return

        imgs = extract_serl_images_dict(raw0, self.image_key)
        main = imgs.get(self.main_camera, imgs[sorted(imgs.keys())[0]])
        h, w, c = main.shape

        spaces = {
            "main_images": gym.spaces.Box(0, 255, shape=(self._num_envs, h, w, c), dtype=np.uint8),
            "states": gym.spaces.Box(-np.inf, np.inf, shape=(self._num_envs, self._state_dim), dtype=np.float32),
        }
        if self.use_wrist_as_extra_view:
            spaces["extra_view_images"] = gym.spaces.Box(0, 255, shape=(self._num_envs, 1, h, w, c), dtype=np.uint8)

        self.observation_space = gym.spaces.Dict(spaces)

    # -------------------- metrics --------------------
    def _init_metrics(self):
        self.success_once = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.fail_once = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.returns = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)

    def _reset_metrics(self, idx: Optional[torch.Tensor] = None):
        if idx is None:
            idx = torch.arange(self.num_envs, device=self.device)
        self.prev_step_reward[idx] = 0.0
        self._elapsed_steps[idx] = 0
        if self.record_metrics:
            self.success_once[idx] = False
            self.fail_once[idx] = False
            self.returns[idx] = 0.0

    def _record_metrics(self, step_reward: torch.Tensor, infos: Dict[str, Any]):
        if not self.record_metrics:
            return infos
        self.returns += step_reward
        ep = {
            "return": self.returns.clone(),
            "episode_len": self.elapsed_steps.clone(),
        }
        denom = torch.clamp(ep["episode_len"].float(), min=1.0)
        ep["reward"] = ep["return"] / denom

        if "success" in infos:
            self.success_once |= infos["success"].bool()
            ep["success_once"] = self.success_once.clone()

        if "fail" in infos:
            self.fail_once |= infos["fail"].bool()
            ep["fail_once"] = self.fail_once.clone()

        infos["episode"] = ep
        return infos

    # -------------------- obs wrapping --------------------
    def _pick_images(self, raw_obs):
        images = extract_serl_images_dict(raw_obs, self.image_key)
        keys = sorted(images.keys())
        main = images.get(self.main_camera, images[keys[0]])
        extra = images.get(self.extra_camera, main)
        return main, extra

    def _wrap_obs_one(self, raw_obs) -> Dict[str, Any]:
        state = torch.from_numpy(extract_serl_state(raw_obs, self.state_key)).to(self.device)

        if self.obs_mode == "state":
            out = {"states": state}
            if self.wrap_obs_mode != "simple":
                out["task_descriptions"] = str(self.task_prompt)
            return out

        main_np, extra_np = self._pick_images(raw_obs)
        main = torch.from_numpy(np.ascontiguousarray(main_np, np.uint8)).to(self.device)
        extra = torch.from_numpy(np.ascontiguousarray(extra_np, np.uint8)).to(self.device).unsqueeze(0)

        if self.wrap_obs_mode == "simple":
            return {"main_images": main, "extra_view_images": extra if self.use_wrist_as_extra_view else None, "states": state}

        if self.wrap_obs_mode == "openvla":
            return {"main_images": main, "extra_view_images": extra if self.use_wrist_as_extra_view else None, "states": state, "task_descriptions": str(self.task_prompt)}

        if self.wrap_obs_mode == "openpi":
            return {"main_images": main, "wrist_images": extra.squeeze(0), "extra_view_images": extra if self.use_wrist_as_extra_view else None, "states": state, "task_descriptions": str(self.task_prompt)}

        raise ValueError(f"Unknown wrap_obs_mode: {self.wrap_obs_mode}")

    def _collate_obs(self, obs_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        keys = set().union(*[o.keys() for o in obs_list])

        for k in sorted(keys):
            vals = [o.get(k, None) for o in obs_list]
            if all(v is None for v in vals):
                out[k] = None
            elif isinstance(vals[0], torch.Tensor):
                out[k] = torch.stack(vals, dim=0)
            else:
                out[k] = vals

        # pad states to max
        states = [o["states"].view(-1) for o in obs_list]
        max_d = max(s.numel() for s in states)
        padded = torch.zeros((len(states), max_d), device=self.device)
        for i, s in enumerate(states):
            padded[i, : s.numel()] = s
        out["states"] = padded
        return out

    def _collate_infos(self, infos: List[dict]) -> Dict[str, Any]:
        keys = set().union(*[inf.keys() for inf in infos if isinstance(inf, dict)])
        out: Dict[str, Any] = {}
        for k in sorted(keys):
            vals = [_to_scalar(inf.get(k, None)) for inf in infos]
            is_bool = all(isinstance(v, (bool, np.bool_)) or v is None for v in vals)
            is_num = all(isinstance(v, (int, float, np.number)) or v is None for v in vals)

            if is_bool:
                out[k] = torch.tensor([bool(v) if v is not None else False for v in vals], device=self.device, dtype=torch.bool)
            elif is_num:
                out[k] = torch.tensor([float(v) if v is not None else 0.0 for v in vals], device=self.device, dtype=torch.float32)
            else:
                out[k] = vals

        # normalize success/fail
        if "success" not in out and "is_success" in out:
            out["success"] = out["is_success"].bool()
        if "success" in out and isinstance(out["success"], torch.Tensor) and out["success"].dtype != torch.bool:
            out["success"] = out["success"] > 0.5
        if "fail" in out and isinstance(out["fail"], torch.Tensor) and out["fail"].dtype != torch.bool:
            out["fail"] = out["fail"] > 0.5

        return out

    # -------------------- video --------------------
    def add_new_frames(self, obs: Dict[str, Any], plot_infos: Dict[str, Any]):
        if "main_images" not in obs:
            return
        imgs = obs["main_images"].detach().cpu().numpy()
        images = []
        for env_id in range(imgs.shape[0]):
            info_item = {}
            for k, v in plot_infos.items():
                if isinstance(v, torch.Tensor):
                    v = v.detach().cpu().numpy()
                info_item[k] = v[env_id] if isinstance(v, (list, np.ndarray)) and np.size(v) > 1 else v
            img = imgs[env_id]
            if getattr(self.video_cfg, "info_on_video", False):
                img = put_info_on_image(img, info_item)
            images.append(img)
        full_image = tile_images(images, nrows=int(np.sqrt(self.num_envs)))
        self.render_images.append(full_image)

    def flush_video(self, sub_dir=None):
        if not (self.video_cfg and getattr(self.video_cfg, "save_video", False)):
            return
        output_dir = os.path.join(self.video_cfg.video_base_dir, f"seed_{self.seed}")
        if sub_dir:
            output_dir = os.path.join(output_dir, sub_dir)
        save_rollout_video(self.render_images, output_dir=output_dir, video_name=str(self.video_cnt))
        self.video_cnt += 1
        self.render_images = []

    # ==========================================================
    # RLinf API
    # ==========================================================
    def reset(self, *, seed: Optional[Union[int, List[int]]] = None, options: Optional[dict] = None):
        env_idx = None if options is None else options.get("env_idx", None)
        reset_options = {} if options is None else {k: v for k, v in options.items() if k != "env_idx"}

        if env_idx is not None:
            env_idx = torch.as_tensor(env_idx, dtype=torch.int64, device=self.device)
            if self._last_obs is None:
                env_idx = None

        idxs = set(range(self.num_envs)) if env_idx is None else set(env_idx.detach().cpu().tolist())

        obs_list, info_list = [], []
        for i in range(self.num_envs):
            if i in idxs:
                seed_i = None if seed is None else (int(seed[i]) if isinstance(seed, (list, tuple, np.ndarray)) else int(seed) + i)
                raw_obs, info = self._gym_reset(self.envs[i], seed_i, reset_options)
                obs_list.append(self._wrap_obs_one(raw_obs))
                info_list.append(info if isinstance(info, dict) else {})
            else:
                obs_list.append({k: v[i] for k, v in self._last_obs.items()} if self._last_obs else self._wrap_obs_one(self._gym_reset(self.envs[i])[0]))
                info_list.append({})

        obs = self._collate_obs(obs_list)
        infos = self._collate_infos(info_list)

        if env_idx is None:
            self._reset_metrics()
            self._needs_reset[:] = False
        else:
            self._reset_metrics(env_idx)
            self._needs_reset[env_idx] = False

        self._is_start = True
        self._last_obs, self._last_info = obs, infos
        if self.video_cfg and getattr(self.video_cfg, "save_video", False):
            self.render_images = []
        return obs, infos

    def step(self, actions: Union[np.ndarray, torch.Tensor] = None, auto_reset: bool = True):
        if actions is None:
            raise ValueError("actions cannot be None")
        if isinstance(actions, dict):
            raise NotImplementedError("dict actions not supported")

        act_np = actions.detach().cpu().numpy() if isinstance(actions, torch.Tensor) else np.asarray(actions)
        if act_np.ndim == 1:
            act_np = np.repeat(act_np[None, :], self.num_envs, axis=0)
        act_np = act_np.astype(np.float32, copy=False)

        obs_list, info_list, rew_list, term_list, trunc_list = [], [], [], [], []
        stepped = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        for i, env in enumerate(self.envs):
            if self._needs_reset[i]:
                if auto_reset and self.auto_reset:
                    env.reset()
                    self._needs_reset[i] = False
                    self._reset_metrics(torch.tensor([i], device=self.device))
                else:
                    obs_list.append({k: v[i] for k, v in self._last_obs.items()})
                    info_list.append({})
                    rew_list.append(0.0)
                    term_list.append(True)
                    trunc_list.append(False)
                    continue

            act_i = np.clip(act_np[i].reshape(-1), -1.0, 1.0)
            out = env.step(act_i)
            if len(out) == 5:
                raw_obs, rew, terminated, truncated, info = out
            else:
                raw_obs, rew, done, info = out
                terminated, truncated = bool(done), False

            obs_list.append(self._wrap_obs_one(raw_obs))
            info_list.append(info if isinstance(info, dict) else {})
            rew_list.append(float(rew))
            term_list.append(bool(terminated))
            trunc_list.append(bool(truncated))
            stepped[i] = True

        self._elapsed_steps[stepped] += 1

        obs = self._collate_obs(obs_list)
        infos = self._collate_infos(info_list)

        raw_reward = torch.tensor(rew_list, device=self.device, dtype=torch.float32)
        step_reward = (raw_reward - self.prev_step_reward) if self.use_rel_reward else raw_reward
        self.prev_step_reward = raw_reward

        step_reward *= float(_cfg_get(self.cfg, "reward_scale", 1e5))

        terminations = torch.tensor(term_list, device=self.device, dtype=torch.bool)
        truncations = torch.tensor(trunc_list, device=self.device, dtype=torch.bool)

        infos = self._record_metrics(step_reward, infos)

        if self.ignore_terminations:
            terminations = torch.zeros_like(terminations)

        dones = terminations | truncations

        if self.video_cfg and getattr(self.video_cfg, "save_video", False):
            self.add_new_frames(obs, {"reward": step_reward, "done": dones, "task": self.instruction})

        _auto = bool(auto_reset) and bool(self.auto_reset)
        if dones.any() and not _auto:
            self._needs_reset |= dones

        self._last_obs, self._last_info = obs, infos

        if dones.any() and _auto:
            if self.video_cfg and getattr(self.video_cfg, "save_video", False):
                self.flush_video(sub_dir="eval")
            obs, infos = self._handle_auto_reset(dones, obs, infos)

        return obs, step_reward, terminations, truncations, infos

    def _handle_auto_reset(self, dones: torch.Tensor, obs: Dict[str, Any], infos: Dict[str, Any]):
        final_obs = torch_clone(obs)
        final_info = torch_clone(infos)

        env_idx = torch.arange(self.num_envs, device=self.device)[dones]
        new_obs, new_infos = self.reset(options={"env_idx": env_idx})

        new_infos = dict(new_infos)
        new_infos.update({
            "final_observation": final_obs,
            "final_info": final_info,
            "_final_info": dones,
            "_final_observation": dones,
            "_elapsed_steps": dones,
        })
        self._needs_reset[env_idx] = False
        self._last_obs, self._last_info = new_obs, new_infos
        return new_obs, new_infos

    def chunk_step(self, chunk_actions: Union[np.ndarray, torch.Tensor]):
        ca = chunk_actions if isinstance(chunk_actions, torch.Tensor) else torch.from_numpy(np.asarray(chunk_actions))
        if ca.ndim != 3:
            raise ValueError(f"chunk_actions must be [B,T,A], got {tuple(ca.shape)}")
        B, T, _ = ca.shape
        if B != self.num_envs:
            raise ValueError(f"chunk_actions batch={B} != num_envs={self.num_envs}")

        rewards, terms, truncs = [], [], []
        infos = {}
        for t in range(T):
            obs, r, term, trunc, infos = self.step(ca[:, t].to(self.device), auto_reset=False)
            rewards.append(r)
            terms.append(term)
            truncs.append(trunc)

        rewards = torch.stack(rewards, dim=1)
        terms = torch.stack(terms, dim=1)
        truncs = torch.stack(truncs, dim=1)

        past_done = (terms.any(dim=1) | truncs.any(dim=1))
        if past_done.any() and self.auto_reset:
            obs, infos = self._handle_auto_reset(past_done, obs, infos)

        # only mark done at last timestep
        chunk_terms = torch.zeros_like(terms)
        chunk_terms[:, -1] = terms.any(dim=1)
        chunk_truncs = torch.zeros_like(truncs)
        chunk_truncs[:, -1] = truncs.any(dim=1)
        return obs, rewards, chunk_terms, chunk_truncs, infos

    def sample_action_space(self):
        return torch.from_numpy(np.asarray(self.action_space.sample(), np.float32)).to(self.device)

    def close(self):
        for e in self.envs:
            try:
                e.close()
            except Exception:
                pass
        self.envs = []

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def _init_reset_state_ids(self):
        self.reset_state_ids = None

    def update_reset_state_ids(self):
        return
