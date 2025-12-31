# rlinf/envs/mujoco/serl_panda_pick_cube.py
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

# SERL registers envs into gym (0.26), not gymnasium
import gym


# ==========================================================
# Utils: unwrap object scalar
# ==========================================================
def _unwrap_object_scalar(x, max_unwrap: int = 10):
    """
    Unwrap numpy object scalar repeatedly:
      () object -> python object
    """
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


# ==========================================================
# State extraction (robust flatten)
# ==========================================================
def _flatten_any_safe(x, visited=None, depth=0, max_depth=20):
    """
    Robust flatten for SERL 'state' obs:
      obs['state'] is an object scalar -> dict -> numeric arrays/scalars

    Returns: 1D float32 vector
    - detects cyclic references via visited ids
    - unwraps object scalar safely
    """
    if visited is None:
        visited = set()
    if depth > max_depth:
        raise RuntimeError("flatten exceeded max_depth; obs too deep/cyclic")

    obj_id = id(x)
    if obj_id in visited:
        return np.zeros((0,), dtype=np.float32)
    visited.add(obj_id)

    # unwrap object scalar early
    arr = np.asarray(x)
    if arr.dtype == object and arr.shape == ():
        y = _unwrap_object_scalar(x, max_unwrap=10)
        if id(y) == id(x):
            return np.zeros((0,), dtype=np.float32)
        return _flatten_any_safe(y, visited, depth + 1, max_depth)

    # dict: concat sorted keys for determinism
    if isinstance(x, dict):
        parts = []
        for k in sorted(x.keys()):
            parts.append(_flatten_any_safe(x[k], visited, depth + 1, max_depth))
        return np.concatenate(parts, axis=0) if parts else np.zeros((0,), dtype=np.float32)

    # list/tuple: concat
    if isinstance(x, (list, tuple)):
        parts = [_flatten_any_safe(v, visited, depth + 1, max_depth) for v in x]
        return np.concatenate(parts, axis=0) if parts else np.zeros((0,), dtype=np.float32)

    # object array: recurse each element
    arr = np.asarray(x)
    if arr.dtype == object:
        parts = []
        for v in arr.ravel().tolist():
            parts.append(_flatten_any_safe(v, visited, depth + 1, max_depth))
        return np.concatenate(parts, axis=0) if parts else np.zeros((0,), dtype=np.float32)

    # numeric
    return arr.astype(np.float32, copy=False).reshape(-1)


def extract_serl_state(raw_obs, state_key: str = "state") -> np.ndarray:
    """
    SERL PandaPickCube(-Vision) obs: dict with key 'state' whose value is object scalar.
    We unwrap -> dict -> flatten numerics.
    Returns: float32 1D vector.
    """
    if isinstance(raw_obs, tuple) and len(raw_obs) == 2:
        raw_obs = raw_obs[0]

    if isinstance(raw_obs, dict):
        raw_obs = raw_obs[state_key]
    return _flatten_any_safe(raw_obs)


# ==========================================================
# Image extraction
# ==========================================================
def extract_serl_images_dict(raw_obs, image_key: str = "images") -> Dict[str, np.ndarray]:
    """
    SERL PandaPickCubeVision obs:
      raw_obs: dict with keys ['state', 'images']
      raw_obs['images'] is object scalar
      raw_obs['images'].item() -> dict { 'front': uint8 HWC, 'wrist': uint8 HWC }
    Returns:
      images_dict: { 'front': uint8 HWC, 'wrist': uint8 HWC }
    """
    if isinstance(raw_obs, tuple) and len(raw_obs) == 2:
        raw_obs = raw_obs[0]
    assert isinstance(raw_obs, dict), f"Expected dict obs, got {type(raw_obs)}"
    images_obj = raw_obs[image_key]
    images_dict = _unwrap_object_scalar(images_obj)

    if not isinstance(images_dict, dict):
        raise TypeError(f"Expected images_dict to be dict, got {type(images_dict)}")

    out = {}
    for k, v in images_dict.items():
        out[k] = np.asarray(v, dtype=np.uint8)
    return out


def extract_serl_image_tensor(
    raw_obs,
    image_key: str = "images",
    camera: str = "front",
    channel_first: bool = True,
    normalize: bool = True,
) -> np.ndarray:
    """
    For tensor obs_format:
      camera:
        - 'front' or 'wrist' -> 3xHxW
        - 'both' -> 6xHxW (front||wrist concat on channel)
    Returns float32, CHW, in [0,1] by default.
    """
    images_dict = extract_serl_images_dict(raw_obs, image_key=image_key)

    if camera == "both":
        front = images_dict["front"]
        wrist = images_dict["wrist"]
        img = np.concatenate([front, wrist], axis=-1)  # H W 6
    else:
        if camera not in images_dict:
            raise KeyError(f"camera '{camera}' not in images_dict keys={list(images_dict.keys())}")
        img = images_dict[camera]  # H W 3

    if img.ndim == 4 and img.shape[0] == 1:
        img = img[0]
    if img.ndim != 3:
        raise ValueError(f"Unexpected image ndim={img.ndim} shape={img.shape}")

    if normalize:
        img = img.astype(np.float32) / 255.0
    else:
        img = img.astype(np.float32)

    if channel_first:
        img = np.transpose(img, (2, 0, 1))

    return img


# ==========================================================
# SERL Env Wrapper (RLinf compatible)
# ==========================================================
class SERLPandaPickCubeEnv(gym.Env):
    """
    RLinf-compatible SERL env wrapper.

    Supports:
      - PandaPickCube-v0 (state-only)
      - PandaPickCubeVision-v0 (image+state)

    Config:
      - obs_mode: "state" | "image"
      - obs_format:
          - "tensor": returns numpy tensor batch:
              image -> (B,C,H,W) float32 in [0,1]
              state -> (B,D) float32
          - "vla": returns dict batch compatible with OpenVLA (mimic ManiSkillEnv extracted_obs):
              {
                "main_images": uint8 (B,H,W,3),
                "extra_view_images": uint8 (B,1,H,W,3) or None,
                "states": float32 (B,D),
                "task_descriptions": list[str] length B
              }

      - task_prompt: str instruction
      - use_wrist_as_extra_view: bool (extra_view_images)
      - camera: front | wrist | both (only used in tensor+image mode)

    Vectorization:
      - hosts num_envs sub-envs inside one process
    """

    metadata = {"render_modes": []}

    def __init__(self, cfg, rank: int, num_envs: int, ret_device: str = "cpu"):
        super().__init__()
        self.cfg = cfg
        self.rank = int(rank)
        self._num_envs = int(num_envs)
        self.ret_device = ret_device

        # headless mujoco
        os.environ.setdefault("MUJOCO_GL", getattr(cfg, "mujoco_gl", "egl"))

        # import triggers env registration into gym registry
        import franka_sim  # noqa: F401

        self.env_id = getattr(cfg, "gym_id", "PandaPickCube-v0")
        self.render = bool(getattr(cfg, "render", False))

        # obs settings
        self.obs_mode = getattr(cfg, "obs_mode", "state")          # "state" or "image"
        self.obs_format = getattr(cfg, "obs_format", "tensor")     # "tensor" or "vla"

        self.state_key = getattr(cfg, "state_key", "state")
        self.image_key = getattr(cfg, "image_key", "images")

        self.camera = getattr(cfg, "camera", "front")              # for tensor image mode
        self.use_wrist_as_extra_view = bool(getattr(cfg, "use_wrist_as_extra_view", True))
        self.task_prompt = getattr(cfg, "task_prompt", "Pick up the cube.")

        # seed
        base_seed = int(getattr(cfg, "seed", 0))
        self.seed = base_seed + self.rank * 1000

        # create sub-envs
        self.envs: List[gym.Env] = []
        for i in range(self._num_envs):
            e = gym.make(self.env_id, disable_env_checker=True)
            try:
                e.reset(seed=self.seed + i)
            except TypeError:
                try:
                    e.seed(self.seed + i)
                except Exception:
                    pass
            self.envs.append(e)

        self.single_action_space = self.envs[0].action_space

        # infer obs space by real reset output
        raw0 = self.envs[0].reset()
        if isinstance(raw0, tuple) and len(raw0) == 2:
            raw0 = raw0[0]

        # Determine observation space / shape for tensor mode
        if self.obs_format == "tensor":
            if self.obs_mode == "image":
                img0 = extract_serl_image_tensor(
                    raw0, image_key=self.image_key, camera=self.camera,
                    channel_first=True, normalize=True
                )
                c, h, w = img0.shape
                self.obs_shape = (c, h, w)
                self.observation_space = gym.spaces.Box(
                    low=0.0, high=1.0, shape=(self._num_envs, c, h, w), dtype=np.float32
                )
                print(f"[SERL DEBUG] obs_format=tensor obs_mode=image camera={self.camera} obs_shape={self.obs_shape}")
            else:
                flat0 = extract_serl_state(raw0, state_key=self.state_key)
                d = int(flat0.shape[0])
                self.obs_shape = (d,)
                self.observation_space = gym.spaces.Box(
                    low=-np.inf, high=np.inf, shape=(self._num_envs, d), dtype=np.float32
                )
                print(f"[SERL DEBUG] obs_format=tensor obs_mode=state obs_dim={d}")
        else:
            # vla format: we return dict, so observation_space is not strictly used by RLinf pipeline
            # but define a placeholder
            try:
                flat0 = extract_serl_state(raw0, state_key=self.state_key)
                d = int(flat0.shape[0])
            except Exception:
                d = 0
            self.obs_shape = None
            self.observation_space = gym.spaces.Dict({})
            print(f"[SERL DEBUG] obs_format=vla obs_mode={self.obs_mode} states_dim~{d}")

        # infer action dim
        act_sample = np.asarray(self.single_action_space.sample(), dtype=np.float32).reshape(-1)
        self.act_dim = int(act_sample.shape[0])

        # define batched action space for convenience (not strictly required)
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(self._num_envs, self.act_dim), dtype=np.float32
        )

    @property
    def num_envs(self):
        return self._num_envs

    # ------------------------------------------------------
    # VLA-format builders (mimic ManiSkillEnv extracted_obs)
    # ------------------------------------------------------
    def _build_vla_obs_one(self, raw_obs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build per-env extracted_obs like ManiSkillEnv._extract_obs_image():
          main_images: uint8 HWC
          extra_view_images: uint8 HWC (optional)
          states: float32 vector
          task_descriptions: str
        """
        # images
        images_dict = extract_serl_images_dict(raw_obs, image_key=self.image_key)
        front = images_dict.get("front", None)
        wrist = images_dict.get("wrist", None)
        if front is None:
            # fallback: take first key
            k0 = sorted(images_dict.keys())[0]
            front = images_dict[k0]
        if wrist is None:
            wrist = front

        # states
        try:
            state_vec = extract_serl_state(raw_obs, state_key=self.state_key).astype(np.float32)
        except Exception:
            state_vec = np.zeros((0,), dtype=np.float32)

        obs = {
            "main_images": front,                 # HWC uint8
            "states": state_vec,                  # (D,) float32
            "task_descriptions": self.task_prompt # str
        }
        if self.use_wrist_as_extra_view:
            obs["extra_view_images"] = wrist      # HWC uint8
        else:
            obs["extra_view_images"] = None
        return obs

    def _collate_vla_obs_batch(self, obs_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Collate list of per-env vla obs into batch dict:
          main_images: uint8 (B,H,W,3)
          extra_view_images: uint8 (B,1,H,W,3) or None
          states: float32 (B,D)
          task_descriptions: list[str] length B
        """
        B = len(obs_list)

        main_imgs = np.stack([o["main_images"] for o in obs_list], axis=0).astype(np.uint8)

        # states (pad if inconsistent dims)
        states_list = [o["states"] for o in obs_list]
        max_d = max([s.shape[0] for s in states_list]) if states_list else 0
        if max_d == 0:
            states = np.zeros((B, 0), dtype=np.float32)
        else:
            states = np.zeros((B, max_d), dtype=np.float32)
            for i, s in enumerate(states_list):
                states[i, : s.shape[0]] = s.astype(np.float32)

        task_desc = [o["task_descriptions"] for o in obs_list]

        extra = None
        if obs_list[0].get("extra_view_images", None) is not None:
            extra_imgs = np.stack([o["extra_view_images"] for o in obs_list], axis=0).astype(np.uint8)
            # ManiSkill simple mode tends to use (B, num_views, H, W, C)
            extra = extra_imgs[:, None, ...]  # (B,1,H,W,3)

        return {
            "main_images": main_imgs,
            "extra_view_images": extra,
            "states": states,
            "task_descriptions": task_desc,
        }

    # ------------------------------------------------------
    # core helpers
    # ------------------------------------------------------
    def _reset_one(self, env):
        out = env.reset()
        if isinstance(out, tuple) and len(out) == 2:
            raw_obs, info = out
        else:
            raw_obs, info = out, {}

        if self.obs_format == "vla":
            obs = self._build_vla_obs_one(raw_obs)
        else:
            if self.obs_mode == "image":
                obs = extract_serl_image_tensor(
                    raw_obs, image_key=self.image_key, camera=self.camera,
                    channel_first=True, normalize=True
                )
            else:
                obs = extract_serl_state(raw_obs, state_key=self.state_key)

        return obs, info

    def _step_one(self, env, act):
        out = env.step(act)

        # gymnasium: obs, reward, terminated, truncated, info
        if isinstance(out, tuple) and len(out) == 5:
            raw_obs, rew, terminated, truncated, info = out
        # gym: obs, reward, done, info
        elif isinstance(out, tuple) and len(out) == 4:
            raw_obs, rew, done, info = out
            terminated, truncated = bool(done), False
        else:
            raise RuntimeError(f"Unexpected env.step return length: {len(out)}")

        if self.obs_format == "vla":
            obs = self._build_vla_obs_one(raw_obs)
        else:
            if self.obs_mode == "image":
                obs = extract_serl_image_tensor(
                    raw_obs, image_key=self.image_key, camera=self.camera,
                    channel_first=True, normalize=True
                )
            else:
                obs = extract_serl_state(raw_obs, state_key=self.state_key)

        return obs, float(rew), bool(terminated), bool(truncated), info

    # ------------------------------------------------------
    # RLinf-style batched API
    # ------------------------------------------------------
    def reset(self, options: Optional[Dict[str, Any]] = None):
        obs_batch = []
        for e in self.envs:
            o, _ = self._reset_one(e)
            obs_batch.append(o)

        if self.obs_format == "vla":
            obs = self._collate_vla_obs_batch(obs_batch)
        else:
            obs = np.stack(obs_batch, axis=0).astype(np.float32)

        return obs, {}

    def step(self, actions: np.ndarray):
        actions = np.asarray(actions, dtype=np.float32)
        actions = np.clip(actions, -1.0, 1.0)

        obs_batch, rew_batch, term_batch, trunc_batch = [], [], [], []
        success_batch = []

        for i, env in enumerate(self.envs):
            act = actions[i].reshape(-1)
            obs, rew, terminated, truncated, info = self._step_one(env, act)

            # auto-reset on done
            if terminated or truncated:
                obs, _ = self._reset_one(env)

            obs_batch.append(obs)
            rew_batch.append(rew)
            term_batch.append(terminated)
            trunc_batch.append(truncated)

            s = 0.0
            if isinstance(info, dict):
                s = info.get("success", info.get("is_success", 0.0))
            success_batch.append(float(s))

        if self.obs_format == "vla":
            obs = self._collate_vla_obs_batch(obs_batch)
        else:
            obs = np.stack(obs_batch, axis=0).astype(np.float32)

        reward = np.asarray(rew_batch, dtype=np.float32)
        terminated = np.asarray(term_batch, dtype=np.bool_)
        truncated = np.asarray(trunc_batch, dtype=np.bool_)

        info = {"success": np.asarray(success_batch, dtype=np.float32)}
        return obs, reward, terminated, truncated, info

    def close(self):
        for e in self.envs:
            try:
                e.close()
            except Exception:
                pass
