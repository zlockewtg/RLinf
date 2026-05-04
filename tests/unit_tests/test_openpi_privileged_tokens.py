import torch
from types import SimpleNamespace

from rlinf.envs.behavior.privileged_obs import PRIVILEGED_TEACHER_OBS_DIM
from rlinf.models.embodiment.openpi.openpi_action_model import (
    OpenPi0ForRLActionPrediction,
)
from rlinf.models.embodiment.openpi.privileged_tokens import (
    PRIVILEGED_TEACHER_TOKEN_GROUPS,
    PrivilegedTeacherTokenProjector,
    split_privileged_obs_by_token_group,
)


def test_privileged_token_group_dims_sum_to_viral_obs_dim():
    assert len(PRIVILEGED_TEACHER_TOKEN_GROUPS) == 5
    assert sum(group.dim for group in PRIVILEGED_TEACHER_TOKEN_GROUPS) == (
        PRIVILEGED_TEACHER_OBS_DIM
    )
    assert tuple(group.name for group in PRIVILEGED_TEACHER_TOKEN_GROUPS) == (
        "robot_motion",
        "action_history",
        "stage",
        "place_context",
        "grasp_context",
    )


def test_split_privileged_obs_by_token_group_preserves_batch_dim():
    obs = torch.randn(3, PRIVILEGED_TEACHER_OBS_DIM)
    groups = split_privileged_obs_by_token_group(obs)

    assert len(groups) == 5
    assert [group.shape for group in groups] == [
        (3, spec.dim) for spec in PRIVILEGED_TEACHER_TOKEN_GROUPS
    ]


def test_privileged_teacher_token_projector_outputs_prefix_tokens():
    obs = torch.randn(2, PRIVILEGED_TEACHER_OBS_DIM)
    projector = PrivilegedTeacherTokenProjector(
        embed_dim=64,
        hidden_dim=32,
        num_tokens=5,
    )

    tokens = projector(obs)

    assert tokens.shape == (2, 5, 64)
    assert torch.isfinite(tokens).all()


def test_openpi_obs_processor_keeps_proprio_and_privileged_states():
    policy = object.__new__(OpenPi0ForRLActionPrediction)
    policy.config = SimpleNamespace(
        config_name="pi05_behavior",
        use_privileged_teacher_obs=True,
    )
    env_obs = {
        "main_images": torch.zeros(2, 8, 8, 3, dtype=torch.uint8),
        "wrist_images": torch.zeros(2, 2, 8, 8, 3, dtype=torch.uint8),
        "task_descriptions": ["turn on the radio", "turn on the radio"],
        "states": torch.ones(2, PRIVILEGED_TEACHER_OBS_DIM),
        "proprio_states": torch.ones(2, 32),
        "privileged_states": torch.ones(2, PRIVILEGED_TEACHER_OBS_DIM),
    }

    processed = policy.obs_processor(env_obs)

    assert processed["observation/state"] is env_obs["proprio_states"]
    assert processed["observation/privileged_state"] is env_obs["privileged_states"]
