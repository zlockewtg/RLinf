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

from dataclasses import dataclass

import torch
from torch import nn

from rlinf.envs.behavior.privileged_obs import (
    PRIVILEGED_TEACHER_OBS_DIM,
    PRIVILEGED_TEACHER_OBS_SLICES,
)


@dataclass(frozen=True)
class PrivilegedTokenGroup:
    name: str
    terms: tuple[str, ...]

    @property
    def dim(self) -> int:
        return sum(
            PRIVILEGED_TEACHER_OBS_SLICES[term].stop
            - PRIVILEGED_TEACHER_OBS_SLICES[term].start
            for term in self.terms
        )


PRIVILEGED_TEACHER_TOKEN_GROUPS: tuple[PrivilegedTokenGroup, ...] = (
    PrivilegedTokenGroup(
        "robot_motion",
        (
            "base_lin_vel",
            "base_ang_vel",
            "projected_gravity",
            "dof_pos",
            "dof_vel",
        ),
    ),
    PrivilegedTokenGroup(
        "action_history",
        (
            "actions",
            "delta_actions",
            "homie_commands",
        ),
    ),
    PrivilegedTokenGroup("stage", ("stage",)),
    PrivilegedTokenGroup(
        "place_context",
        (
            "placement_pos",
            "table_pelvis_transform",
            "hold_fingers_tips_force",
            "hold_obj_transform",
            "hold_hand_object_transform",
            "target_place_pos",
        ),
    ),
    PrivilegedTokenGroup(
        "grasp_context",
        (
            "grasp_fingers_tips_force",
            "grasp_obj_transform",
            "grasp_hand_object_transform",
            "target_lift_pos",
        ),
    ),
)


def split_privileged_obs_by_token_group(
    privileged_obs: torch.Tensor,
) -> list[torch.Tensor]:
    """Split a 226D VIRAL-style observation into semantic token groups."""
    assert privileged_obs.shape[-1] == PRIVILEGED_TEACHER_OBS_DIM, (
        f"privileged_obs expected last dim {PRIVILEGED_TEACHER_OBS_DIM}, "
        f"got {privileged_obs.shape[-1]}"
    )
    groups = []
    for group in PRIVILEGED_TEACHER_TOKEN_GROUPS:
        groups.append(
            torch.cat(
                [
                    privileged_obs[..., PRIVILEGED_TEACHER_OBS_SLICES[term]]
                    for term in group.terms
                ],
                dim=-1,
            )
        )
    return groups


class PrivilegedTeacherTokenProjector(nn.Module):
    """Projects the 226D privileged teacher observation into prefix tokens."""

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int = 256,
        num_tokens: int = 5,
    ):
        super().__init__()
        assert num_tokens == len(PRIVILEGED_TEACHER_TOKEN_GROUPS), (
            f"privileged teacher token count must be "
            f"{len(PRIVILEGED_TEACHER_TOKEN_GROUPS)}, got {num_tokens}"
        )
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_tokens = num_tokens
        self.group_names = tuple(group.name for group in PRIVILEGED_TEACHER_TOKEN_GROUPS)
        self.projectors = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(group.dim),
                    nn.Linear(group.dim, hidden_dim),
                    nn.SiLU(),
                    nn.Linear(hidden_dim, embed_dim),
                )
                for group in PRIVILEGED_TEACHER_TOKEN_GROUPS
            ]
        )

    def forward(self, privileged_obs: torch.Tensor) -> torch.Tensor:
        groups = split_privileged_obs_by_token_group(privileged_obs)
        tokens = [
            projector(group_obs)
            for projector, group_obs in zip(self.projectors, groups, strict=True)
        ]
        return torch.stack(tokens, dim=-2)
