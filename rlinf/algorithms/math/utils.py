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

import torch


def get_attention_masks_and_position_ids(
    prompt_lengths, response_lengths, total_seq_len, max_prompt_len
):
    B = prompt_lengths.size(0)
    total_len = total_seq_len

    # Attention Mask
    arange_ids = (
        torch.arange(total_len, device=prompt_lengths.device).unsqueeze(0).expand(B, -1)
    )
    prompt_start = max_prompt_len - prompt_lengths
    response_end = max_prompt_len + response_lengths
    attention_mask = (arange_ids >= prompt_start.unsqueeze(1)) & (
        arange_ids < response_end.unsqueeze(1)
    )

    # Position IDs
    position_ids = torch.zeros_like(arange_ids, device=prompt_lengths.device)
    seq_indices = torch.arange(total_len, device=prompt_lengths.device).unsqueeze(
        0
    )  # [1, total_len]

    valid_positions = seq_indices >= prompt_start.unsqueeze(1)  # [B, total_len]
    relative_positions = seq_indices - prompt_start.unsqueeze(1)
    position_ids[valid_positions] = relative_positions[valid_positions]

    return attention_mask, position_ids
