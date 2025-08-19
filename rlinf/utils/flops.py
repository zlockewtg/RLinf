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

from dataclasses import dataclass
from typing import Optional


def lmhead_flops(hidden_size, vocab_size, batch_size, seq_length):
    return 2 * batch_size * hidden_size * vocab_size * seq_length


def qkv_project_flops(
    hidden_size, num_attn_heads, num_kv_heads, batch_size, seq_length
):
    hidden_size_kv = hidden_size // (num_attn_heads // num_kv_heads)
    return 4 * batch_size * seq_length * hidden_size * (hidden_size + hidden_size_kv)


def attention_score_flops(hidden_size, batch_size, seq_length):
    return 4 * batch_size * seq_length * seq_length * hidden_size


def mlp_flops(
    hidden_size, mlp_intermediate_size, batch_size, seq_length, mlp_type: str = "swiglu"
):
    return 6 * batch_size * seq_length * hidden_size * mlp_intermediate_size


def rmsnorm_flops(hidden_size, batch_size, seq_length):
    return 4 * batch_size * seq_length * hidden_size


@dataclass
class ModelConfig:
    num_layers: int = 0
    """Number of transformer layers in a transformer block."""

    hidden_size: int = 0
    """Transformer hidden size."""

    num_attention_heads: int = 0
    """Number of transformer attention heads."""

    num_query_groups: Optional[int] = None
    """Number of query groups for group query attention. If None, normal attention is used."""

    ffn_hidden_size: Optional[int] = None
    """Transformer Feed-Forward Network hidden size. This is set to 4*hidden_size
    if not provided."""

    override_vocab_size: int = 0


class FLOPSCalculator:
    def __init__(self, model_config: ModelConfig):
        self.model_config = model_config

    def flops(self, batch_size, prompt_length, decode_length):
        prefill_decode_flops = self._calculate_prefill_flops(
            batch_size=batch_size, prompt_length=prompt_length
        ) + self._calculate_decode_flops(
            batch_size=batch_size,
            prompt_length=prompt_length,
            decode_length=decode_length,
        )

        prefill_total_flops = self._calculate_prefill_flops(
            batch_size=batch_size, prompt_length=prompt_length + decode_length
        )

        return prefill_decode_flops, prefill_total_flops

    def _calculate_prefill_flops(self, batch_size, prompt_length):
        """
        Calculate the FLOPs for prefill phases of a transformer model.

        Args:
            batch_size (int): Global Batch size.
            prompt_length (int): Length of the input prompt.

        Returns:
            float: The total FLOPs for the prefill phase.

        Calculation:
            The computation is based on the following formula:
            FLO_{prefill} = L \times (4BS_{prompt}H(H + H_{kv} + S_{prompt}) + 6BS_{prompt}HI + 8BS_{prompt}H) + 2BS_{prompt}HV
        """
        num_layers = self.model_config.num_layers
        hidden_size = self.model_config.hidden_size
        num_attn_heads = self.model_config.num_attention_heads
        num_query_groups = self.model_config.num_query_groups
        num_kv_heads = num_attn_heads // num_query_groups
        mlp_intermediate_size = self.model_config.ffn_hidden_size
        vocab_size = self.model_config.override_vocab_size

        prefill_flops = (
            num_layers
            * (
                qkv_project_flops(
                    hidden_size, num_attn_heads, num_kv_heads, batch_size, prompt_length
                )
                + attention_score_flops(hidden_size, batch_size, prompt_length)
                + mlp_flops(
                    hidden_size,
                    mlp_intermediate_size,
                    batch_size,
                    prompt_length,
                    "swiglu",
                )
                + 2 * rmsnorm_flops(hidden_size, batch_size, prompt_length)
            )
            + rmsnorm_flops(hidden_size, batch_size, prompt_length)
            + lmhead_flops(hidden_size, vocab_size, batch_size, prompt_length)
        )

        return prefill_flops

    def _calculate_decode_flops(self, batch_size, prompt_length, decode_length):
        """
        Calculate the FLOPs for decode phases of a transformer model.

        Args:
            batch_size (int): Global Batch size.
            prompt_length (int): Length of the input prompt.
            decode_length (int): Length of the generated response.

        Returns:
            float: The total FLOPs for the decode phase.

        Calculation:
            The computation is based on the following formula:
            FLO_{decode\\_step} = L \times (4BH(H + H_{kv}) + 4BHS_{cache} + 6BHI + 8BH) +  2BHV`
            where:
            - L is the decode length
            - H is the hidden size
            - H_{kv} is the hidden size for key-value heads
            - S_{cache} is the sequence length of kv cache
            - I is the intermediate size of the MLP
            - V is the vocabulary size
            - B is the batch size

        Note:
            We multiply the fixed part (independent of decode_length) by decode_length directly.
            For the variable part (attention score calculation part), we use the arithmetic progression formula
            to calculate directly instead of gradually accumulating.
        """

        num_layers = self.model_config.num_layers
        hidden_size = self.model_config.hidden_size
        num_attn_heads = self.model_config.num_attention_heads
        num_query_groups = self.model_config.num_query_groups
        num_kv_heads = num_attn_heads // num_query_groups
        mlp_intermediate_size = self.model_config.ffn_hidden_size
        vocab_size = self.model_config.override_vocab_size

        decode_flops = decode_length * (
            num_layers
            * (
                qkv_project_flops(
                    hidden_size, num_attn_heads, num_kv_heads, batch_size, 1
                )
                + mlp_flops(hidden_size, mlp_intermediate_size, batch_size, 1)
                + 2 * rmsnorm_flops(hidden_size, batch_size, 1)
            )
            + lmhead_flops(hidden_size, vocab_size, batch_size, 1)
        )
        +4 * batch_size * num_layers * hidden_size * (
            decode_length + 2 * prompt_length
        ) * decode_length / 2
        # attention score with kv cache

        return decode_flops
