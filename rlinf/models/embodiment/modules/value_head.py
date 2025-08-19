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

import torch.nn as nn


class ValueHead(nn.Module):
    def __init__(self, hidden_size, output_dim=1):
        super().__init__()
        self.head_l1 = nn.Linear(hidden_size, 512)
        self.head_act1 = nn.GELU()
        self.head_l2 = nn.Linear(512, 128)
        self.head_act2 = nn.GELU()
        self.head_l3 = nn.Linear(128, output_dim, bias=False)

        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_normal_(
            self.head_l1.weight, mode="fan_out", nonlinearity="relu"
        )
        nn.init.zeros_(self.head_l1.bias)
        nn.init.kaiming_normal_(
            self.head_l2.weight, mode="fan_out", nonlinearity="relu"
        )
        nn.init.zeros_(self.head_l2.bias)
        nn.init.normal_(self.head_l3.weight, mean=0.0, std=0.02)

    def forward(self, x):
        x = self.head_act1(self.head_l1(x))
        x = self.head_act2(self.head_l2(x))
        x = self.head_l3(x)
        return x
