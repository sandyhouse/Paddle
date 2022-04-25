# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import paddle
import paddle.nn.functional as F
import numpy as np
from .naive_gate import NaiveGate
from ..utils import limit_by_capacity


class GShardGate(NaiveGate):
    def __init__(self,
                 d_model,
                 num_expert,
                 world_size,
                 topk=2,
                 capacity=(1.2, 2.4),
                 random_routing=True,
                 group=None):
        super().__init__(d_model, num_expert, world_size, topk)
        self.capacity = capacity
        self.random_routing = random_routing
        self.group = group

    def forward(self, x):
        topk_val, topk_idx, gate_score = super().forward(
            x, return_all_scores=True)

        return topk_val, topk_idx, gate_score
