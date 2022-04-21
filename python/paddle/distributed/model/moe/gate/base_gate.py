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

import paddle.nn as nn


class BaseGate(nn.Layer):
    def __init__(self, num_expert, world_size):
        super().__init__()
        self.world_size = world_size
        self.num_expert = num_expert
        self.tot_expert = world_size * num_expert
        self.loss = None

    def forward(self, x):
        raise NotImplementedError("Please implement the forward function.")

    def set_loss(self, loss):
        self.loss = loss

    def set_fuse_gshard(self, score, local_expert_count):
        self.score = score
        self.local_expert_count = local_expert_count

    def get_fuse_gshard(self):
        return self.score, self.local_expert_count

    def get_loss(self, clear=True):
        loss = self.loss
        if clear:
            self.loss = None
        return loss
