#!/usr/bin/env python3
# encoding: utf-8
# Copyright 2025 Huawei Technologies Co., Ltd
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
# ============================================================================

from typing import List
from vllm_mindspore.sequence import SequenceGroup


class Scheduler:
    def __init__(self):
        self.sequence_groups: List[SequenceGroup] = []

    def add_sequence_group(self, sequence_group: SequenceGroup) -> None:
        """添加序列组到调度器"""
        self.sequence_groups.append(sequence_group)

    def _schedule(self) -> None:
        """调度器的核心循环"""
        for sequence_group in self.sequence_groups:
            if sequence_group.is_finished():
                # 释放资源
                self.sequence_groups.remove(sequence_group)
            else:
                # 处理未完成的序列组
                running_sequences = sequence_group.get_running_sequences()
                # 这里可以添加处理逻辑，例如更新状态或执行其他操作 