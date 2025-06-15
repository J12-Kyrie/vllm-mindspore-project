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

from typing import Dict, List
from collections import defaultdict
from vllm_mindspore.sequence import Sequence


class BlockManager:
    def __init__(self):
        # 用于存储每个序列的物理块映射表 {seq_id: [block_num1, block_num2]}
        self.block_tables: Dict[int, List[int]] = {}
        # 用于存储每个物理块的引用计数 {block_num: ref_count}
        self.ref_counts: Dict[int, int] = defaultdict(int)

    def fork(self, parent_seq: Sequence, child_seqs: List[Sequence]) -> None:
        """
        为子序列创建共享父序列前缀的KV缓存。
        """
        # 获取父序列的block_table
        parent_block_table = self.block_tables.get(parent_seq.seq_id, [])
        
        # 为每个子序列浅拷贝父序列的block_table
        for child_seq in child_seqs:
            self.block_tables[child_seq.seq_id] = parent_block_table.copy()
            
            # 为所有被共享的物理块增加引用计数
            for block_num in parent_block_table:
                self.ref_counts[block_num] += 1

    def allocate(self, seq_id: int, block_nums: List[int]) -> None:
        """分配物理块给序列"""
        self.block_tables[seq_id] = block_nums
        for block_num in block_nums:
            self.ref_counts[block_num] += 1

    def free(self, seq_id: int) -> None:
        """释放序列占用的物理块"""
        if seq_id in self.block_tables:
            for block_num in self.block_tables[seq_id]:
                self.ref_counts[block_num] -= 1
                if self.ref_counts[block_num] == 0:
                    del self.ref_counts[block_num]
            del self.block_tables[seq_id]

    def get_block_table(self, seq_id: int) -> List[int]:
        """获取序列的block_table"""
        return self.block_tables.get(seq_id, [])

    def get_ref_count(self, block_num: int) -> int:
        """获取物理块的引用计数"""
        return self.ref_counts.get(block_num, 0) 