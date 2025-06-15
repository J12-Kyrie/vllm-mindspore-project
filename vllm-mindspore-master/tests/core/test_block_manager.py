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

import pytest
from vllm_mindspore.core.block_manager import BlockManager
from vllm_mindspore.sequence import Sequence


def test_block_manager_fork():
    # 创建BlockManager实例
    block_manager = BlockManager()
    
    # 创建父序列和子序列，使用字典作为inputs
    parent_seq = Sequence(seq_id=1, inputs={"type": "token", "prompt_token_ids": [1, 2, 3]}, block_size=10)
    child_seq1 = Sequence(seq_id=2, inputs={"type": "token", "prompt_token_ids": [1, 2, 3]}, block_size=10)
    child_seq2 = Sequence(seq_id=3, inputs={"type": "token", "prompt_token_ids": [1, 2, 3]}, block_size=10)
    
    # 为父序列分配物理块
    block_manager.allocate(parent_seq.seq_id, [1, 2, 3])
    
    # 调用fork方法
    block_manager.fork(parent_seq, [child_seq1, child_seq2])
    
    # 验证子序列的block_table与父序列相同
    assert block_manager.get_block_table(child_seq1.seq_id) == block_manager.get_block_table(parent_seq.seq_id)
    assert block_manager.get_block_table(child_seq2.seq_id) == block_manager.get_block_table(parent_seq.seq_id)
    
    # 验证被共享的物理块的引用计数是否正确
    for block_num in block_manager.get_block_table(parent_seq.seq_id):
        assert block_manager.get_ref_count(block_num) == 3  # 父序列 + 2个子序列 