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

import os
import sys
import pytest
import mindspore as ms
from mindspore import Tensor
import numpy as np

# 添加项目根目录到 Python 路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
sys.path.insert(0, project_root)

# 设置环境变量
from .set_env import EnvVarManager
env_manager = EnvVarManager()
env_manager.setup_cpu_environment()

# 设置 MindSpore 上下文为 CPU
ms.set_context(mode=ms.GRAPH_MODE, device_target="CPU")

from vllm_mindspore.beam_search import BeamSearchSequence, BeamSearchOutput, BeamSearchInstance, get_beam_search_score

def test_beam_search_sequence():
    """测试 BeamSearchSequence 的基本功能"""
    # 创建序列
    tokens = [1, 2, 3]
    logprobs = [{"1": 0.1, "2": 0.2}, {"3": 0.3}]
    sequence = BeamSearchSequence(
        tokens=tokens,
        logprobs=logprobs,
        cum_logprob=0.0
    )
    
    # 验证属性
    assert sequence.tokens == tokens
    assert sequence.logprobs == logprobs
    assert sequence.cum_logprob == 0.0
    assert sequence.text is None
    assert sequence.finish_reason is None

def test_beam_search_output():
    """测试 BeamSearchOutput 的基本功能"""
    # 创建序列
    sequence1 = BeamSearchSequence(
        tokens=[1, 2, 3],
        logprobs=[],
        cum_logprob=0.0
    )
    sequence2 = BeamSearchSequence(
        tokens=[1, 2, 4],
        logprobs=[],
        cum_logprob=-0.5
    )
    
    # 创建输出
    output = BeamSearchOutput(sequences=[sequence1, sequence2])
    
    # 验证属性
    assert len(output.sequences) == 2
    assert output.sequences[0] == sequence1
    assert output.sequences[1] == sequence2

def test_beam_search_instance():
    """测试 BeamSearchInstance 的基本功能"""
    # 创建实例
    prompt_tokens = [1, 2, 3]
    instance = BeamSearchInstance(prompt_tokens=prompt_tokens)
    
    # 验证初始状态
    assert len(instance.beams) == 1
    assert instance.beams[0].tokens == prompt_tokens
    assert instance.beams[0].cum_logprob == 0.0
    assert len(instance.completed) == 0

def test_beam_search_score():
    """测试 beam search 分数计算"""
    # 测试基本分数计算
    tokens = [1, 2, 3]
    cum_logprob = -1.0
    eos_token_id = 3
    score = get_beam_search_score(tokens, cum_logprob, eos_token_id)
    assert score == -1.0  # 因为最后一个token是EOS，所以长度减1
    
    # 测试长度惩罚
    length_penalty = 0.5
    score = get_beam_search_score(tokens, cum_logprob, eos_token_id, length_penalty)
    assert score == -1.0 / (2 ** 0.5)  # 长度2的0.5次方

if __name__ == "__main__":
    pytest.main(["-v", "test_beam_search_ms.py"]) 