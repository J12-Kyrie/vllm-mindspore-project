#!/usr/bin/env python3
# encoding: utf-8
# Copyright 2025 Huawei Technologies Co., Ltd
# Copyright 2024 The vLLM team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""Worker functions"""
import gc
import os
import math
from typing import Tuple, Optional, List, Dict
from dataclasses import dataclass

import torch
import mindspore as ms
from mindspore import Tensor

from vllm.config import VllmConfig
# from vllm.distributed import (
#     ensure_kv_transfer_initialized,
#     ensure_model_parallel_initialized,
#     init_distributed_environment,
#     set_custom_all_reduce,
# )

from vllm.logger import init_logger

from vllm_mindspore.utils import get_valid_dtype
from vllm.model_executor import set_random_seed
from vllm.sequence import SequenceGroupMetadata
from vllm.sampling_params import SamplingParams
from vllm_mindspore.beam_search import BeamSearchSequence, BeamSearchOutput

logger = init_logger(__name__)

@dataclass
class BeamSearchState:
    """Beam search state for tracking sequences."""
    sequences: List[BeamSearchSequence]
    parent_sequence: Optional[BeamSearchSequence] = None
    beam_width: int = 4
    max_tokens: int = 100
    length_penalty: float = 1.0
    temperature: float = 1.0

class BeamSearchWorker:
    """Worker for handling beam search requests."""
    
    def __init__(self, model_runner, cache_engine):
        self.model_runner = model_runner
        self.cache_engine = cache_engine
        self.beam_states: Dict[str, BeamSearchState] = {}
        
    def add_beam_search_request(
        self,
        request_id: str,
        prompt_tokens: List[int],
        beam_width: int = 4,
        max_tokens: int = 100,
        length_penalty: float = 1.0,
        temperature: float = 1.0,
    ) -> None:
        """Add a new beam search request."""
        initial_sequence = BeamSearchSequence(
            tokens=prompt_tokens,
            logprobs=[],
            cum_logprob=0.0
        )
        
        self.beam_states[request_id] = BeamSearchState(
            sequences=[initial_sequence],
            beam_width=beam_width,
            max_tokens=max_tokens,
            length_penalty=length_penalty,
            temperature=temperature
        )
    
    def fork_sequence(
        self,
        request_id: str,
        parent_sequence: BeamSearchSequence,
        new_token: int,
        logprob: float
    ) -> BeamSearchSequence:
        """Fork a new sequence from parent sequence."""
        beam_state = self.beam_states[request_id]
        
        # 创建新的序列
        new_sequence = BeamSearchSequence(
            tokens=parent_sequence.tokens + [new_token],
            logprobs=parent_sequence.logprobs + [{new_token: logprob}],
            cum_logprob=parent_sequence.cum_logprob + logprob
        )
        
        # 更新 beam 状态
        beam_state.sequences.append(new_sequence)
        new_sequence.parent_sequence = parent_sequence
        
        return new_sequence
    
    def get_batch_sequences(self) -> List[BeamSearchSequence]:
        """Get all active sequences for batch processing."""
        all_sequences = []
        for beam_state in self.beam_states.values():
            all_sequences.extend(beam_state.sequences)
        return all_sequences
    
    def update_beam_states(self, request_id: str) -> None:
        """Update beam states after processing a batch."""
        beam_state = self.beam_states[request_id]
        
        # 按累积对数概率排序
        beam_state.sequences.sort(
            key=lambda x: x.cum_logprob / (len(x.tokens) ** beam_state.length_penalty),
            reverse=True
        )
        
        # 保留 top-k 个序列
        beam_state.sequences = beam_state.sequences[:beam_state.beam_width]
    
    def get_beam_search_output(self, request_id: str) -> BeamSearchOutput:
        """Get the final beam search output for a request."""
        beam_state = self.beam_states[request_id]
        return BeamSearchOutput(sequences=beam_state.sequences)

def _prepare_input_for_warmup(model_config, model_runner, cache_engine, is_prefill, is_mtp_model=False):
    bs = 1
    seq_len = model_runner.scheduler_config.max_num_batched_tokens if is_prefill else 1
    dummy_data = model_runner.input_registry.dummy_data_for_profiling(model_config, seq_len, model_runner.mm_registry)
    block_tables = [i for i in range(math.ceil(seq_len / cache_engine.block_size))]
    seqs = [
        SequenceGroupMetadata(
            request_id=str(idx),
            is_prompt=is_prefill,
            seq_data={idx: dummy_data.seq_data},
            sampling_params=SamplingParams(),
            block_tables={idx: block_tables},
            lora_request=None,
            multi_modal_data=None,
            multi_modal_placeholders=None,
        )
        for idx in range(bs)
    ]

    model_input = model_runner.prepare_model_input(seqs)
    block_tables = model_input.attn_metadata.block_tables
    if block_tables is not None and block_tables.numel() <= 0:
        model_input.attn_metadata.block_tables = torch.zeros((1, 1), dtype=torch.int32)

    previous_hidden_states = None if not is_mtp_model else \
        torch.ones([bs, seq_len, model_config.get_hidden_size()], dtype=get_valid_dtype(model_config.dtype))
    return model_input, previous_hidden_states


def _warm_up_model(self) -> None:
    # cache_engine is a list with length equal to the size of pipeline-parallel, and only pp=1 is supported.
    kv_cache = self.cache_engine[0].gpu_cache
    is_mtp_model = self.speculative_config is not None and self.model_config.hf_config.model_type == "deepseek_mtp"
    if is_mtp_model:
        # prefill mtp model
        model_input, previous_hidden_states = _prepare_input_for_warmup(self.model_config, self.model_runner,
                                                                        self.cache_engine[0], True, is_mtp_model)
        self.model_runner.execute_model(model_input, kv_cache, None, previous_hidden_states=previous_hidden_states)

    # warmup for decode
    if self.vllm_config.scheduler_config.is_multi_step:
        model_input, _ = _prepare_input_for_warmup(self.model_config, self.model_runner._base_model_runner,
                                                   self.cache_engine[0], False)
        self.model_runner._base_model_runner.execute_model(model_input, kv_cache, None)
    else:
        model_input, previous_hidden_states = _prepare_input_for_warmup(self.model_config, self.model_runner,
                                                                        self.cache_engine[0], False, is_mtp_model)
        self.model_runner.execute_model(model_input, kv_cache, None, previous_hidden_states=previous_hidden_states)

    torch.cuda.synchronize()

    # Reset the seed to ensure that the random state is not affected by
    # the model initialization and profiling.
    set_random_seed(self.model_config.seed)
