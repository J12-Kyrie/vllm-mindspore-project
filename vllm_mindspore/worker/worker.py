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
from typing import Tuple, Optional, Dict, Any

import torch

from vllm.config import VllmConfig
from vllm.distributed import (
    ensure_kv_transfer_initialized,
    ensure_model_parallel_initialized,
    init_distributed_environment,
    set_custom_all_reduce,
)

from vllm.logger import init_logger

from vllm_mindspore.utils import get_valid_dtype
from vllm.model_executor import set_random_seed
from vllm.sequence import SequenceGroupMetadata
from vllm.sampling_params import SamplingParams
from vllm_mindspore.beam_search import (
    NPUBeamSearchSampler,
    create_npu_beam_search_sampler
)
from vllm_mindspore.worker.model_runner import (
    BeamSearchModelRunner,
    create_beam_search_model_runner
)
from vllm_mindspore.worker.cache_engine import (
    BeamSearchCacheEngine,
    create_beam_search_cache_engine
)


logger = init_logger(__name__)


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


class BeamSearchWorker:
    """Worker extension for beam search support."""
    
    def __init__(self, base_worker):
        self.base_worker = base_worker
        self.beam_search_model_runner: Optional[BeamSearchModelRunner] = None
        self.beam_search_cache_engine: Optional[BeamSearchCacheEngine] = None
        self._beam_search_enabled = False
    
    def enable_beam_search(self, beam_width: int, max_length: int, 
                          length_penalty: float = 1.0) -> None:
        """Enable beam search for this worker."""
        # Create beam search model runner
        self.beam_search_model_runner = create_beam_search_model_runner(
            self.base_worker.model_runner
        )
        
        # Create sampling params for beam search
        from vllm.sampling_params import SamplingParams
        sampling_params = SamplingParams(
            n=beam_width,
            max_tokens=max_length,
            length_penalty=length_penalty,
            use_beam_search=True
        )
        
        # Get cache engine and model config
        cache_engine = self.base_worker.cache_engine[0] if self.base_worker.cache_engine else None
        model_config = self.base_worker.model_config
        
        self.beam_search_model_runner.enable_beam_search(
            sampling_params=sampling_params,
            cache_engine=cache_engine,
            model_config=model_config
        )
        
        # Create beam search cache engine
        if hasattr(self.base_worker, 'cache_engine') and self.base_worker.cache_engine:
            self.beam_search_cache_engine = create_beam_search_cache_engine(
                self.base_worker.cache_engine[0]  # Assuming pp=1
            )
            self.beam_search_cache_engine.enable_beam_search_cache()
        
        self._beam_search_enabled = True
        logger.info(f"Worker enabled beam search with width={beam_width}, max_length={max_length}")
    
    def disable_beam_search(self) -> None:
        """Disable beam search for this worker."""
        if self.beam_search_model_runner:
            self.beam_search_model_runner.disable_beam_search()
            self.beam_search_model_runner = None
        
        if self.beam_search_cache_engine:
            self.beam_search_cache_engine.disable_beam_search_cache()
            self.beam_search_cache_engine = None
        
        self._beam_search_enabled = False
        logger.info("Worker disabled beam search")
    
    def is_beam_search_enabled(self) -> bool:
        """Check if beam search is currently enabled."""
        return (self._beam_search_enabled and 
                self.beam_search_model_runner is not None and
                self.beam_search_model_runner.is_beam_search_enabled())
    
    def execute_model_with_beam_search(self, model_input, kv_caches, 
                                      intermediate_tensors=None, **kwargs):
        """Execute model with beam search if enabled."""
        if not self.is_beam_search_enabled():
            return self.base_worker.model_runner.execute_model(
                model_input, kv_caches, intermediate_tensors, **kwargs
            )
        
        # Use beam search cache if available
        if self.beam_search_cache_engine:
            # Use the base cache engine's kv_caches since beam search cache wraps it
            kv_caches = self.beam_search_cache_engine.gpu_cache
        
        # Execute with beam search
        return self.beam_search_model_runner.execute_with_beam_search(
            model_input, kv_caches, intermediate_tensors
        )
    
    def get_beam_search_status(self) -> Dict[str, Any]:
        """Get current beam search status."""
        if not self.is_beam_search_enabled():
            return {'enabled': False}
        
        status = {'enabled': True}
        
        # Get model runner status
        if self.beam_search_model_runner:
            model_status = self.beam_search_model_runner.get_beam_search_results()
            if model_status:
                status.update(model_status)
        
        # Get cache engine status
        if self.beam_search_cache_engine:
            cache_status = {
                'beam_cache_enabled': self.beam_search_cache_engine.beam_cache_enabled,
                'active_beams': len(self.beam_search_cache_engine.beam_block_mapping),
                'allocated_blocks': len(self.beam_search_cache_engine.block_ref_count)
            }
            status['cache'] = cache_status
        
        return status
    
    def warmup_beam_search(self) -> None:
        """Warmup beam search components."""
        if not self.is_beam_search_enabled():
            return
        
        logger.info("Warming up beam search components...")
        
        # Prepare dummy input for beam search warmup
        model_config = self.base_worker.model_config
        cache_engine = self.base_worker.cache_engine[0] if self.base_worker.cache_engine else None
        
        if cache_engine:
            model_input, _ = _prepare_input_for_warmup(
                model_config, 
                self.base_worker.model_runner, 
                cache_engine, 
                is_prefill=True
            )
            
            # Execute warmup with beam search
            kv_cache = cache_engine.gpu_cache
            self.execute_model_with_beam_search(model_input, kv_cache)
            
            torch.cuda.synchronize()
            logger.info("Beam search warmup completed")


def create_beam_search_worker(base_worker) -> BeamSearchWorker:
    """Factory function to create a beam search enabled worker."""
    return BeamSearchWorker(base_worker)


def prepare_beam_search_input_for_warmup(model_config, model_runner, cache_engine, 
                                        beam_width: int, is_prefill: bool = True):
    """Prepare input for beam search warmup."""
    bs = beam_width  # Use beam width as batch size for warmup
    seq_len = model_runner.scheduler_config.max_num_batched_tokens if is_prefill else 1
    dummy_data = model_runner.input_registry.dummy_data_for_profiling(
        model_config, seq_len, model_runner.mm_registry
    )
    
    block_tables = [i for i in range(math.ceil(seq_len / cache_engine.block_size))]
    seqs = [
        SequenceGroupMetadata(
            request_id=f"beam_{idx}",
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
        model_input.attn_metadata.block_tables = torch.zeros((bs, 1), dtype=torch.int32)
    
    return model_input
