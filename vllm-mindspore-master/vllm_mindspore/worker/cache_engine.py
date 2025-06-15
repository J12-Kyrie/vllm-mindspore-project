#!/usr/bin/env python3
# isort:skip_file
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
"""CacheEngine class for managing the KV cache."""

import gc
import os
from typing import Dict, List, Optional, Tuple

import mindspore as ms
from mindspore import mutable, mint, Tensor

from vllm.config import VllmConfig
from vllm_mindspore.beam_search import BeamSearchSequence, BeamSearchOutput, BeamSearchInstance, get_beam_search_score
from vllm_mindspore.utils import MsKVCache, get_valid_dtype
from vllm.logger import init_logger

logger = init_logger(__name__)


def create_block(shape, dtype, name=None, device=None):
    blocks = mint.empty(shape, dtype=dtype, device=device)
    return blocks


def ms_allocate_kv_cache(
    self,
    num_blocks: int,
    device: str,
) -> List[MsKVCache]:
    """Allocates KV cache on the specified device."""
    kv_cache_shape = self.attn_backend.get_kv_cache_shape(
        num_blocks, self.block_size, self.num_kv_heads, self.head_size)
    kv_cache: List[MsKVCache] = []

    self.dtype = get_valid_dtype(self.dtype)

    for _ in range(self.num_attention_layers):
        device_type = "CPU" if device == "cpu" else "Ascend"
        current_cache = []
        for i in range(kv_cache_shape[0]):
            cache_blocks = create_block(kv_cache_shape[1:],
                                        self.dtype,
                                        device=device_type)
            current_cache.append(mutable(cache_blocks))
        kv_cache.append(mutable(tuple(current_cache)))
    return mutable(kv_cache)


def ms_swap_in(self, src_to_dst: ms.Tensor) -> None:
    for i in range(self.num_attention_layers):
        self.attn_backend.swap_blocks(self.cpu_cache[i], self.gpu_cache[i],
                                      src_to_dst, False)


def ms_swap_out(self, src_to_dst: ms.Tensor) -> None:
    for i in range(self.num_attention_layers):
        self.attn_backend.swap_blocks(self.gpu_cache[i], self.cpu_cache[i],
                                      src_to_dst, True)


class CacheEngine:
    """Cache engine for vLLM."""
    
    def __init__(self, model_config):
        self.model_config = model_config
        self.kv_caches: Dict[str, List[Tuple[Tensor, Tensor]]] = {}
        self.slot_mappings: Dict[str, List[int]] = {}
        
    def fork_kv_cache(
        self,
        request_id: str,
        parent_sequence: BeamSearchSequence,
        child_sequence: BeamSearchSequence
    ) -> None:
        """Fork KV cache from parent sequence to child sequence."""
        if request_id not in self.kv_caches:
            return
            
        parent_kv_cache = self.kv_caches[request_id]
        parent_slot_mapping = self.slot_mappings[request_id]
        
        # 创建新的 KV cache
        child_kv_cache = []
        for layer_idx, (parent_k, parent_v) in enumerate(parent_kv_cache):
            # 复制父序列的 KV cache - 修复 MindSpore Tensor 操作
            parent_slot_tensor = ms.Tensor(parent_slot_mapping, dtype=ms.int32)
            child_k = ms.ops.gather(parent_k, parent_slot_tensor, axis=0)
            child_v = ms.ops.gather(parent_v, parent_slot_tensor, axis=0)
            
            # 创建新的 slot mapping
            child_slot_mapping = list(range(len(parent_slot_mapping)))
            
            # 更新 KV cache
            child_kv_cache.append((child_k, child_v))
            
        # 存储新的 KV cache
        self.kv_caches[request_id] = child_kv_cache
        self.slot_mappings[request_id] = child_slot_mapping
        
    def get_kv_cache(
        self,
        request_id: str
    ) -> Optional[List[Tuple[Tensor, Tensor]]]:
        """Get KV cache for a request."""
        return self.kv_caches.get(request_id)
        
    def update_kv_cache(
        self,
        request_id: str,
        kv_cache: List[Tuple[Tensor, Tensor]],
        slot_mapping: List[int]
    ) -> None:
        """Update KV cache for a request."""
        self.kv_caches[request_id] = kv_cache
        self.slot_mappings[request_id] = slot_mapping
        
    def clear_kv_cache(self, request_id: str) -> None:
        """Clear KV cache for a request."""
        if request_id in self.kv_caches:
            del self.kv_caches[request_id]
        if request_id in self.slot_mappings:
            del self.slot_mappings[request_id]
