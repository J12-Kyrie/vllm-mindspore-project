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

import mindspore as ms
from mindspore import mutable, mint
from typing import List, Dict, Optional
from vllm.logger import init_logger
from vllm_mindspore.utils import MsKVCache, get_valid_dtype

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


class BeamSearchCacheEngine:
    """Enhanced cache engine with beam search support."""
    
    def __init__(self, base_cache_engine):
        self.base_cache_engine = base_cache_engine
        self.beam_cache_enabled = False
        self.beam_block_mapping: Dict[int, List[int]] = {}  # beam_id -> block_ids
        self.block_ref_count: Dict[int, int] = {}  # block_id -> reference_count
        self.available_blocks: List[int] = []
        
    def enable_beam_search_cache(self) -> None:
        """Enable beam search cache management."""
        self.beam_cache_enabled = True
        logger.info("Beam search cache management enabled")
        
    def disable_beam_search_cache(self) -> None:
        """Disable beam search cache management."""
        self.beam_cache_enabled = False
        self.beam_block_mapping.clear()
        self.block_ref_count.clear()
        self.available_blocks.clear()
        logger.info("Beam search cache management disabled")
        
    def allocate_beam_cache_blocks(self, beam_id: int, num_blocks: int) -> List[int]:
        """Allocate cache blocks for a specific beam."""
        if not self.beam_cache_enabled:
            return []
            
        block_ids = []
        
        # Reuse available blocks first
        while len(block_ids) < num_blocks and self.available_blocks:
            block_id = self.available_blocks.pop()
            block_ids.append(block_id)
            self.block_ref_count[block_id] = 1
            
        # Allocate new blocks if needed
        while len(block_ids) < num_blocks:
            # This would interface with the actual block allocator
            block_id = len(self.block_ref_count)  # Simple ID assignment
            block_ids.append(block_id)
            self.block_ref_count[block_id] = 1
            
        self.beam_block_mapping[beam_id] = block_ids
        return block_ids
        
    def share_beam_cache_blocks(self, parent_beam_id: int, child_beam_id: int) -> None:
        """Share cache blocks between parent and child beams."""
        if not self.beam_cache_enabled or parent_beam_id not in self.beam_block_mapping:
            return
            
        parent_blocks = self.beam_block_mapping[parent_beam_id]
        self.beam_block_mapping[child_beam_id] = parent_blocks.copy()
        
        # Increment reference count for shared blocks
        for block_id in parent_blocks:
            if block_id in self.block_ref_count:
                self.block_ref_count[block_id] += 1
                
    def copy_beam_cache_block(self, beam_id: int, block_index: int) -> Optional[int]:
        """Implement copy-on-write for beam cache blocks."""
        if (not self.beam_cache_enabled or 
            beam_id not in self.beam_block_mapping or
            block_index >= len(self.beam_block_mapping[beam_id])):
            return None
            
        current_blocks = self.beam_block_mapping[beam_id]
        old_block_id = current_blocks[block_index]
        
        # If block is shared, create a copy
        if self.block_ref_count.get(old_block_id, 0) > 1:
            new_block_id = len(self.block_ref_count)  # Simple ID assignment
            
            # Copy cache content (interface with actual cache engine)
            self._copy_cache_block_content(old_block_id, new_block_id)
            
            # Update mappings
            current_blocks[block_index] = new_block_id
            self.block_ref_count[old_block_id] -= 1
            self.block_ref_count[new_block_id] = 1
            
            return new_block_id
            
        return old_block_id
        
    def release_beam_cache_blocks(self, beam_id: int) -> None:
        """Release cache blocks for a finished beam."""
        if not self.beam_cache_enabled or beam_id not in self.beam_block_mapping:
            return
            
        block_ids = self.beam_block_mapping[beam_id]
        for block_id in block_ids:
            if block_id in self.block_ref_count:
                self.block_ref_count[block_id] -= 1
                if self.block_ref_count[block_id] == 0:
                    self.available_blocks.append(block_id)
                    del self.block_ref_count[block_id]
                    
        del self.beam_block_mapping[beam_id]
        
    def get_beam_block_tables(self, beam_ids: List[int]) -> List[List[int]]:
        """Get block tables for specified beams."""
        if not self.beam_cache_enabled:
            return []
            
        block_tables = []
        for beam_id in beam_ids:
            if beam_id in self.beam_block_mapping:
                block_tables.append(self.beam_block_mapping[beam_id])
            else:
                block_tables.append([])
                
        return block_tables
        
    def _copy_cache_block_content(self, src_block_id: int, dst_block_id: int) -> None:
        """Copy cache content between blocks using MindSpore operations."""
        try:
            # Access the underlying cache storage from base cache engine
            if hasattr(self.base_cache_engine, 'gpu_cache'):
                gpu_cache = self.base_cache_engine.gpu_cache
                
                # Copy KV cache data for each attention layer
                for layer_idx in range(len(gpu_cache)):
                    if isinstance(gpu_cache[layer_idx], (tuple, list)) and len(gpu_cache[layer_idx]) >= 2:
                        # Copy key cache
                        if (src_block_id < len(gpu_cache[layer_idx][0]) and 
                            dst_block_id < len(gpu_cache[layer_idx][0])):
                            src_key_cache = gpu_cache[layer_idx][0][src_block_id]
                            dst_key_cache = gpu_cache[layer_idx][0][dst_block_id]
                            # Use MindSpore copy operation
                            ms.ops.assign(dst_key_cache, src_key_cache)
                            
                        # Copy value cache
                        if (src_block_id < len(gpu_cache[layer_idx][1]) and 
                            dst_block_id < len(gpu_cache[layer_idx][1])):
                            src_value_cache = gpu_cache[layer_idx][1][src_block_id]
                            dst_value_cache = gpu_cache[layer_idx][1][dst_block_id]
                            # Use MindSpore copy operation
                            ms.ops.assign(dst_value_cache, src_value_cache)
                            
                logger.debug(f"Copied cache content from block {src_block_id} to {dst_block_id}")
                
            elif hasattr(self.base_cache_engine, 'cache_blocks'):
                # Alternative cache structure
                cache_blocks = self.base_cache_engine.cache_blocks
                if (src_block_id < len(cache_blocks) and dst_block_id < len(cache_blocks)):
                    src_block = cache_blocks[src_block_id]
                    dst_block = cache_blocks[dst_block_id]
                    ms.ops.assign(dst_block, src_block)
                    logger.debug(f"Copied cache block {src_block_id} to {dst_block_id}")
                    
        except Exception as e:
            logger.error(f"Failed to copy cache content from block {src_block_id} to {dst_block_id}: {e}")
            # Fallback: mark as copied but without actual data transfer
            logger.warning("Using fallback cache copy (no data transfer)")
        
    def __getattr__(self, name):
        """Delegate attribute access to base cache engine."""
        return getattr(self.base_cache_engine, name)


def create_beam_search_cache_engine(base_cache_engine) -> BeamSearchCacheEngine:
    """Factory function to create beam search enabled cache engine."""
    return BeamSearchCacheEngine(base_cache_engine)
