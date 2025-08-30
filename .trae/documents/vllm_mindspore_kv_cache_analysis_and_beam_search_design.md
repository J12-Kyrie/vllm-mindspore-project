\#!/usr/bin/env python3

# encoding: utf-8

# Copyright 2025 Huawei Technologies Co., Ltd

# Copyright 2024 The vLLM team.

#

# Licensed under the Apache License, Version 2.0 (the "License");

# you may not use this file except in compliance with the License.

# You may obtain a copy of the License at

#

# <http://www.apache.org/licenses/LICENSE-2.0>

#

# Unless required by applicable law or agreed to in writing, software

# distributed under the License is distributed on an "AS IS" BASIS,

# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

# See the License for the specific language governing permissions and

# limitations under the License.

# ============================================================================

"""NPU-optimized beam search sampling strategy for vllm-mindspore."""

import math
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
from collections import defaultdict

import torch
import mindspore as ms
from mindspore import mutable, mint, ops

from vllm.beam\_search import BeamSearchSequence, BeamSearchOutput, get\_beam\_search\_score
from vllm.sampling\_params import SamplingParams
from vllm.sequence import SequenceGroupMetadata, SequenceData, SequenceStatus
from vllm.logger import init\_logger
from vllm\_mindspore.utils import MsKVCache, get\_valid\_dtype
from vllm\_mindspore.engine.attention.backends.ms\_attn import MSAttentionMetadata

logger = init\_logger(__name__)

@dataclass
class BeamState:
"""State information for a single beam candidate."""
beam\_id: int
parent\_beam\_id: Optional\[int]
tokens: List\[int]
logprobs: List\[Dict\[int, float]]
cumulative\_logprob: float
cache\_block\_ids: List\[int]
is\_finished: bool = False
finish\_reason: Optional\[str] = None

class KVCacheBeamTracker:
"""Manages KV cache allocation and sharing for beam search candidates."""

```
def __init__(self, cache_engine, block_size: int):
    self.cache_engine = cache_engine
    self.block_size = block_size
    self.beam_cache_mapping: Dict[int, List[int]] = {}  # beam_id -> block_ids
    self.block_ref_count: Dict[int, int] = defaultdict(int)  # block_id -> ref_count
    self.available_blocks: Set[int] = set()
    self.next_block_id = 0
    
def allocate_cache_blocks(self, beam_id: int, num_blocks: int) -> List[int]:
    """Allocate cache blocks for a new beam."""
    block_ids = []
    
    # Try to reuse available blocks first
    while len(block_ids) < num_blocks and self.available_blocks:
        block_id = self.available_blocks.pop()
        block_ids.append(block_id)
        self.block_ref_count[block_id] = 1
        
    # Allocate new blocks if needed
    while len(block_ids) < num_blocks:
        block_id = self.next_block_id
        self.next_block_id += 1
        block_ids.append(block_id)
        self.block_ref_count[block_id] = 1
        
    self.beam_cache_mapping[beam_id] = block_ids
    return block_ids

def share_cache_blocks(self, parent_beam_id: int, child_beam_id: int) -> None:
    """Share cache blocks between parent and child beams."""
    if parent_beam_id not in self.beam_cache_mapping:
        return
        
    parent_blocks = self.beam_cache_mapping[parent_beam_id]
    # Share all parent blocks with child
    self.beam_cache_mapping[child_beam_id] = parent_blocks.copy()
    
    # Increment reference count for shared blocks
    for block_id in parent_blocks:
        self.block_ref_count[block_id] += 1

def copy_on_write(self, beam_id: int, block_index: int) -> int:
    """Implement copy-on-write for cache blocks when beam diverges."""
    if beam_id not in self.beam_cache_mapping:
        return -1
        
    current_blocks = self.beam_cache_mapping[beam_id]
    if block_index >= len(current_blocks):
        return -1
        
    old_block_id = current_blocks[block_index]
    
    # If block is shared, create a copy
    if self.block_ref_count[old_block_id] > 1:
        new_block_id = self.next_block_id
        self.next_block_id += 1
        
        # Copy cache content (this would interface with actual cache engine)
        self._copy_cache_block(old_block_id, new_block_id)
        
        # Update mappings
        current_blocks[block_index] = new_block_id
        
```

