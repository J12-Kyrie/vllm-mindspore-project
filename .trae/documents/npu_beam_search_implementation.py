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

"""NPU-optimized beam search sampling strategy for vllm-mindspore."""

import math
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
from collections import defaultdict

import torch
import mindspore as ms
from mindspore import mutable, mint, ops

from vllm.beam_search import BeamSearchSequence, BeamSearchOutput, get_beam_search_score
from vllm.sampling_params import SamplingParams
from vllm.sequence import SequenceGroupMetadata, SequenceData, SequenceStatus
from vllm.logger import init_logger
from vllm_mindspore.utils import MsKVCache, get_valid_dtype
from vllm_mindspore.engine.attention.backends.ms_attn import MSAttentionMetadata

logger = init_logger(__name__)


@dataclass
class BeamState:
    """State information for a single beam candidate."""
    beam_id: int
    parent_beam_id: Optional[int]
    tokens: List[int]
    logprobs: List[Dict[int, float]]
    cumulative_logprob: float
    cache_block_ids: List[int]
    is_finished: bool = False
    finish_reason: Optional[str] = None


class KVCacheBeamTracker:
    """Manages KV cache allocation and sharing for beam search candidates."""
    
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
            self.block_ref_count[old_block_id] -= 1
            self.block_ref_count[new_block_id] = 1
            
            return new_block_id
        
        return old_block_id
    
    def release_beam_cache(self, beam_id: int) -> None:
        """Release cache blocks for a finished or pruned beam."""
        if beam_id not in self.beam_cache_mapping:
            return
            
        block_ids = self.beam_cache_mapping[beam_id]
        for block_id in block_ids:
            self.block_ref_count[block_id] -= 1
            if self.block_ref_count[block_id] == 0:
                self.available_blocks.add(block_id)
                del self.block_ref_count[block_id]
                
        del self.beam_cache_mapping[beam_id]
    
    def _copy_cache_block(self, src_block_id: int, dst_block_id: int) -> None:
        """Copy cache content between blocks (placeholder for actual implementation)."""
        # This would interface with the actual cache engine to copy KV cache data
        pass


class NPUBeamScoreCalculator:
    """NPU-optimized beam scoring using MindSpore operations."""
    
    def __init__(self, vocab_size: int, device: str = "Ascend"):
        self.vocab_size = vocab_size
        self.device = device
        
    def compute_beam_scores(self, 
                          logits: ms.Tensor,
                          beam_states: List[BeamState],
                          length_penalty: float = 1.0,
                          eos_token_id: int = 2) -> ms.Tensor:
        """Compute scores for all beam candidates efficiently."""
        batch_size = len(beam_states)
        
        # Convert logits to log probabilities
        log_probs = ops.log_softmax(logits, axis=-1)  # [batch_size, vocab_size]
        
        # Prepare beam data for vectorized computation
        beam_lengths = ms.Tensor([len(beam.tokens) for beam in beam_states], dtype=ms.int32)
        cum_logprobs = ms.Tensor([beam.cumulative_logprob for beam in beam_states], dtype=ms.float32)
        
        # Expand log_probs for all possible next tokens
        expanded_log_probs = log_probs.unsqueeze(1).expand(-1, self.vocab_size, -1)  # [batch_size, vocab_size, vocab_size]
        
        # Calculate new cumulative log probabilities
        new_cum_logprobs = cum_logprobs.unsqueeze(-1) + log_probs  # [batch_size, vocab_size]
        
        # Apply length penalty
        if length_penalty != 1.0:
            # Length penalty: (5 + len) / (5 + 1) ** length_penalty
            length_factor = ops.pow((5.0 + beam_lengths.float() + 1) / 6.0, length_penalty)
            length_factor = length_factor.unsqueeze(-1)  # [batch_size, 1]
            scores = new_cum_logprobs / length_factor
        else:
            scores = new_cum_logprobs
            
        return scores
    
    def select_top_beams(self, scores: ms.Tensor, beam_width: int) -> Tuple[ms.Tensor, ms.Tensor]:
        """Select top-k beams using NPU-optimized operations."""
        # Flatten scores to select globally best candidates
        flat_scores = scores.view(-1)  # [batch_size * vocab_size]
        
        # Get top-k scores and indices
        top_scores, top_indices = ops.topk(flat_scores, beam_width, largest=True)
        
        # Convert flat indices back to (beam_idx, token_idx)
        beam_indices = top_indices // scores.shape[-1]
        token_indices = top_indices % scores.shape[-1]
        
        return top_scores, ops.stack([beam_indices, token_indices], axis=1)


class NPUBeamSearchSampler:
    """NPU-optimized beam search sampler for vllm-mindspore."""
    
    def __init__(self, 
                 beam_width: int,
                 cache_engine,
                 model_config,
                 length_penalty: float = 1.0,
                 max_tokens: int = 512,
                 eos_token_id: int = 2):
        self.beam_width = beam_width
        self.cache_engine = cache_engine
        self.model_config = model_config
        self.length_penalty = length_penalty
        self.max_tokens = max_tokens
        self.eos_token_id = eos_token_id
        
        # Initialize components
        self.cache_tracker = KVCacheBeamTracker(
            cache_engine, 
            cache_engine.block_size if hasattr(cache_engine, 'block_size') else 16
        )
        self.score_calculator = NPUBeamScoreCalculator(
            model_config.vocab_size if hasattr(model_config, 'vocab_size') else 32000
        )
        
        # Beam state management
        self.active_beams: List[BeamState] = []
        self.completed_beams: List[BeamState] = []
        self.next_beam_id = 0
        
    def initialize_beams(self, prompt_tokens: List[int]) -> None:
        """Initialize beam search with the prompt."""
        # Create initial beam
        initial_beam = BeamState(
            beam_id=self.next_beam_id,
            parent_beam_id=None,
            tokens=prompt_tokens.copy(),
            logprobs=[],
            cumulative_logprob=0.0,
            cache_block_ids=[]
        )
        self.next_beam_id += 1
        
        # Allocate cache blocks for initial beam
        num_blocks_needed = math.ceil(len(prompt_tokens) / self.cache_tracker.block_size)
        initial_beam.cache_block_ids = self.cache_tracker.allocate_cache_blocks(
            initial_beam.beam_id, num_blocks_needed
        )
        
        self.active_beams = [initial_beam]
        self.completed_beams = []
        
    def step(self, logits: ms.Tensor) -> bool:
        """Perform one step of beam search."""
        if not self.active_beams:
            return False
            
        # Compute scores for all beam candidates
        scores = self.score_calculator.compute_beam_scores(
            logits, self.active_beams, self.length_penalty, self.eos_token_id
        )
        
        # Select top beam candidates
        top_scores, top_indices = self.score_calculator.select_top_beams(
            scores, self.beam_width
        )
        
        # Create new beam states
        new_beams = []
        for i in range(len(top_scores)):
            beam_idx = top_indices[i][0].item()
            token_idx = top_indices[i][1].item()
            score = top_scores[i].item()
            
            if beam_idx >= len(self.active_beams):
                continue
                
            parent_beam = self.active_beams[beam_idx]
            
            # Create new beam state
            new_beam = BeamState(
                beam_id=self.next_beam_id,
                parent_beam_id=parent_beam.beam_id,
                tokens=parent_beam.tokens + [token_idx],
                logprobs=parent_beam.logprobs + [{token_idx: score - parent_beam.cumulative_logprob}],
                cumulative_logprob=score,
                cache_block_ids=[]
            )
            self.next_beam_id += 1
            
            # Handle cache block allocation and sharing
            self.cache_tracker.share_cache_blocks(parent_beam.beam_id, new_beam.beam_id)
            
            # Check if beam is finished
            if (token_idx == self.eos_token_id or 
                len(new_beam.tokens) >= self.max_tokens):
                new_beam.is_finished = True
                new_beam.finish_reason = "stop" if token_idx == self.eos_token_id else "length"
                self.completed_beams.append(new_beam)
            else:
                new_beams.append(new_beam)
                
        # Clean up old beam cache
        for beam in self.active_beams:
            self.cache_tracker.release_beam_cache(beam.beam_id)
            
        # Update active beams
        self.active_beams = new_beams
        
        # Continue if we have active beams and haven't reached max completed beams
        return len(self.active_beams) > 0 and len(self.completed_beams) < self.beam_width
    
    def finalize(self) -> BeamSearchOutput:
        """Finalize beam search and return results."""
        # Add any remaining active beams to completed beams
        for beam in self.active_beams:
            beam.is_finished = True
            beam.finish_reason = "length"
            self.completed_beams.append(beam)
            
        # Sort completed beams by score
        self.completed_beams.sort(
            key=lambda b: get_beam_search_score(
                b.tokens, b.cumulative_logprob, self.eos_token_id, self.length_penalty
            ),
            reverse=True
        )
        
        # Convert to BeamSearchSequence format
        sequences = []
        for beam in self.completed_beams[:self.beam_width]:
            sequence = BeamSearchSequence(
                tokens=beam.tokens,
                logprobs=beam.logprobs,
                cum_logprob=beam.cumulative_logprob,
                finish_reason=beam.finish_reason
            )
            sequences.append(sequence)
            
        # Clean up all cache blocks
        for beam in self.completed_beams:
            self.cache_tracker.release_beam_cache(beam.beam_id)
        for beam in self.active_beams:
            self.cache_tracker.release_beam_cache(beam.beam_id)
            
        return BeamSearchOutput(sequences=sequences)
    
    def prepare_model_input(self, 
                          seq_group_metadata: SequenceGroupMetadata) -> SequenceGroupMetadata:
        """Prepare model input with beam search metadata."""
        # This method would integrate with the existing model runner
        # to prepare input tensors with proper attention metadata
        
        # Update attention metadata with beam information
        if hasattr(seq_group_metadata, 'attn_metadata'):
            attn_metadata = seq_group_metadata.attn_metadata
            if isinstance(attn_metadata, MSAttentionMetadata):
                # Add beam-specific block tables and slot mappings
                beam_block_tables = []
                for beam in self.active_beams:
                    beam_block_tables.append(beam.cache_block_ids)
                
                # Update block tables with beam information
                if beam_block_tables:
                    # Convert to tensor format expected by attention backend
                    block_tables_tensor = torch.tensor(beam_block_tables, dtype=torch.int32)
                    attn_metadata.block_tables = block_tables_tensor
                    
        return seq_group_metadata


def create_npu_beam_search_sampler(sampling_params: SamplingParams,
                                  cache_engine,
                                  model_config) -> Optional[NPUBeamSearchSampler]:
    """Factory function to create NPU beam search sampler."""
    # Check if beam search is requested
    if not hasattr(sampling_params, 'use_beam_search') or not sampling_params.use_beam_search:
        return None
        
    beam_width = getattr(sampling_params, 'best_of', 1)
    if beam_width <= 1:
        return None
        
    return NPUBeamSearchSampler(
        beam_width=beam_width,
        cache_engine=cache_engine,
        model_config=model_config,
        length_penalty=getattr(sampling_params, 'length_penalty', 1.0),
        max_tokens=getattr(sampling_params, 'max_tokens', 512),
        eos_token_id=getattr(sampling_params, 'stop_token_ids', [2])[0] if 
                     hasattr(sampling_params, 'stop_token_ids') and sampling_params.stop_token_ids else 2
    )


# Integration functions for existing vllm-mindspore components

def integrate_beam_search_with_model_runner(model_runner, sampling_params):
    """Integration point for model runner to support beam search."""
    # This function would be called from model_runner.py to set up beam search
    beam_sampler = create_npu_beam_search_sampler(
        sampling_params, 
        model_runner.cache_engine[0] if hasattr(model_runner, 'cache_engine') else None,
        model_runner.model_config
    )
    return beam_sampler


def extend_attention_metadata_for_beams(attn_metadata: MSAttentionMetadata,
                                       beam_sampler: NPUBeamSearchSampler) -> MSAttentionMetadata:
    """Extend attention metadata with beam search information."""
    if beam_sampler and beam_sampler.active_beams:
        # Update block tables and slot mappings for beam search
        beam_block_tables = []
        for beam in beam_sampler.active_beams:
            beam_block_tables.extend(beam.cache_block_ids)
            
        if beam_block_tables:
            # Update attention metadata with beam-specific information
            attn_metadata.num_prefills = len(beam_sampler.active_beams)
            # Additional beam-specific metadata updates would go here
            
    return attn_metadata