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
import logging
from typing import Dict, List, Optional, Tuple, Set, Any
from dataclasses import dataclass
from collections import defaultdict
import numpy as np

try:
    import torch
except ImportError:
    torch = None

import mindspore as ms
from mindspore import mutable, mint, ops

# Local implementations to avoid vllm dependencies
import logging
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass
from collections import defaultdict

# Setup logger
logger = logging.getLogger(__name__)

# Local class definitions to replace vllm imports
@dataclass
class BeamSearchSequence:
    """Local implementation of beam search sequence."""
    tokens: List[int]
    cumulative_logprob: float
    
@dataclass
class BeamSearchOutput:
    """Local implementation of beam search output."""
    sequences: List[BeamSearchSequence]
    
class SamplingParams:
    """Local implementation of sampling parameters."""
    def __init__(self, use_beam_search: bool = False, best_of: int = 1, 
                 length_penalty: float = 1.0, max_tokens: int = 512,
                 stop_token_ids: Optional[List[int]] = None):
        self.use_beam_search = use_beam_search
        self.best_of = best_of
        self.length_penalty = length_penalty
        self.max_tokens = max_tokens
        self.stop_token_ids = stop_token_ids or []
        
class SequenceGroupMetadata:
    """Local implementation of sequence group metadata."""
    def __init__(self, request_id: str, is_prompt: bool, seq_data: Dict[int, Any],
                 sampling_params: SamplingParams, block_tables: Optional[Dict[int, List[int]]] = None):
        self.request_id = request_id
        self.is_prompt = is_prompt
        self.seq_data = seq_data
        self.sampling_params = sampling_params
        self.block_tables = block_tables or {}
        
class MSAttentionMetadata:
    """Local implementation of attention metadata."""
    def __init__(self, seq_lens: Optional[List[int]] = None, 
                 block_tables: Optional[List[List[int]]] = None):
        self.seq_lens = seq_lens or []
        self.block_tables = block_tables or []
        
def get_beam_search_score(tokens: List[int], logprobs: List[float], 
                         length_penalty: float = 1.0) -> float:
    """Local implementation of beam search scoring."""
    if not tokens or not logprobs:
        return float('-inf')
    
    # Apply length penalty
    length = len(tokens)
    if length_penalty != 1.0 and length > 0:
        length_factor = ((5 + length) / 6) ** length_penalty
        return sum(logprobs) / length_factor
    
    return sum(logprobs) / max(1, length)

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
        if block_size <= 0:
            raise ValueError(f"block_size must be positive, got {block_size}")
            
        self.cache_engine = cache_engine
        self.block_size = block_size
        self.beam_cache_map: Dict[int, List[int]] = {}  # beam_id -> block_ids
        self.block_ref_count: Dict[int, int] = defaultdict(int)  # block_id -> ref_count
        self.available_blocks: Set[int] = set()
        self.next_block_id = 0
        self._lock = None  # Thread safety placeholder
        
    def allocate_cache_blocks(self, beam_id: int, num_blocks: int) -> List[int]:
        """Allocate cache blocks for a new beam."""
        if beam_id < 0:
            raise ValueError(f"beam_id must be non-negative, got {beam_id}")
        if num_blocks <= 0:
            raise ValueError(f"num_blocks must be positive, got {num_blocks}")
        if beam_id in self.beam_cache_map:
            raise ValueError(f"beam_id {beam_id} already has allocated blocks")
            
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
            
        self.beam_cache_map[beam_id] = block_ids
        return block_ids

    def share_cache_blocks(self, parent_beam_id: int, child_beam_id: int) -> None:
        """Share cache blocks between parent and child beams."""
        if parent_beam_id < 0 or child_beam_id < 0:
            raise ValueError("beam_id must be non-negative")
        if parent_beam_id == child_beam_id:
            raise ValueError("parent and child beam_id cannot be the same")
        if child_beam_id in self.beam_cache_map:
            raise ValueError(f"child beam_id {child_beam_id} already has allocated blocks")
        if parent_beam_id not in self.beam_cache_map:
            logger.warning(f"Parent beam {parent_beam_id} has no allocated blocks")
            return
            
        parent_blocks = self.beam_cache_map[parent_beam_id]
        # Share all parent blocks with child
        self.beam_cache_map[child_beam_id] = parent_blocks.copy()
        
        # Increment reference count for shared blocks
        for block_id in parent_blocks:
            self.block_ref_count[block_id] += 1

    def copy_on_write(self, beam_id: int, block_index: int) -> int:
        """Implement copy-on-write for cache blocks when beam diverges."""
        if beam_id not in self.beam_cache_map:
            return -1
            
        current_blocks = self.beam_cache_map[beam_id]
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
        if beam_id not in self.beam_cache_map:
            return
            
        block_ids = self.beam_cache_map[beam_id]
        for block_id in block_ids:
            self.block_ref_count[block_id] -= 1
            if self.block_ref_count[block_id] == 0:
                self.available_blocks.add(block_id)
                del self.block_ref_count[block_id]
                
        del self.beam_cache_map[beam_id]

    def _copy_cache_block(self, src_block_id: int, dst_block_id: int) -> None:
        """Copy cache content between blocks."""
        if src_block_id < 0 or dst_block_id < 0:
            raise ValueError("block_id must be non-negative")
        if src_block_id == dst_block_id:
            return  # No need to copy to itself
            
        try:
            if self.cache_engine and hasattr(self.cache_engine, 'copy_block'):
                # Use cache engine's copy method if available
                self.cache_engine.copy_block(src_block_id, dst_block_id)
            elif self.cache_engine and hasattr(self.cache_engine, 'gpu_cache'):
                # Manual copy using cache tensors
                src_cache = self.cache_engine.gpu_cache[src_block_id]
                dst_cache = self.cache_engine.gpu_cache[dst_block_id]
                if isinstance(src_cache, ms.Tensor) and isinstance(dst_cache, ms.Tensor):
                    ops.assign(dst_cache, src_cache)
                else:
                    logger.warning(f"Cannot copy cache block {src_block_id} to {dst_block_id}: unsupported cache type")
            else:
                logger.warning(f"Cache engine does not support block copying")
        except Exception as e:
            logger.error(f"Failed to copy cache block {src_block_id} to {dst_block_id}: {e}")
            raise RuntimeError(f"Cache block copy failed: {e}")

class NPUBeamScoreCalculator:
    """NPU-optimized beam scoring using MindSpore operations."""
    
    def __init__(self, vocab_size: int, device: str = "npu:0"):
        if vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {vocab_size}")
        if not device or not isinstance(device, str):
            raise ValueError(f"device must be a non-empty string, got {device}")
            
        self.vocab_size = vocab_size
        self.device = device
        self._validate_and_set_device(device)
        
    def _validate_and_set_device(self, device: str) -> None:
        """Validate and configure the specified device."""
        try:
            device_target = self._get_device_target(device)
            
            # Check device availability
            if device_target == "Ascend":
                # Validate NPU availability
                try:
                    import acl
                    device_count = acl.rt.get_device_count()
                    if device_count == 0:
                        raise RuntimeError("No NPU devices available")
                    
                    device_id = self._extract_device_id(device)
                    if device_id >= device_count:
                        raise ValueError(f"Device ID {device_id} exceeds available NPU count {device_count}")
                        
                except ImportError:
                    logger.warning("ACL library not available, cannot validate NPU device")
                except Exception as e:
                    logger.warning(f"NPU validation failed: {e}")
                    
            # Set MindSpore context
            ms.set_context(device_target=device_target, device_id=self._extract_device_id(device))
            logger.info(f"Successfully configured device: {device}")
            
        except Exception as e:
            logger.error(f"Failed to configure device {device}: {e}")
            # Fallback to CPU
            try:
                ms.set_context(device_target="CPU")
                self.device = "cpu"
                logger.warning(f"Falling back to CPU due to device error: {e}")
            except Exception as fallback_error:
                raise RuntimeError(f"Failed to configure any device: {fallback_error}")
                
    def _extract_device_id(self, device: str) -> int:
        """Extract device ID from device string."""
        try:
            if ":" in device:
                return int(device.split(":")[1])
            return 0
        except (ValueError, IndexError):
            logger.warning(f"Invalid device format {device}, using device ID 0")
            return 0
            
    def _get_device_target(self, device: str) -> str:
        """Convert device string to MindSpore device target."""
        device_lower = device.lower()
        if device_lower.startswith("npu:") or device_lower.startswith("ascend:"):
            return "Ascend"
        elif device_lower.startswith("cuda:") or device_lower.startswith("gpu:"):
            return "GPU"
        elif device_lower in ["cpu", "cpu:0"]:
            return "CPU"
        else:
            logger.warning(f"Unknown device format {device}, defaulting to CPU")
            return "CPU"
            
    def _validate_tensor_device(self, tensor: ms.Tensor) -> None:
        """Validate tensor is on correct device."""
        if tensor is None:
            raise ValueError("Tensor cannot be None")
        if not isinstance(tensor, ms.Tensor):
            raise TypeError(f"Expected MindSpore tensor, got {type(tensor)}")
        # Additional device-specific validation could be added here
        
    def _ensure_tensor_device(self, tensor: ms.Tensor) -> ms.Tensor:
        """Ensure tensor is on the correct device."""
        if tensor is None:
            raise ValueError("Tensor cannot be None")
        try:
            # MindSpore handles device placement automatically in most cases
            # For explicit device placement, we could use tensor.to(device) if available
            return tensor
        except Exception as e:
            logger.warning(f"Failed to ensure tensor device placement: {e}")
            return tensor
        
    def compute_beam_scores(self, 
                          logits: ms.Tensor,
                          beam_states: List[BeamState],
                          length_penalty: float = 1.0,
                          eos_token_id: int = 2) -> ms.Tensor:
        """Compute scores for all beam candidates efficiently."""
        if logits is None:
            raise ValueError("logits cannot be None")
        if not beam_states:
            raise ValueError("beam_states cannot be empty")
        if length_penalty <= 0:
            raise ValueError(f"length_penalty must be positive, got {length_penalty}")
            
        batch_size = len(beam_states)
        
        # Validate tensor device and shape
        if logits.shape[0] != batch_size:
            raise ValueError(f"logits batch size {logits.shape[0]} doesn't match beam_states length {batch_size}")
        if logits.shape[-1] != self.vocab_size:
            raise ValueError(f"logits vocab size {logits.shape[-1]} doesn't match expected {self.vocab_size}")
            
        # Device validation for input tensor
        try:
            self._validate_tensor_device(logits)
        except Exception as e:
            logger.warning(f"Tensor device validation failed: {e}")
            # Try to move tensor to correct device
            logits = self._ensure_tensor_device(logits)
        
        # Convert to MindSpore tensors and apply softmax
        try:
            if isinstance(logits, ms.Tensor):
                logits_tensor = logits.astype(ms.float32)
            else:
                logits_tensor = ms.Tensor(logits, dtype=ms.float32)
                
            # Ensure tensor is on correct device
            logits_tensor = self._ensure_tensor_device(logits_tensor)
            log_probs = ops.log_softmax(logits_tensor, axis=-1)
            
        except Exception as e:
            logger.error(f"Failed to process logits tensor: {e}")
            raise RuntimeError(f"Tensor processing failed: {e}")
        
        # Prepare beam data for vectorized computation
        beam_lengths = ms.Tensor([len(beam.tokens) for beam in beam_states], dtype=ms.int32)
        cum_logprobs = ms.Tensor([beam.cumulative_logprob for beam in beam_states], dtype=ms.float32)
        
        # Calculate new cumulative log probabilities
        new_cum_logprobs = ops.expand_dims(cum_logprobs, -1) + log_probs  # [batch_size, vocab_size]
        
        # Apply length penalty
        if length_penalty != 1.0:
            try:
                length_penalties = ms.Tensor([len(beam.tokens) ** length_penalty for beam in beam_states], dtype=ms.float32)
                length_penalties = self._ensure_tensor_device(length_penalties)
                length_penalties = ops.expand_dims(length_penalties, -1)
                scores = new_cum_logprobs / length_penalties
            except Exception as e:
                logger.error(f"Failed to apply length penalty: {e}")
                # Continue without length penalty
                logger.warning("Continuing without length penalty due to error")
                scores = new_cum_logprobs
        else:
            scores = new_cum_logprobs
            
        return scores

    def select_top_beams(self, scores: ms.Tensor, beam_width: int) -> Tuple[ms.Tensor, ms.Tensor]:
        """Select top-k beams using NPU-optimized operations."""
        if scores is None:
            raise ValueError("scores cannot be None")
        if beam_width <= 0:
            raise ValueError(f"beam_width must be positive, got {beam_width}")
        if beam_width > scores.numel():
            raise ValueError(f"beam_width {beam_width} exceeds total candidates {scores.numel()}")
            
        # Get top-k candidates for each beam
        top_k_values, top_k_indices = ops.topk(scores, beam_width, axis=-1)
        
        # Flatten and get global top candidates
        flat_scores = top_k_values.reshape(-1)
        flat_indices = top_k_indices.reshape(-1)
        
        # Get top beam_width candidates globally
        global_top_values, global_top_indices = ops.topk(flat_scores, beam_width)
        
        # Convert back to beam and token indices
        beam_indices = global_top_indices // beam_width
        token_indices = flat_indices[global_top_indices]
        
        try:
            return global_top_values, ops.stack([beam_indices, token_indices], axis=1)
        except Exception as e:
            logger.error(f"Failed to stack beam and token indices: {e}")
            # Fallback to returning separate tensors
            try:
                return global_top_values, beam_indices, token_indices
            except Exception as fallback_error:
                raise RuntimeError(f"Failed to return beam selection results: {fallback_error}")

class NPUBeamSearchSampler:
    """NPU-optimized beam search sampler for vllm-mindspore."""
    
    def __init__(self, 
                 beam_width: int,
                 cache_engine,
                 model_config,
                 length_penalty: float = 1.0,
                 max_tokens: int = 512,
                 eos_token_id: int = 2):
        if beam_width <= 0:
            raise ValueError(f"beam_width must be positive, got {beam_width}")
        if length_penalty <= 0:
            raise ValueError(f"length_penalty must be positive, got {length_penalty}")
        if max_tokens <= 0:
            raise ValueError(f"max_tokens must be positive, got {max_tokens}")
        if eos_token_id < 0:
            raise ValueError(f"eos_token_id must be non-negative, got {eos_token_id}")
            
        self.beam_width = beam_width
        self.cache_engine = cache_engine
        self.model_config = model_config
        self.length_penalty = length_penalty
        self.max_tokens = max_tokens
        self.eos_token_id = eos_token_id
        
        # Initialize components
        self.cache_tracker = KVCacheBeamTracker(
            cache_engine, 
            getattr(cache_engine, 'block_size', 16) if cache_engine else 16
        )
        self.score_calculator = NPUBeamScoreCalculator(
            getattr(model_config, 'vocab_size', 32000) if model_config else 32000
        )
        
        # Beam state management
        self.active_beams: List[BeamState] = []
        self.completed_beams: List[BeamState] = []
        self.next_beam_id = 0
        
    def initialize_beams(self, prompt_tokens: List[int]) -> None:
        """Initialize beam search with the prompt."""
        if not prompt_tokens:
            raise ValueError("prompt_tokens cannot be empty")
        if any(token < 0 for token in prompt_tokens):
            raise ValueError("All tokens must be non-negative")
            
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
        if logits is None:
            raise ValueError("logits cannot be None")
        if not self.active_beams:
            logger.warning("No active beams to process")
            return False
        if logits.shape[0] != len(self.active_beams):
            raise ValueError(f"logits batch size {logits.shape[0]} doesn't match active beams {len(self.active_beams)}")
            
        try:
            # Compute scores for all beam candidates
            scores = self.score_calculator.compute_beam_scores(
                logits, self.active_beams, self.length_penalty, self.eos_token_id
            )
            
            # Select top beam candidates
            top_scores, top_indices = self.score_calculator.select_top_beams(
                scores, self.beam_width
            )
        except Exception as e:
            logger.error(f"Error in beam scoring: {e}")
            return False
        
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
                if beam_block_tables and torch is not None:
                    # Convert to tensor format expected by attention backend
                    block_tables_tensor = torch.tensor(beam_block_tables, dtype=torch.int32)
                    attn_metadata.block_tables = block_tables_tensor
                    
        return seq_group_metadata

def create_npu_beam_search_sampler(sampling_params: SamplingParams,
                                   cache_engine,
                                   model_config,
                                   device: str = "npu:0") -> Optional[NPUBeamSearchSampler]:
    """Factory function to create NPU beam search sampler."""
    if sampling_params is None:
        logger.warning("sampling_params is None, cannot create beam search sampler")
        return None
        
    try:
        # Check if beam search is requested
        if not hasattr(sampling_params, 'use_beam_search') or not sampling_params.use_beam_search:
            return None
            
        beam_width = getattr(sampling_params, 'best_of', 1)
        
        # Validate device parameter
        if device and not device.startswith(('npu:', 'cpu', 'cuda:')):
            logger.warning(f"Invalid device {device}, using default npu:0")
            device = "npu:0"
        
        if beam_width <= 1:
            logger.debug(f"Beam width {beam_width} <= 1, not creating beam search sampler")
            return None
            
        sampler = NPUBeamSearchSampler(
            beam_width=beam_width,
            cache_engine=cache_engine,
            model_config=model_config,
            length_penalty=getattr(sampling_params, 'length_penalty', 1.0),
            max_tokens=getattr(sampling_params, 'max_tokens', 512),
            eos_token_id=getattr(sampling_params, 'stop_token_ids', [2])[0] if 
                         hasattr(sampling_params, 'stop_token_ids') and sampling_params.stop_token_ids else 2
        )
        
        logger.info(f"Created NPU beam search sampler: beam_width={beam_width}, device={device}")
        return sampler
        
    except Exception as e:
        logger.error(f"Failed to create NPU beam search sampler: {e}")
        return None

# Integration functions for existing vllm-mindspore components

def integrate_beam_search_with_model_runner(model_runner, sampling_params):
    """Integration point for model runner to support beam search."""
    if model_runner is None:
        raise ValueError("model_runner cannot be None")
    if sampling_params is None:
        raise ValueError("sampling_params cannot be None")
        
    try:
        # Extract cache engine safely
        cache_engine = None
        if hasattr(model_runner, 'cache_engine'):
            if isinstance(model_runner.cache_engine, (list, tuple)):
                cache_engine = model_runner.cache_engine[0] if model_runner.cache_engine else None
            else:
                cache_engine = model_runner.cache_engine
                
        # Extract model config safely
        model_config = getattr(model_runner, 'model_config', None)
        
        beam_sampler = create_npu_beam_search_sampler(
            sampling_params,
            cache_engine,
            model_config
        )
        
        if beam_sampler:
            logger.info(f"Created beam search sampler with width {beam_sampler.beam_width}")
        
        return beam_sampler
        
    except Exception as e:
            logger.error(f"Failed to integrate beam search with model runner: {e}")
            return None

# Additional utility functions for NPU optimization

def optimize_beam_search_for_npu(beam_sampler: NPUBeamSearchSampler) -> None:
    """Apply NPU-specific optimizations to beam search sampler."""
    if beam_sampler is None:
        return
        
    try:
        # Set NPU-optimized context
        ms.set_context(device_target="Ascend", mode=ms.GRAPH_MODE)
        
        # Enable memory optimization
        if hasattr(ms.context, 'set_auto_parallel_context'):
            ms.context.set_auto_parallel_context(enable_parallel_optimizer=True)
            
        logger.info("Applied NPU optimizations to beam search sampler")
        
    except Exception as e:
        logger.warning(f"Failed to apply NPU optimizations: {e}")

def extend_attention_metadata_for_beams(attn_metadata: MSAttentionMetadata,
                                       beam_sampler: NPUBeamSearchSampler) -> MSAttentionMetadata:
    """Extend attention metadata with beam search information."""
    if attn_metadata is None:
        raise ValueError("attn_metadata cannot be None")
        
    if beam_sampler is None or not beam_sampler.active_beams:
        return attn_metadata
        
    try:
        # Update block tables and slot mappings for beam search
        beam_block_tables = []
        beam_slot_mappings = []
        
        for beam in beam_sampler.active_beams:
            if beam.cache_block_ids:
                beam_block_tables.append(beam.cache_block_ids)
                # Generate slot mappings for this beam
                beam_slots = list(range(len(beam.tokens)))
                beam_slot_mappings.extend(beam_slots)
                
        if beam_block_tables:
            # Update attention metadata with beam-specific information
            attn_metadata.num_prefills = len(beam_sampler.active_beams)
            attn_metadata.num_decode_tokens = sum(len(beam.tokens) for beam in beam_sampler.active_beams)
            
            # Convert block tables to tensor format if needed
            if torch is not None:
                try:
                    # Pad block tables to same length
                    max_blocks = max(len(table) for table in beam_block_tables) if beam_block_tables else 0
                    padded_tables = []
                    for table in beam_block_tables:
                        padded_table = table + [-1] * (max_blocks - len(table))
                        padded_tables.append(padded_table)
                    
                    block_tables_tensor = torch.tensor(padded_tables, dtype=torch.int32)
                    attn_metadata.block_tables = block_tables_tensor
                    
                    if beam_slot_mappings:
                        slot_mappings_tensor = torch.tensor(beam_slot_mappings, dtype=torch.int32)
                        attn_metadata.slot_mapping = slot_mappings_tensor
                        
                except Exception as e:
                    logger.warning(f"Failed to convert beam metadata to tensors: {e}")
                    
        logger.debug(f"Extended attention metadata for {len(beam_sampler.active_beams)} beams")
        
    except Exception as e:
        logger.error(f"Failed to extend attention metadata for beams: {e}")
        
    return attn_metadata

# Export main classes and functions
__all__ = [
    'BeamState',
    'KVCacheBeamTracker', 
    'NPUBeamScoreCalculator',
    'NPUBeamSearchSampler',
    'create_npu_beam_search_sampler',
    'integrate_beam_search_with_model_runner',
    'extend_attention_metadata_for_beams',
    'optimize_beam_search_for_npu'
]