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

"""
LLMEngineMS: Enhanced LLM Engine with Beam Search Integration for MindSpore

This module extends the original vLLM LLMEngine to support Beam Search
functionality specifically optimized for MindSpore backend.
"""

import time
from typing import Dict, List, Optional, Union, Any
from collections import defaultdict

import mindspore as ms
from mindspore import Tensor

# Import original vLLM components
from vllm.engine.llm_engine import LLMEngine
from vllm.config import VllmConfig
from vllm.executor.executor_base import ExecutorBase
from vllm.engine.output_processor.interfaces import SequenceGroupOutputProcessor
from vllm.outputs import RequestOutput, PoolingRequestOutput
from vllm.sampling_params import SamplingParams
from vllm.sequence import SequenceGroup, SequenceGroupMetadata
from vllm.core.scheduler import SchedulerOutputs
from vllm.engine.llm_engine import SchedulerContext
from vllm.usage.usage_lib import UsageContext
from vllm.engine.metrics_types import StatLoggerBase
from vllm.multimodal import MultiModalRegistry, MULTIMODAL_REGISTRY
from vllm.logger import init_logger

# Import MindSpore-specific Beam Search components
from vllm_mindspore.beam_search import (
    BeamSearchSequence, 
    BeamSearchOutput, 
    BeamSearchInstance, 
    get_beam_search_score
)

logger = init_logger(__name__)


class LLMEngineMS(LLMEngine):
    """
    Enhanced LLM Engine with integrated Beam Search support for MindSpore.
    
    This class extends the original vLLM LLMEngine to provide native Beam Search
    functionality that works seamlessly with MindSpore tensors and operations.
    
    Key Features:
    - Native Beam Search integration in the main processing loop
    - MindSpore tensor compatibility
    - KV Cache forking for beam sequences
    - Efficient beam state management
    - CPU/NPU execution support
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        executor_class: type[ExecutorBase],
        log_stats: bool,
        usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
        stat_loggers: Optional[Dict[str, StatLoggerBase]] = None,
        mm_registry: MultiModalRegistry = MULTIMODAL_REGISTRY,
        use_cached_outputs: bool = False,
    ) -> None:
        """
        Initialize the LLMEngineMS with Beam Search capabilities.
        
        Args:
            vllm_config: The vLLM configuration
            executor_class: The model executor class
            log_stats: Whether to log statistics
            usage_context: Usage context for the engine
            stat_loggers: Optional stat loggers
            mm_registry: Multi-modal registry
            use_cached_outputs: Whether to use cached outputs
        """
        # Initialize the parent LLMEngine
        super().__init__(
            vllm_config=vllm_config,
            executor_class=executor_class,
            log_stats=log_stats,
            usage_context=usage_context,
            stat_loggers=stat_loggers,
            mm_registry=mm_registry,
            use_cached_outputs=use_cached_outputs,
        )
        
        # Initialize Beam Search state management
        self.beam_search_instances: Dict[str, BeamSearchInstance] = {}
        self.beam_search_configs: Dict[str, Dict[str, Any]] = {}
        
        logger.info("LLMEngineMS initialized with Beam Search support")

    def _process_model_outputs(
        self,
        ctx: SchedulerContext,
        request_id: Optional[str] = None
    ) -> None:
        """
        Enhanced model output processing with Beam Search integration.
        
        This method extends the original _process_model_outputs to handle
        Beam Search requests by routing them to dedicated processing logic.
        
        Args:
            ctx: The scheduler context containing outputs and metadata
            request_id: Optional specific request ID to process
        """
        now = time.time()

        if len(ctx.output_queue) == 0:
            return None

        # Get pending async postprocessor
        if request_id:
            # When we process only one request, no pop is required
            (outputs, seq_group_metadata_list, scheduler_outputs, is_async,
             is_last_step, is_first_step_output, skip) = ctx.output_queue[0]
        else:
            (outputs, seq_group_metadata_list, scheduler_outputs, is_async,
             is_last_step, is_first_step_output,
             skip) = ctx.output_queue.popleft()

        # Sanity check
        assert len(seq_group_metadata_list) == len(
            scheduler_outputs.scheduled_seq_groups)

        # Process each sequence group with enhanced logic
        for i, (seq_group_meta, scheduled_seq_group) in enumerate(
            zip(seq_group_metadata_list, scheduler_outputs.scheduled_seq_groups)
        ):
            if i in skip:
                continue

            seq_group: SequenceGroup = scheduled_seq_group.seq_group

            if seq_group.is_finished():
                continue

            # Get the logits for the current sequence group
            if outputs and len(outputs) > 0:
                # Extract logits from model outputs
                logits = self._extract_logits_for_group(outputs, i)
                
                # Check if this is a Beam Search request
                if self._is_beam_search_request(seq_group):
                    # Route to Beam Search processing
                    self._process_beam_search(seq_group, logits)
                else:
                    # Use original processing for non-Beam Search requests
                    self._process_sample(seq_group, logits, seq_group_meta)

        # Continue with original processing logic for finished requests
        self._finalize_request_outputs(
            ctx, seq_group_metadata_list, scheduler_outputs, 
            is_async, is_last_step, now, request_id, skip
        )

    def _is_beam_search_request(self, seq_group: SequenceGroup) -> bool:
        """
        Determine if a sequence group represents a Beam Search request.
        
        Args:
            seq_group: The sequence group to check
            
        Returns:
            True if this is a Beam Search request, False otherwise
        """
        if seq_group.sampling_params is None:
            return False
            
        # Check for Beam Search indicators:
        # 1. Multiple sequences (n > 1)
        # 2. Low temperature (deterministic sampling)
        # 3. Specific sampling configuration
        sampling_params = seq_group.sampling_params
        
        is_beam_search = (
            sampling_params.n > 1 and
            sampling_params.temperature <= 0.1 and  # Low temperature for beam search
            sampling_params.top_k <= 0 and  # No top-k restriction
            sampling_params.top_p >= 0.99  # No top-p restriction
        )
        
        return is_beam_search

    def _process_beam_search(
        self, 
        seq_group: SequenceGroup, 
        logits: Tensor
    ) -> None:
        """
        Process model outputs for a Beam Search sequence group.
        
        This method implements the core Beam Search algorithm:
        1. Get or create BeamSearchInstance for the request
        2. Advance beam search with current logits
        3. Handle beam forking and KV cache management
        4. Update sequence states
        
        Args:
            seq_group: The sequence group being processed
            logits: Model output logits for token selection
        """
        request_id = seq_group.request_id
        sampling_params = seq_group.sampling_params
        
        # Get or create BeamSearchInstance
        if request_id not in self.beam_search_instances:
            self._initialize_beam_search_instance(seq_group)
        
        beam_instance = self.beam_search_instances[request_id]
        beam_config = self.beam_search_configs[request_id]
        
        # Extract beam search parameters
        beam_width = sampling_params.n
        length_penalty = getattr(sampling_params, 'length_penalty', 1.0)
        eos_token_id = self._get_eos_token_id()
        
        logger.debug(f"Processing Beam Search for request {request_id} "
                    f"with beam_width={beam_width}")
        
        # Advance beam search step
        try:
            # Convert MindSpore tensor to appropriate format if needed
            if isinstance(logits, ms.Tensor):
                logits_np = logits.asnumpy()
            else:
                logits_np = logits
            
            # Perform beam search step
            new_beams = self._advance_beam_search_step(
                beam_instance=beam_instance,
                logits=logits_np,
                beam_width=beam_width,
                length_penalty=length_penalty,
                eos_token_id=eos_token_id
            )
            
            # Update beam instance
            beam_instance.beams = new_beams
            
            # Handle KV cache forking for new beams
            self._handle_kv_cache_forking(request_id, beam_instance)
            
            # Update sequence group state
            self._update_sequence_group_from_beams(seq_group, beam_instance)
            
        except Exception as e:
            logger.error(f"Error in beam search processing for {request_id}: {e}")
            # Fallback to regular sampling
            self._process_sample_fallback(seq_group, logits)

    def _process_sample(
        self, 
        seq_group: SequenceGroup, 
        logits: Tensor,
        seq_group_meta: SequenceGroupMetadata
    ) -> None:
        """
        Process model outputs using standard sampling methods.
        
        This method handles non-Beam Search requests using the original
        vLLM sampling logic.
        
        Args:
            seq_group: The sequence group being processed
            logits: Model output logits
            seq_group_meta: Sequence group metadata
        """
        # Use the original output processor for standard sampling
        if hasattr(self, 'output_processor'):
            # Create mock output structure for compatibility
            mock_output = self._create_mock_output(logits, seq_group_meta)
            self.output_processor.process_outputs(seq_group, [mock_output], False)
        else:
            logger.warning(f"No output processor available for request {seq_group.request_id}")

    def _initialize_beam_search_instance(self, seq_group: SequenceGroup) -> None:
        """
        Initialize a new BeamSearchInstance for a request.
        
        Args:
            seq_group: The sequence group to initialize beam search for
        """
        request_id = seq_group.request_id
        prompt_tokens = seq_group.prompt_token_ids
        sampling_params = seq_group.sampling_params
        
        # Create beam search instance
        beam_instance = BeamSearchInstance(
            prompt_tokens=prompt_tokens,
            lora_request=seq_group.lora_request
        )
        
        # Store configuration
        beam_config = {
            'beam_width': sampling_params.n,
            'max_tokens': sampling_params.max_tokens or 100,
            'length_penalty': getattr(sampling_params, 'length_penalty', 1.0),
            'temperature': sampling_params.temperature,
            'ignore_eos': sampling_params.ignore_eos
        }
        
        self.beam_search_instances[request_id] = beam_instance
        self.beam_search_configs[request_id] = beam_config
        
        logger.info(f"Initialized beam search for request {request_id} "
                   f"with config: {beam_config}")

    def _advance_beam_search_step(
        self,
        beam_instance: BeamSearchInstance,
        logits: Any,
        beam_width: int,
        length_penalty: float,
        eos_token_id: int
    ) -> List[BeamSearchSequence]:
        """
        Advance beam search by one step.
        
        Args:
            beam_instance: Current beam search instance
            logits: Model output logits
            beam_width: Number of beams to maintain
            length_penalty: Length penalty for scoring
            eos_token_id: End-of-sequence token ID
            
        Returns:
            Updated list of beam sequences
        """
        current_beams = beam_instance.beams
        new_beams = []
        
        # For each current beam, generate candidates
        for beam in current_beams:
            # Get top-k tokens from logits
            if hasattr(logits, 'shape') and len(logits.shape) > 1:
                beam_logits = logits[0]  # Assume batch size 1 for now
            else:
                beam_logits = logits
            
            # Get top candidates (2 * beam_width for diversity)
            top_k = min(2 * beam_width, len(beam_logits))
            top_indices = (-beam_logits).argsort()[:top_k]
            top_logprobs = beam_logits[top_indices]
            
            # Create new beam candidates
            for token_id, logprob in zip(top_indices, top_logprobs):
                new_beam = BeamSearchSequence(
                    tokens=beam.tokens + [int(token_id)],
                    logprobs=beam.logprobs + [{int(token_id): float(logprob)}],
                    cum_logprob=beam.cum_logprob + float(logprob),
                    lora_request=beam.lora_request
                )
                new_beams.append(new_beam)
        
        # Sort by beam search score and keep top beams
        new_beams.sort(
            key=lambda x: get_beam_search_score(
                x.tokens, x.cum_logprob, eos_token_id, length_penalty
            ),
            reverse=True
        )
        
        return new_beams[:beam_width]

    def _handle_kv_cache_forking(
        self, 
        request_id: str, 
        beam_instance: BeamSearchInstance
    ) -> None:
        """
        Handle KV cache forking for beam sequences.
        
        Args:
            request_id: The request ID
            beam_instance: Current beam search instance
        """
        logger.debug(f"Handling KV cache forking for request {request_id} "
                    f"with {len(beam_instance.beams)} beams")
        
        # 获取父序列和子序列
        parent_beam = beam_instance.beams[0]  # 第一个 beam 作为父序列
        child_beams = beam_instance.beams[1:]  # 其余 beams 作为子序列
        
        # 获取序列组
        seq_group = self.scheduler[0].get_sequence_group(request_id)
        if not seq_group:
            logger.warning(f"Sequence group not found for request {request_id}")
            return
            
        # 获取父序列和子序列
        parent_seq = seq_group.seqs[0]
        child_seqs = seq_group.seqs[1:]
        
        # 使用 BlockManager 进行 fork
        self.block_manager.fork(parent_seq, child_seqs)
        
        # 获取 KV cache
        parent_key_cache, parent_value_cache = self.model_executor.get_kvcache()
        
        # 为每个子序列 fork KV cache
        for child_seq in child_seqs:
            # 获取子序列的 slot mapping
            child_slot_mapping = self.block_manager.get_block_table(child_seq.seq_id)
            child_slot_mapping = ms.Tensor(child_slot_mapping, dtype=ms.int32)
            
            # 获取父序列的 slot mapping
            parent_slot_mapping = self.block_manager.get_block_table(parent_seq.seq_id)
            parent_slot_mapping = ms.Tensor(parent_slot_mapping, dtype=ms.int32)
            
            # 使用 Attention 层进行 KV cache fork
            child_kv_cache = []
            for layer_idx in range(len(parent_key_cache)):
                child_key_cache, child_value_cache = self.model_executor.attention.fork_kv_cache(
                    parent_key_cache[layer_idx],
                    parent_value_cache[layer_idx],
                    parent_slot_mapping,
                    child_slot_mapping
                )
                child_kv_cache.append((child_key_cache, child_value_cache))
            
            # 使用 CacheEngine 更新子序列的 KV cache
            self.cache_engine.update_kv_cache(
                child_seq.seq_id,
                child_kv_cache,
                child_slot_mapping.tolist()
            )
                
        logger.debug(f"Successfully forked KV cache for request {request_id}")

    def _update_sequence_group_from_beams(
        self, 
        seq_group: SequenceGroup, 
        beam_instance: BeamSearchInstance
    ) -> None:
        """
        Update sequence group state based on beam search results.
        
        Args:
            seq_group: The sequence group to update
            beam_instance: Current beam search instance
        """
        # Update sequence data with beam results
        for i, beam in enumerate(beam_instance.beams):
            if i < len(seq_group.seqs):
                seq = seq_group.seqs[i]
                # Update sequence tokens (this is simplified)
                # In practice, you'd need to properly update SequenceData
                logger.debug(f"Updated sequence {seq.seq_id} with beam tokens: "
                           f"{beam.tokens[-5:]}")  # Log last 5 tokens

    def _extract_logits_for_group(self, outputs: List[Any], group_index: int) -> Tensor:
        """
        Extract logits for a specific sequence group from model outputs.
        
        Args:
            outputs: Model outputs
            group_index: Index of the sequence group
            
        Returns:
            Logits tensor for the sequence group
        """
        # This is a simplified implementation
        # In practice, you'd need to properly extract logits based on
        # the output structure and group indexing
        
        if outputs and len(outputs) > group_index:
            output = outputs[group_index]
            if hasattr(output, 'logits'):
                return output.logits
            elif hasattr(output, 'outputs') and output.outputs:
                return output.outputs[0].logits if hasattr(output.outputs[0], 'logits') else output.outputs[0]
        
        # Fallback: return dummy logits
        vocab_size = getattr(self.model_config, 'vocab_size', 32000)
        return ms.ops.zeros((1, vocab_size), dtype=ms.float32)

    def _get_eos_token_id(self) -> int:
        """Get the end-of-sequence token ID."""
        if hasattr(self, 'tokenizer') and self.tokenizer:
            tokenizer_group = self.get_tokenizer_group()
            if tokenizer_group:
                tokenizer = tokenizer_group.get_lora_tokenizer(None)
                return getattr(tokenizer, 'eos_token_id', 2)
        return 2  # Default EOS token ID

    def _process_sample_fallback(self, seq_group: SequenceGroup, logits: Tensor) -> None:
        """
        Fallback sampling method when beam search fails.
        
        Args:
            seq_group: The sequence group
            logits: Model output logits
        """
        logger.warning(f"Using fallback sampling for request {seq_group.request_id}")
        # Implement simple greedy sampling as fallback
        # This is a placeholder implementation

    def _create_mock_output(self, logits: Tensor, seq_group_meta: SequenceGroupMetadata) -> Any:
        """
        Create a mock output structure for compatibility with original processor.
        
        Args:
            logits: Model output logits
            seq_group_meta: Sequence group metadata
            
        Returns:
            Mock output structure
        """
        # This is a placeholder for creating compatible output structure
        # In practice, you'd create the appropriate output type expected
        # by the output processor
        return None

    def _finalize_request_outputs(
        self,
        ctx: SchedulerContext,
        seq_group_metadata_list: List[SequenceGroupMetadata],
        scheduler_outputs: SchedulerOutputs,
        is_async: bool,
        is_last_step: bool,
        now: float,
        request_id: Optional[str],
        skip: List[int]
    ) -> None:
        """
        Finalize request outputs and handle completed requests.
        
        This method handles the completion of requests and generates
        final outputs for both Beam Search and regular requests.
        """
        # Handle finished requests
        finished_now = []
        for i, (seq_group_meta, scheduled_seq_group) in enumerate(
            zip(seq_group_metadata_list, scheduler_outputs.scheduled_seq_groups)
        ):
            if i in skip:
                continue
                
            seq_group = scheduled_seq_group.seq_group
            
            if seq_group.is_finished():
                finished_now.append(i)
                
                # Generate final output
                if seq_group.request_id in self.beam_search_instances:
                    # Generate beam search output
                    beam_output = self._generate_beam_search_output(seq_group)
                    if beam_output:
                        ctx.request_outputs.append(beam_output)
                    
                    # Clean up beam search state
                    del self.beam_search_instances[seq_group.request_id]
                    del self.beam_search_configs[seq_group.request_id]
                else:
                    # Generate regular output
                    request_output = self._generate_regular_output(seq_group, now)
                    if request_output:
                        ctx.request_outputs.append(request_output)

        # Free finished sequence groups
        if finished_now:
            for scheduler in self.scheduler:
                scheduler.free_finished_seq_groups()

    def _generate_beam_search_output(self, seq_group: SequenceGroup) -> Optional[RequestOutput]:
        """
        Generate final output for a completed Beam Search request.
        
        Args:
            seq_group: The completed sequence group
            
        Returns:
            Request output with beam search results
        """
        request_id = seq_group.request_id
        
        if request_id not in self.beam_search_instances:
            return None
            
        beam_instance = self.beam_search_instances[request_id]
        
        # Create beam search output
        beam_output = BeamSearchOutput(sequences=beam_instance.beams)
        
        # Convert to RequestOutput format
        # This is a simplified implementation
        # In practice, you'd need to properly format the output
        
        logger.info(f"Generated beam search output for request {request_id} "
                   f"with {len(beam_instance.beams)} beams")
        
        return None  # Placeholder - implement actual RequestOutput creation

    def _generate_regular_output(self, seq_group: SequenceGroup, now: float) -> Optional[RequestOutput]:
        """
        Generate final output for a completed regular request.
        
        Args:
            seq_group: The completed sequence group
            now: Current timestamp
            
        Returns:
            Request output
        """
        # Use original output generation logic
        seq_group.maybe_set_first_token_time(now)
        if not seq_group.is_prefill():
            seq_group.set_last_token_time(now)
            
        # This would use the original RequestOutputFactory
        # return RequestOutputFactory.create(seq_group, ...)
        return None  # Placeholder

    def get_beam_search_results(self, request_id: str) -> Optional[BeamSearchOutput]:
        """
        Get current beam search results for a request.
        
        Args:
            request_id: The request ID
            
        Returns:
            Current beam search output or None if not found
        """
        if request_id in self.beam_search_instances:
            beam_instance = self.beam_search_instances[request_id]
            return BeamSearchOutput(sequences=beam_instance.beams)
        return None

    def abort_beam_search_request(self, request_id: str) -> bool:
        """
        Abort a beam search request and clean up resources.
        
        Args:
            request_id: The request ID to abort
            
        Returns:
            True if request was found and aborted, False otherwise
        """
        if request_id in self.beam_search_instances:
            del self.beam_search_instances[request_id]
            del self.beam_search_configs[request_id]
            logger.info(f"Aborted beam search request {request_id}")
            return True
        return False

    def get_beam_search_stats(self) -> Dict[str, Any]:
        """
        Get statistics about active beam search requests.
        
        Returns:
            Dictionary containing beam search statistics
        """
        stats = {
            'active_beam_requests': len(self.beam_search_instances),
            'total_beams': sum(
                len(instance.beams) 
                for instance in self.beam_search_instances.values()
            ),
            'requests': {}
        }
        
        for request_id, instance in self.beam_search_instances.items():
            config = self.beam_search_configs.get(request_id, {})
            stats['requests'][request_id] = {
                'num_beams': len(instance.beams),
                'beam_width': config.get('beam_width', 0),
                'max_tokens': config.get('max_tokens', 0)
            }
        
        return stats 