# NPU Beam Search Integration Guide

This document provides detailed integration instructions for incorporating the NPU-optimized beam search sampling strategy into the existing vllm-mindspore codebase.

## Overview

The NPU beam search implementation requires modifications to three key components:
1. `cache_engine.py` - Enhanced cache management for beam search
2. `model_runner.py` - Integration with beam search sampler
3. `worker.py` - Workflow coordination for beam search execution

## 1. Cache Engine Modifications

### File: `vllm_mindspore/worker/cache_engine.py`

Add the following enhancements to support beam search cache management:

```python
# Add to imports
from typing import Dict, List, Set
from collections import defaultdict

class CacheEngine:
    def __init__(self, ...):
        # Existing initialization code
        
        # Add beam search specific attributes
        self.beam_cache_enabled = False
        self.beam_block_mapping: Dict[int, List[int]] = {}  # beam_id -> block_ids
        self.block_ref_count: Dict[int, int] = defaultdict(int)
        self.available_blocks: Set[int] = set()
        
    def enable_beam_search_cache(self) -> None:
        """Enable beam search specific cache management."""
        self.beam_cache_enabled = True
        
    def allocate_beam_cache_blocks(self, beam_id: int, num_blocks: int) -> List[int]:
        """Allocate cache blocks for a specific beam."""
        if not self.beam_cache_enabled:
            return self.allocate(num_blocks)  # Fallback to regular allocation
            
        block_ids = []
        
        # Try to reuse available blocks
        while len(block_ids) < num_blocks and self.available_blocks:
            block_id = self.available_blocks.pop()
            block_ids.append(block_id)
            self.block_ref_count[block_id] = 1
            
        # Allocate new blocks if needed
        remaining_blocks = num_blocks - len(block_ids)
        if remaining_blocks > 0:
            new_blocks = self.allocate(remaining_blocks)
            block_ids.extend(new_blocks)
            for block_id in new_blocks:
                self.block_ref_count[block_id] = 1
                
        self.beam_block_mapping[beam_id] = block_ids
        return block_ids
        
    def share_beam_cache_blocks(self, parent_beam_id: int, child_beam_id: int) -> None:
        """Share cache blocks between parent and child beams."""
        if not self.beam_cache_enabled or parent_beam_id not in self.beam_block_mapping:
            return
            
        parent_blocks = self.beam_block_mapping[parent_beam_id]
        self.beam_block_mapping[child_beam_id] = parent_blocks.copy()
        
        # Increment reference count
        for block_id in parent_blocks:
            self.block_ref_count[block_id] += 1
            
    def copy_beam_cache_block(self, beam_id: int, block_index: int) -> int:
        """Implement copy-on-write for beam cache blocks."""
        if (not self.beam_cache_enabled or 
            beam_id not in self.beam_block_mapping or 
            block_index >= len(self.beam_block_mapping[beam_id])):
            return -1
            
        current_blocks = self.beam_block_mapping[beam_id]
        old_block_id = current_blocks[block_index]
        
        # If block is shared, create a copy
        if self.block_ref_count[old_block_id] > 1:
            new_blocks = self.allocate(1)
            new_block_id = new_blocks[0]
            
            # Copy cache content
            self._copy_cache_content(old_block_id, new_block_id)
            
            # Update mappings
            current_blocks[block_index] = new_block_id
            self.block_ref_count[old_block_id] -= 1
            self.block_ref_count[new_block_id] = 1
            
            return new_block_id
            
        return old_block_id
        
    def release_beam_cache(self, beam_id: int) -> None:
        """Release cache blocks for a finished beam."""
        if not self.beam_cache_enabled or beam_id not in self.beam_block_mapping:
            return
            
        block_ids = self.beam_block_mapping[beam_id]
        for block_id in block_ids:
            self.block_ref_count[block_id] -= 1
            if self.block_ref_count[block_id] == 0:
                self.free([block_id])
                if block_id in self.block_ref_count:
                    del self.block_ref_count[block_id]
            else:
                self.available_blocks.add(block_id)
                
        del self.beam_block_mapping[beam_id]
        
    def _copy_cache_content(self, src_block_id: int, dst_block_id: int) -> None:
        """Copy KV cache content between blocks."""
        # Implementation would depend on the specific cache structure
        # This is a placeholder for the actual cache copying logic
        pass
```

## 2. Model Runner Modifications

### File: `vllm_mindspore/worker/model_runner.py`

Integrate beam search sampler with the model execution pipeline:

```python
# Add to imports
from vllm_mindspore.beam_search.npu_beam_search import (
    NPUBeamSearchSampler, 
    create_npu_beam_search_sampler,
    extend_attention_metadata_for_beams
)
from vllm.sampling_params import SamplingParams
from typing import Optional, Dict, Any

class ModelRunner:
    def __init__(self, ...):
        # Existing initialization code
        
        # Add beam search support
        self.beam_samplers: Dict[str, NPUBeamSearchSampler] = {}
        self.beam_search_enabled = False
        
    def enable_beam_search(self) -> None:
        """Enable beam search functionality."""
        self.beam_search_enabled = True
        if hasattr(self, 'cache_engine') and self.cache_engine:
            for cache_engine in self.cache_engine:
                cache_engine.enable_beam_search_cache()
                
    def prepare_input_tensors(self, seq_group_metadata_list) -> Dict[str, Any]:
        """Prepare input tensors with beam search support."""
        # Existing input preparation code
        input_tensors = self._prepare_base_input_tensors(seq_group_metadata_list)
        
        # Handle beam search specific preparation
        if self.beam_search_enabled:
            input_tensors = self._prepare_beam_search_inputs(
                input_tensors, seq_group_metadata_list
            )
            
        return input_tensors
        
    def _prepare_beam_search_inputs(self, 
                                   input_tensors: Dict[str, Any],
                                   seq_group_metadata_list) -> Dict[str, Any]:
        """Prepare beam search specific input tensors."""
        for seq_group_metadata in seq_group_metadata_list:
            request_id = seq_group_metadata.request_id
            
            if request_id in self.beam_samplers:
                beam_sampler = self.beam_samplers[request_id]
                
                # Update attention metadata for beam search
                if 'attn_metadata' in input_tensors:
                    input_tensors['attn_metadata'] = extend_attention_metadata_for_beams(
                        input_tensors['attn_metadata'], beam_sampler
                    )
                    
                # Prepare beam-specific input tokens
                if beam_sampler.active_beams:
                    beam_tokens = []
                    for beam in beam_sampler.active_beams:
                        beam_tokens.extend(beam.tokens)
                    
                    # Update input tokens tensor
                    if 'input_tokens' in input_tensors:
                        # Modify input tokens to include all active beams
                        pass  # Implementation depends on tensor structure
                        
        return input_tensors
        
    def execute_model(self, 
                     seq_group_metadata_list,
                     kv_caches) -> List[SamplerOutput]:
        """Execute model with beam search support."""
        # Check if any sequence groups require beam search
        beam_search_requests = self._identify_beam_search_requests(seq_group_metadata_list)
        
        if beam_search_requests:
            return self._execute_beam_search_model(
                seq_group_metadata_list, kv_caches, beam_search_requests
            )
        else:
            # Use existing execution path
            return self._execute_regular_model(seq_group_metadata_list, kv_caches)
            
    def _identify_beam_search_requests(self, seq_group_metadata_list) -> List[str]:
        """Identify which requests require beam search."""
        beam_requests = []
        for seq_group_metadata in seq_group_metadata_list:
            sampling_params = seq_group_metadata.sampling_params
            if (hasattr(sampling_params, 'use_beam_search') and 
                sampling_params.use_beam_search and
                getattr(sampling_params, 'best_of', 1) > 1):
                beam_requests.append(seq_group_metadata.request_id)
        return beam_requests
        
    def _execute_beam_search_model(self, 
                                  seq_group_metadata_list,
                                  kv_caches,
                                  beam_search_requests) -> List[SamplerOutput]:
        """Execute model with beam search logic."""
        # Initialize beam samplers for new requests
        for seq_group_metadata in seq_group_metadata_list:
            request_id = seq_group_metadata.request_id
            if (request_id in beam_search_requests and 
                request_id not in self.beam_samplers):
                
                beam_sampler = create_npu_beam_search_sampler(
                    seq_group_metadata.sampling_params,
                    self.cache_engine[0] if self.cache_engine else None,
                    self.model_config
                )
                
                if beam_sampler:
                    self.beam_samplers[request_id] = beam_sampler
                    # Initialize beams with prompt tokens
                    prompt_tokens = seq_group_metadata.seq_data[0].get_token_ids()
                    beam_sampler.initialize_beams(prompt_tokens)
                    
        # Prepare input tensors
        input_tensors = self.prepare_input_tensors(seq_group_metadata_list)
        
        # Execute forward pass
        hidden_states = self.model(**input_tensors)
        
        # Process beam search sampling
        sampler_outputs = []
        for seq_group_metadata in seq_group_metadata_list:
            request_id = seq_group_metadata.request_id
            
            if request_id in self.beam_samplers:
                beam_sampler = self.beam_samplers[request_id]
                
                # Extract logits for this request
                logits = self._extract_request_logits(hidden_states, request_id)
                
                # Perform beam search step
                continue_search = beam_sampler.step(logits)
                
                if not continue_search:
                    # Finalize beam search
                    beam_output = beam_sampler.finalize()
                    sampler_output = self._convert_beam_output_to_sampler_output(
                        beam_output, seq_group_metadata
                    )
                    sampler_outputs.append(sampler_output)
                    
                    # Clean up
                    del self.beam_samplers[request_id]
                else:
                    # Continue beam search
                    sampler_output = self._create_intermediate_sampler_output(
                        beam_sampler, seq_group_metadata
                    )
                    sampler_outputs.append(sampler_output)
            else:
                # Regular sampling for non-beam search requests
                sampler_output = self._regular_sampling(hidden_states, seq_group_metadata)
                sampler_outputs.append(sampler_output)
                
        return sampler_outputs
        
    def _extract_request_logits(self, hidden_states, request_id):
        """Extract logits for a specific request from model output."""
        # Implementation depends on how hidden_states are structured
        # This is a placeholder for the actual logits extraction
        return hidden_states  # Simplified
        
    def _convert_beam_output_to_sampler_output(self, beam_output, seq_group_metadata):
        """Convert beam search output to SamplerOutput format."""
        # Implementation to convert BeamSearchOutput to SamplerOutput
        # This would create appropriate SequenceOutput objects
        pass
        
    def _create_intermediate_sampler_output(self, beam_sampler, seq_group_metadata):
        """Create intermediate sampler output for ongoing beam search."""
        # Implementation for intermediate beam search results
        pass
```

## 3. Worker Modifications

### File: `vllm_mindspore/worker/worker.py`

Coordinate beam search execution at the worker level:

```python
# Add to imports
from vllm.sampling_params import SamplingParams
from typing import List, Optional

class Worker:
    def __init__(self, ...):
        # Existing initialization code
        
        # Add beam search coordination
        self.beam_search_requests: Set[str] = set()
        
    def execute_model(self, 
                     seq_group_metadata_list: List[SequenceGroupMetadata],
                     blocks_to_swap_in: Dict[int, int],
                     blocks_to_swap_out: Dict[int, int],
                     blocks_to_copy: Dict[int, List[int]]) -> List[SamplerOutput]:
        """Execute model with beam search coordination."""
        
        # Check for beam search requests
        self._update_beam_search_requests(seq_group_metadata_list)
        
        # Enable beam search if needed
        if self.beam_search_requests and not self.model_runner.beam_search_enabled:
            self.model_runner.enable_beam_search()
            
        # Handle cache operations for beam search
        if self.beam_search_requests:
            self._handle_beam_search_cache_operations(
                seq_group_metadata_list, blocks_to_copy
            )
            
        # Execute model
        output = self.model_runner.execute_model(
            seq_group_metadata_list=seq_group_metadata_list,
            kv_caches=self.cache_engine,
        )
        
        # Post-process beam search results
        if self.beam_search_requests:
            output = self._post_process_beam_search_output(
                output, seq_group_metadata_list
            )
            
        return output
        
    def _update_beam_search_requests(self, seq_group_metadata_list):
        """Update tracking of beam search requests."""
        current_requests = set()
        
        for seq_group_metadata in seq_group_metadata_list:
            sampling_params = seq_group_metadata.sampling_params
            if (hasattr(sampling_params, 'use_beam_search') and 
                sampling_params.use_beam_search and
                getattr(sampling_params, 'best_of', 1) > 1):
                current_requests.add(seq_group_metadata.request_id)
                
        # Update beam search requests set
        self.beam_search_requests = current_requests
        
    def _handle_beam_search_cache_operations(self, 
                                            seq_group_metadata_list,
                                            blocks_to_copy):
        """Handle cache operations specific to beam search."""
        for seq_group_metadata in seq_group_metadata_list:
            request_id = seq_group_metadata.request_id
            
            if request_id in self.beam_search_requests:
                # Handle beam-specific cache copying
                if request_id in self.model_runner.beam_samplers:
                    beam_sampler = self.model_runner.beam_samplers[request_id]
                    
                    # Implement beam-specific cache operations
                    for beam in beam_sampler.active_beams:
                        if beam.parent_beam_id is not None:
                            # Handle cache sharing/copying for beam branching
                            self._handle_beam_cache_branching(beam)
                            
    def _handle_beam_cache_branching(self, beam):
        """Handle cache operations when beams branch."""
        # Implementation for beam-specific cache branching
        # This would coordinate with the cache engine to handle
        # copy-on-write operations for beam search
        pass
        
    def _post_process_beam_search_output(self, 
                                       output,
                                       seq_group_metadata_list):
        """Post-process beam search output."""
        # Handle any post-processing needed for beam search results
        # This might include result aggregation or formatting
        return output
```

## 4. Usage Example

Here's how to use the NPU beam search implementation:

```python
from vllm import LLM, SamplingParams
from vllm_mindspore.beam_search.npu_beam_search import NPUBeamSearchSampler

# Initialize LLM with beam search support
llm = LLM(model="your-model-path", 
          tensor_parallel_size=1,
          device="npu")

# Configure beam search parameters
sampling_params = SamplingParams(
    use_beam_search=True,
    best_of=4,  # Beam width
    length_penalty=1.2,
    max_tokens=100,
    temperature=0.0  # Deterministic for beam search
)

# Generate with beam search
prompts = ["The future of artificial intelligence is"]
outputs = llm.generate(prompts, sampling_params)

# Process beam search results
for output in outputs:
    prompt = output.prompt
    for i, completion in enumerate(output.outputs):
        print(f"Beam {i+1}: {completion.text}")
        print(f"Score: {completion.cumulative_logprob}")
```

## 5. Performance Considerations

### NPU Optimization Strategies

1. **Memory Management**:
   - Implement efficient KV cache sharing between beams
   - Use copy-on-write to minimize memory overhead
   - Batch beam operations for better NPU utilization

2. **Computation Optimization**:
   - Vectorize beam scoring operations using MindSpore ops
   - Minimize host-device transfers
   - Use NPU-native operations for top-k selection

3. **Cache Efficiency**:
   - Implement smart cache block allocation
   - Reuse cache blocks across beam generations
   - Optimize cache access patterns for NPU memory hierarchy

## 6. Testing and Validation

To validate the beam search implementation:

1. **Unit Tests**: Test individual components (cache tracker, score calculator)
2. **Integration Tests**: Test with existing vLLM workflows
3. **Performance Tests**: Compare with CPU/GPU beam search implementations
4. **Accuracy Tests**: Verify beam search results match expected outputs

## 7. Future Enhancements

1. **Dynamic Beam Width**: Adaptive beam width based on generation quality
2. **Diverse Beam Search**: Implement diverse beam search variants
3. **Constrained Generation**: Support for guided/constrained beam search
4. **Multi-GPU Beam Search**: Scale beam search across multiple NPUs

This integration guide provides a comprehensive framework for implementing NPU-optimized beam search in the vllm-mindspore plugin while maintaining compatibility with the existing architecture.