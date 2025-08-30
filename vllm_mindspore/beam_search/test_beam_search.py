#!/usr/bin/env python3
"""Comprehensive test suite for NPU beam search implementation."""

import pytest
import numpy as np
import mindspore as ms
from unittest.mock import Mock, MagicMock, patch
from typing import List, Optional

# Import the beam search components
from npu_beam_search import (
    BeamState,
    KVCacheBeamTracker,
    NPUBeamScoreCalculator,
    NPUBeamSearchSampler,
    create_npu_beam_search_sampler,
    integrate_beam_search_with_model_runner,
    extend_attention_metadata_for_beams
)

class TestBeamState:
    """Test cases for BeamState class."""
    
    def test_beam_state_creation(self):
        """Test basic beam state creation."""
        beam = BeamState(
            beam_id=0,
            parent_beam_id=None,
            tokens=[1, 2, 3],
            logprobs=[{1: -0.1}, {2: -0.2}, {3: -0.3}],
            cumulative_logprob=-0.6,
            cache_block_ids=[0, 1]
        )
        
        assert beam.beam_id == 0
        assert beam.parent_beam_id is None
        assert beam.tokens == [1, 2, 3]
        assert len(beam.logprobs) == 3
        assert beam.cumulative_logprob == -0.6
        assert beam.cache_block_ids == [0, 1]
        assert not beam.is_finished
        assert beam.finish_reason is None
    
    def test_beam_state_finished(self):
        """Test finished beam state."""
        beam = BeamState(
            beam_id=1,
            parent_beam_id=0,
            tokens=[1, 2, 3, 2],  # EOS token
            logprobs=[{1: -0.1}, {2: -0.2}, {3: -0.3}, {2: -0.1}],
            cumulative_logprob=-0.7,
            cache_block_ids=[0, 1],
            is_finished=True,
            finish_reason="stop"
        )
        
        assert beam.is_finished
        assert beam.finish_reason == "stop"

class TestKVCacheBeamTracker:
    """Test cases for KVCacheBeamTracker class."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.mock_cache_engine = Mock()
        self.mock_cache_engine.get_num_free_gpu_blocks.return_value = 100
        self.mock_cache_engine.allocate_gpu_block.return_value = [0, 1, 2]
        self.block_size = 16
        
    def test_cache_tracker_creation(self):
        """Test cache tracker creation."""
        tracker = KVCacheBeamTracker(self.mock_cache_engine, self.block_size)
        
        assert tracker.cache_engine == self.mock_cache_engine
        assert tracker.block_size == self.block_size
        assert len(tracker.beam_cache_map) == 0
        assert len(tracker.block_ref_count) == 0
    
    def test_cache_tracker_invalid_inputs(self):
        """Test cache tracker with invalid inputs."""
        with pytest.raises(ValueError):
            KVCacheBeamTracker(None, self.block_size)
            
        with pytest.raises(ValueError):
            KVCacheBeamTracker(self.mock_cache_engine, 0)
    
    def test_allocate_cache_blocks(self):
        """Test cache block allocation."""
        tracker = KVCacheBeamTracker(self.mock_cache_engine, self.block_size)
        
        # Test successful allocation
        block_ids = tracker.allocate_cache_blocks(beam_id=0, num_blocks=3)
        
        assert len(block_ids) == 3
        assert 0 in tracker.beam_cache_map
        assert tracker.beam_cache_map[0] == block_ids
        
        # Verify reference counts
        for block_id in block_ids:
            assert tracker.block_ref_count[block_id] == 1
    
    def test_share_cache_blocks(self):
        """Test cache block sharing between beams."""
        tracker = KVCacheBeamTracker(self.mock_cache_engine, self.block_size)
        
        # Allocate blocks for parent beam
        parent_blocks = tracker.allocate_cache_blocks(beam_id=0, num_blocks=2)
        
        # Share blocks with child beam
        tracker.share_cache_blocks(parent_beam_id=0, child_beam_id=1)
        
        assert 1 in tracker.beam_cache_map
        assert tracker.beam_cache_map[1] == parent_blocks
        
        # Verify reference counts increased
        for block_id in parent_blocks:
            assert tracker.block_ref_count[block_id] == 2
    
    def test_release_beam_cache(self):
        """Test cache release for beam."""
        tracker = KVCacheBeamTracker(self.mock_cache_engine, self.block_size)
        
        # Allocate and share blocks
        parent_blocks = tracker.allocate_cache_blocks(beam_id=0, num_blocks=2)
        tracker.share_cache_blocks(parent_beam_id=0, child_beam_id=1)
        
        # Release parent beam cache
        tracker.release_beam_cache(beam_id=0)
        
        assert 0 not in tracker.beam_cache_map
        # Blocks should still exist for child beam
        for block_id in parent_blocks:
            assert tracker.block_ref_count[block_id] == 1
        
        # Release child beam cache
        tracker.release_beam_cache(beam_id=1)
        
        assert 1 not in tracker.beam_cache_map
        # All blocks should be released
        for block_id in parent_blocks:
            assert block_id not in tracker.block_ref_count

class TestNPUBeamScoreCalculator:
    """Test cases for NPUBeamScoreCalculator class."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.vocab_size = 1000
        self.device = "cpu"  # Use CPU for testing
        
    def test_score_calculator_creation(self):
        """Test score calculator creation."""
        calculator = NPUBeamScoreCalculator(self.vocab_size, self.device)
        
        assert calculator.vocab_size == self.vocab_size
        assert calculator.device == "cpu"  # Should fallback to CPU
    
    def test_score_calculator_invalid_inputs(self):
        """Test score calculator with invalid inputs."""
        with pytest.raises(ValueError):
            NPUBeamScoreCalculator(0, self.device)
            
        with pytest.raises(ValueError):
            NPUBeamScoreCalculator(self.vocab_size, "")
    
    def test_compute_beam_scores(self):
        """Test beam score computation."""
        calculator = NPUBeamScoreCalculator(self.vocab_size, self.device)
        
        # Create test logits
        batch_size = 2
        logits = ms.Tensor(np.random.randn(batch_size, self.vocab_size), ms.float32)
        
        # Create test beam states
        beam_states = [
            BeamState(0, None, [1, 2], [], -0.5, []),
            BeamState(1, None, [1, 3], [], -0.7, [])
        ]
        
        scores = calculator.compute_beam_scores(
            logits, beam_states, length_penalty=1.0, eos_token_id=2
        )
        
        assert scores.shape == (batch_size, self.vocab_size)
        assert isinstance(scores, ms.Tensor)
    
    def test_select_top_beams(self):
        """Test top beam selection."""
        calculator = NPUBeamScoreCalculator(self.vocab_size, self.device)
        
        # Create test scores
        batch_size = 2
        scores = ms.Tensor(np.random.randn(batch_size, self.vocab_size), ms.float32)
        beam_width = 3
        
        top_scores, top_indices = calculator.select_top_beams(scores, beam_width)
        
        assert top_scores.shape == (beam_width,)
        assert top_indices.shape == (beam_width, 2)  # (beam_idx, token_idx)
        assert isinstance(top_scores, ms.Tensor)
        assert isinstance(top_indices, ms.Tensor)

class TestNPUBeamSearchSampler:
    """Test cases for NPUBeamSearchSampler class."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.beam_width = 3
        self.mock_cache_engine = Mock()
        self.mock_cache_engine.get_num_free_gpu_blocks.return_value = 100
        self.mock_cache_engine.allocate_gpu_block.return_value = [0, 1, 2]
        
        self.mock_model_config = Mock()
        self.mock_model_config.vocab_size = 1000
        
    def test_sampler_creation(self):
        """Test beam search sampler creation."""
        sampler = NPUBeamSearchSampler(
            beam_width=self.beam_width,
            cache_engine=self.mock_cache_engine,
            model_config=self.mock_model_config
        )
        
        assert sampler.beam_width == self.beam_width
        assert sampler.cache_engine == self.mock_cache_engine
        assert sampler.model_config == self.mock_model_config
        assert len(sampler.active_beams) == 0
        assert len(sampler.completed_beams) == 0
    
    def test_sampler_invalid_inputs(self):
        """Test sampler with invalid inputs."""
        with pytest.raises(ValueError):
            NPUBeamSearchSampler(0, self.mock_cache_engine, self.mock_model_config)
            
        with pytest.raises(ValueError):
            NPUBeamSearchSampler(self.beam_width, self.mock_cache_engine, 
                               self.mock_model_config, length_penalty=0)
    
    def test_initialize_beams(self):
        """Test beam initialization."""
        sampler = NPUBeamSearchSampler(
            beam_width=self.beam_width,
            cache_engine=self.mock_cache_engine,
            model_config=self.mock_model_config
        )
        
        prompt_tokens = [1, 2, 3, 4]
        sampler.initialize_beams(prompt_tokens)
        
        assert len(sampler.active_beams) == 1
        assert sampler.active_beams[0].tokens == prompt_tokens
        assert sampler.active_beams[0].beam_id == 0
        assert len(sampler.completed_beams) == 0
    
    def test_initialize_beams_invalid_input(self):
        """Test beam initialization with invalid input."""
        sampler = NPUBeamSearchSampler(
            beam_width=self.beam_width,
            cache_engine=self.mock_cache_engine,
            model_config=self.mock_model_config
        )
        
        with pytest.raises(ValueError):
            sampler.initialize_beams([])
            
        with pytest.raises(ValueError):
            sampler.initialize_beams([1, -1, 3])  # Negative token

class TestFactoryFunction:
    """Test cases for factory function."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.mock_sampling_params = Mock()
        self.mock_sampling_params.use_beam_search = True
        self.mock_sampling_params.best_of = 3
        self.mock_sampling_params.length_penalty = 1.0
        self.mock_sampling_params.max_tokens = 512
        self.mock_sampling_params.stop_token_ids = [2]
        
        self.mock_cache_engine = Mock()
        self.mock_model_config = Mock()
        self.mock_model_config.vocab_size = 1000
    
    def test_create_beam_search_sampler(self):
        """Test successful beam search sampler creation."""
        sampler = create_npu_beam_search_sampler(
            self.mock_sampling_params,
            self.mock_cache_engine,
            self.mock_model_config
        )
        
        assert sampler is not None
        assert isinstance(sampler, NPUBeamSearchSampler)
        assert sampler.beam_width == 3
    
    def test_create_beam_search_sampler_no_beam_search(self):
        """Test when beam search is not requested."""
        self.mock_sampling_params.use_beam_search = False
        
        sampler = create_npu_beam_search_sampler(
            self.mock_sampling_params,
            self.mock_cache_engine,
            self.mock_model_config
        )
        
        assert sampler is None
    
    def test_create_beam_search_sampler_invalid_params(self):
        """Test with invalid parameters."""
        sampler = create_npu_beam_search_sampler(
            None,
            self.mock_cache_engine,
            self.mock_model_config
        )
        
        assert sampler is None

class TestIntegrationFunctions:
    """Test cases for integration functions."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.mock_model_runner = Mock()
        self.mock_cache_engine = Mock()
        self.mock_model_runner.cache_engine = self.mock_cache_engine
        self.mock_model_runner.model_config = Mock()
        self.mock_model_runner.model_config.vocab_size = 1000
        
        self.mock_sampling_params = Mock()
        self.mock_sampling_params.use_beam_search = True
        self.mock_sampling_params.best_of = 3
    
    def test_integrate_beam_search_with_model_runner(self):
        """Test beam search integration with model runner."""
        sampler = integrate_beam_search_with_model_runner(
            self.mock_model_runner,
            self.mock_sampling_params
        )
        
        # Should return a sampler or None based on parameters
        assert sampler is None or isinstance(sampler, NPUBeamSearchSampler)
    
    def test_integrate_beam_search_invalid_inputs(self):
        """Test integration with invalid inputs."""
        with pytest.raises(ValueError):
            integrate_beam_search_with_model_runner(None, self.mock_sampling_params)
            
        with pytest.raises(ValueError):
            integrate_beam_search_with_model_runner(self.mock_model_runner, None)

class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_empty_beam_step(self):
        """Test step with no active beams."""
        sampler = NPUBeamSearchSampler(
            beam_width=3,
            cache_engine=Mock(),
            model_config=Mock()
        )
        
        # Create dummy logits
        logits = ms.Tensor(np.random.randn(1, 1000), ms.float32)
        
        # Should return False when no active beams
        result = sampler.step(logits)
        assert result is False
    
    def test_mismatched_logits_beams(self):
        """Test step with mismatched logits and beam count."""
        mock_cache_engine = Mock()
        mock_cache_engine.get_num_free_gpu_blocks.return_value = 100
        mock_cache_engine.allocate_gpu_block.return_value = [0, 1]
        
        sampler = NPUBeamSearchSampler(
            beam_width=3,
            cache_engine=mock_cache_engine,
            model_config=Mock()
        )
        
        # Initialize with one beam
        sampler.initialize_beams([1, 2, 3])
        
        # Create logits for wrong batch size
        logits = ms.Tensor(np.random.randn(2, 1000), ms.float32)  # 2 != 1
        
        with pytest.raises(ValueError):
            sampler.step(logits)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])