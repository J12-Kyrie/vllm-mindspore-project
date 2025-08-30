#!/usr/bin/env python3
# encoding: utf-8
# Copyright 2025 Huawei Technologies Co., Ltd
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

"""Test script for NPU beam search implementation."""

import sys
import os
import torch
from typing import List, Dict, Any

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from vllm_mindspore.beam_search import (
        BeamState,
        KVCacheBeamTracker,
        NPUBeamScoreCalculator,
        NPUBeamSearchSampler,
        create_npu_beam_search_sampler
    )
    from vllm_mindspore.worker.cache_engine import (
        BeamSearchCacheEngine,
        create_beam_search_cache_engine
    )
    from vllm_mindspore.worker.model_runner import (
        BeamSearchModelRunner,
        create_beam_search_model_runner
    )
    from vllm_mindspore.worker.worker import (
        BeamSearchWorker,
        create_beam_search_worker
    )
    print("✓ All beam search modules imported successfully")
except ImportError as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)


def test_beam_state():
    """Test BeamState functionality."""
    print("\n=== Testing BeamState ===")
    
    try:
        # Create a beam state
        beam_state = BeamState(
            sequence_ids=[1, 2, 3],
            score=0.5,
            length=3,
            finished=False
        )
        
        assert beam_state.sequence_ids == [1, 2, 3]
        assert beam_state.score == 0.5
        assert beam_state.length == 3
        assert not beam_state.finished
        
        print("✓ BeamState creation and attribute access works")
        
        # Test beam state update
        beam_state.sequence_ids.append(4)
        beam_state.score = 0.7
        beam_state.length = 4
        
        assert beam_state.sequence_ids == [1, 2, 3, 4]
        assert beam_state.score == 0.7
        assert beam_state.length == 4
        
        print("✓ BeamState update works")
        
    except Exception as e:
        print(f"✗ BeamState test failed: {e}")
        return False
    
    return True


def test_kv_cache_beam_tracker():
    """Test KVCacheBeamTracker functionality."""
    print("\n=== Testing KVCacheBeamTracker ===")
    
    try:
        # Create a KV cache beam tracker
        tracker = KVCacheBeamTracker(
            beam_width=4,
            max_blocks_per_beam=10,
            block_size=16
        )
        
        assert tracker.beam_width == 4
        assert tracker.max_blocks_per_beam == 10
        assert tracker.block_size == 16
        
        print("✓ KVCacheBeamTracker creation works")
        
        # Test beam allocation
        beam_id = tracker.allocate_beam()
        assert beam_id == 0
        
        beam_id2 = tracker.allocate_beam()
        assert beam_id2 == 1
        
        print("✓ KVCacheBeamTracker beam allocation works")
        
        # Test beam release
        tracker.release_beam(beam_id)
        assert beam_id not in tracker.beam_blocks
        
        print("✓ KVCacheBeamTracker beam release works")
        
    except Exception as e:
        print(f"✗ KVCacheBeamTracker test failed: {e}")
        return False
    
    return True


def test_npu_beam_score_calculator():
    """Test NPUBeamScoreCalculator functionality."""
    print("\n=== Testing NPUBeamScoreCalculator ===")
    
    try:
        # Create a score calculator
        calculator = NPUBeamScoreCalculator(
            length_penalty=1.0,
            device="cpu"  # Use CPU for testing
        )
        
        assert calculator.length_penalty == 1.0
        assert calculator.device == "cpu"
        
        print("✓ NPUBeamScoreCalculator creation works")
        
        # Test score calculation
        logits = torch.randn(2, 1000)  # 2 beams, 1000 vocab size
        beam_scores = torch.tensor([0.5, 0.3])
        lengths = torch.tensor([5, 3])
        
        scores = calculator.calculate_scores(logits, beam_scores, lengths)
        assert scores.shape == (2, 1000)
        
        print("✓ NPUBeamScoreCalculator score calculation works")
        
    except Exception as e:
        print(f"✗ NPUBeamScoreCalculator test failed: {e}")
        return False
    
    return True


def test_npu_beam_search_sampler():
    """Test NPUBeamSearchSampler functionality."""
    print("\n=== Testing NPUBeamSearchSampler ===")
    
    try:
        # Create a beam search sampler
        sampler = NPUBeamSearchSampler(
            beam_width=4,
            max_length=50,
            length_penalty=1.0,
            device="cpu"
        )
        
        assert sampler.beam_width == 4
        assert sampler.max_length == 50
        
        print("✓ NPUBeamSearchSampler creation works")
        
        # Test initialization
        initial_tokens = torch.tensor([[1, 2, 3]])  # Single sequence
        sampler.initialize(initial_tokens)
        
        assert len(sampler.beams) == 4  # Should have beam_width beams
        
        print("✓ NPUBeamSearchSampler initialization works")
        
        # Test step (simplified)
        logits = torch.randn(4, 1000)  # 4 beams, 1000 vocab size
        sampler.step(logits)
        
        print("✓ NPUBeamSearchSampler step works")
        
    except Exception as e:
        print(f"✗ NPUBeamSearchSampler test failed: {e}")
        return False
    
    return True


def test_factory_functions():
    """Test factory functions."""
    print("\n=== Testing Factory Functions ===")
    
    try:
        # Test beam search sampler factory
        sampler = create_npu_beam_search_sampler(
            beam_width=4,
            max_length=50,
            length_penalty=1.0,
            device="cpu"
        )
        
        assert isinstance(sampler, NPUBeamSearchSampler)
        assert sampler.beam_width == 4
        
        print("✓ create_npu_beam_search_sampler works")
        
    except Exception as e:
        print(f"✗ Factory function test failed: {e}")
        return False
    
    return True


def test_integration():
    """Test integration between components."""
    print("\n=== Testing Component Integration ===")
    
    try:
        # Create components
        sampler = create_npu_beam_search_sampler(
            beam_width=2,
            max_length=20,
            length_penalty=1.0,
            device="cpu"
        )
        
        # Test initialization and basic workflow
        initial_tokens = torch.tensor([[1, 2]])  # Single sequence
        sampler.initialize(initial_tokens)
        
        # Simulate a few steps
        for step in range(3):
            logits = torch.randn(2, 100)  # 2 beams, 100 vocab size
            sampler.step(logits)
        
        # Get results
        results = sampler.get_best_sequences()
        assert len(results) > 0
        
        print("✓ Component integration works")
        
    except Exception as e:
        print(f"✗ Integration test failed: {e}")
        return False
    
    return True


def run_all_tests():
    """Run all tests."""
    print("Starting NPU Beam Search Implementation Tests...")
    
    tests = [
        test_beam_state,
        test_kv_cache_beam_tracker,
        test_npu_beam_score_calculator,
        test_npu_beam_search_sampler,
        test_factory_functions,
        test_integration
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print(f"\n=== Test Results ===")
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! NPU beam search implementation is working correctly.")
        return True
    else:
        print(f"❌ {total - passed} tests failed. Please check the implementation.")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)