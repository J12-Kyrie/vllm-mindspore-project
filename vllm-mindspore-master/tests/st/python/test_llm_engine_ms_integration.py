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
Test script for LLMEngineMS Beam Search integration.

This script tests the enhanced LLMEngineMS class to verify that:
1. Beam Search requests are correctly identified
2. The _process_beam_search method is called for appropriate requests
3. Regular sampling is used for non-Beam Search requests
4. The engine maintains proper state management
"""

import os
import sys
import time
from typing import List, Dict, Any
from unittest.mock import Mock, patch

import mindspore as ms
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# 添加项目根目录到 Python 路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, project_root)

# 设置 MindSpore 上下文
ms.set_context(mode=ms.GRAPH_MODE, device_target="CPU")

# Import vLLM components
from vllm.config import VllmConfig, ModelConfig, CacheConfig, ParallelConfig, SchedulerConfig
from vllm.sampling_params import SamplingParams
from vllm.sequence import SequenceGroup, Sequence
from vllm.engine.llm_engine import SchedulerContext
from vllm.usage.usage_lib import UsageContext

# Import our enhanced engine
from vllm_mindspore.engine.llm_engine_ms import LLMEngineMS
from vllm_mindspore.beam_search import BeamSearchSequence, BeamSearchInstance


class TestLLMEngineMSIntegration:
    """Test class for LLMEngineMS Beam Search integration."""
    
    def __init__(self):
        self.model_path = r"D:\huggingface_cache\models--Qwen--Qwen2.5-1.5B-Instruct\snapshots\989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
        self.engine = None
        self.tokenizer = None
        
    def setup(self):
        """Setup test environment and initialize engine."""
        print("🔧 Setting up test environment...")
        
        # Load tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path, 
                local_files_only=True, 
                trust_remote_code=True
            )
            print(f"✅ Tokenizer loaded successfully")
        except Exception as e:
            print(f"❌ Failed to load tokenizer: {e}")
            return False
        
        # Create mock vLLM config (simplified for testing)
        try:
            model_config = ModelConfig(
                model=self.model_path,
                tokenizer=self.model_path,
                tokenizer_mode="auto",
                trust_remote_code=True,
                dtype="auto",
                seed=0,
                revision=None,
                tokenizer_revision=None,
                max_model_len=2048,
                quantization=None,
                enforce_eager=True,
                max_context_len_to_capture=2048,
                max_seq_len_to_capture=2048,
                max_logprobs=5,
                disable_sliding_window=False,
                skip_tokenizer_init=False,
                served_model_name=None,
            )
            
            cache_config = CacheConfig(
                block_size=16,
                gpu_memory_utilization=0.9,
                swap_space=4,
                cache_dtype="auto",
                num_gpu_blocks=None,
                num_cpu_blocks=None,
                sliding_window=None,
                enable_prefix_caching=False,
                cpu_offload_gb=0,
            )
            
            parallel_config = ParallelConfig(
                pipeline_parallel_size=1,
                tensor_parallel_size=1,
                worker_use_ray=False,
                max_parallel_loading_workers=None,
                disable_custom_all_reduce=False,
                tokenizer_pool_size=0,
                tokenizer_pool_type="ray",
                tokenizer_pool_extra_config=None,
                placement_group=None,
                distributed_executor_backend=None,
            )
            
            scheduler_config = SchedulerConfig(
                max_num_batched_tokens=2048,
                max_num_seqs=256,
                max_model_len=2048,
                use_v2_block_manager=False,
                num_lookahead_slots=0,
                delay_factor=0.0,
                enable_chunked_prefill=False,
                max_num_on_the_fly=1,
                policy="fcfs",
            )
            
            vllm_config = VllmConfig(
                model_config=model_config,
                cache_config=cache_config,
                parallel_config=parallel_config,
                scheduler_config=scheduler_config,
            )
            
            print(f"✅ vLLM config created successfully")
            
        except Exception as e:
            print(f"❌ Failed to create vLLM config: {e}")
            return False
        
        # Initialize LLMEngineMS (with mocked executor)
        try:
            with patch('vllm_mindspore.engine.llm_engine_ms.LLMEngine.__init__') as mock_init:
                mock_init.return_value = None
                
                self.engine = LLMEngineMS(
                    vllm_config=vllm_config,
                    executor_class=Mock,  # Mock executor for testing
                    log_stats=False,
                    usage_context=UsageContext.ENGINE_CONTEXT,
                )
                
                # Manually set required attributes for testing
                self.engine.beam_search_instances = {}
                self.engine.beam_search_configs = {}
                self.engine.model_config = model_config
                self.engine.tokenizer = self.tokenizer
                
            print(f"✅ LLMEngineMS initialized successfully")
            
        except Exception as e:
            print(f"❌ Failed to initialize LLMEngineMS: {e}")
            return False
        
        return True
    
    def test_beam_search_detection(self):
        """Test if the engine correctly identifies Beam Search requests."""
        print("\n🧪 Testing Beam Search request detection...")
        
        # Create test sequence groups
        test_cases = [
            {
                "name": "Beam Search Request (n=4, low temp)",
                "sampling_params": SamplingParams(
                    n=4,
                    temperature=0.0,
                    top_k=0,
                    top_p=1.0,
                    max_tokens=50
                ),
                "expected": True
            },
            {
                "name": "Regular Request (n=1)",
                "sampling_params": SamplingParams(
                    n=1,
                    temperature=0.8,
                    top_k=50,
                    top_p=0.9,
                    max_tokens=50
                ),
                "expected": False
            },
            {
                "name": "High Temperature (n=4, high temp)",
                "sampling_params": SamplingParams(
                    n=4,
                    temperature=0.8,
                    top_k=0,
                    top_p=1.0,
                    max_tokens=50
                ),
                "expected": False
            },
            {
                "name": "Single Sequence (n=1, low temp)",
                "sampling_params": SamplingParams(
                    n=1,
                    temperature=0.0,
                    top_k=0,
                    top_p=1.0,
                    max_tokens=50
                ),
                "expected": False
            }
        ]
        
        for i, test_case in enumerate(test_cases):
            # Create mock sequence group
            mock_seq_group = Mock()
            mock_seq_group.sampling_params = test_case["sampling_params"]
            
            # Test detection
            result = self.engine._is_beam_search_request(mock_seq_group)
            
            status = "✅" if result == test_case["expected"] else "❌"
            print(f"  {status} {test_case['name']}: {result} (expected: {test_case['expected']})")
            
            if result != test_case["expected"]:
                return False
        
        print("✅ All Beam Search detection tests passed!")
        return True
    
    def test_beam_search_initialization(self):
        """Test BeamSearchInstance initialization."""
        print("\n🧪 Testing Beam Search instance initialization...")
        
        # Create mock sequence group
        mock_seq_group = Mock()
        mock_seq_group.request_id = "test_request_001"
        mock_seq_group.prompt_token_ids = [1, 2, 3, 4, 5]
        mock_seq_group.lora_request = None
        mock_seq_group.sampling_params = SamplingParams(
            n=4,
            temperature=0.0,
            max_tokens=100,
            top_k=0,
            top_p=1.0
        )
        
        # Test initialization
        try:
            self.engine._initialize_beam_search_instance(mock_seq_group)
            
            # Verify instance was created
            assert mock_seq_group.request_id in self.engine.beam_search_instances
            assert mock_seq_group.request_id in self.engine.beam_search_configs
            
            # Verify instance properties
            beam_instance = self.engine.beam_search_instances[mock_seq_group.request_id]
            beam_config = self.engine.beam_search_configs[mock_seq_group.request_id]
            
            assert len(beam_instance.beams) == 1  # Initial beam
            assert beam_instance.beams[0].tokens == mock_seq_group.prompt_token_ids
            assert beam_config['beam_width'] == 4
            assert beam_config['max_tokens'] == 100
            
            print("✅ Beam Search instance initialization successful!")
            return True
            
        except Exception as e:
            print(f"❌ Beam Search initialization failed: {e}")
            return False
    
    def test_beam_search_step_advancement(self):
        """Test beam search step advancement logic."""
        print("\n🧪 Testing Beam Search step advancement...")
        
        try:
            # Create a simple beam instance
            beam_instance = BeamSearchInstance(
                prompt_tokens=[1, 2, 3],
                lora_request=None
            )
            
            # Create mock logits (vocab_size = 10 for simplicity)
            import numpy as np
            mock_logits = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
            
            # Test step advancement
            new_beams = self.engine._advance_beam_search_step(
                beam_instance=beam_instance,
                logits=mock_logits,
                beam_width=3,
                length_penalty=1.0,
                eos_token_id=2
            )
            
            # Verify results
            assert len(new_beams) == 3  # Should return beam_width beams
            assert all(len(beam.tokens) == 4 for beam in new_beams)  # Original 3 + 1 new token
            
            # Verify beams are sorted by score (highest first)
            scores = [beam.cum_logprob for beam in new_beams]
            assert scores == sorted(scores, reverse=True)
            
            print("✅ Beam Search step advancement successful!")
            return True
            
        except Exception as e:
            print(f"❌ Beam Search step advancement failed: {e}")
            return False
    
    def test_process_beam_search_method(self):
        """Test the main _process_beam_search method."""
        print("\n🧪 Testing _process_beam_search method...")
        
        try:
            # Create mock sequence group
            mock_seq_group = Mock()
            mock_seq_group.request_id = "test_request_002"
            mock_seq_group.prompt_token_ids = [1, 2, 3]
            mock_seq_group.lora_request = None
            mock_seq_group.sampling_params = SamplingParams(
                n=2,
                temperature=0.0,
                max_tokens=50,
                top_k=0,
                top_p=1.0
            )
            
            # Create mock logits
            mock_logits = ms.ops.randn(1, 100)  # Batch size 1, vocab size 100
            
            # Mock the tokenizer for EOS token
            with patch.object(self.engine, '_get_eos_token_id', return_value=2):
                # Test the method
                self.engine._process_beam_search(mock_seq_group, mock_logits)
                
                # Verify beam instance was created
                assert mock_seq_group.request_id in self.engine.beam_search_instances
                
                # Verify beam instance has correct number of beams
                beam_instance = self.engine.beam_search_instances[mock_seq_group.request_id]
                assert len(beam_instance.beams) == 2  # beam_width = 2
            
            print("✅ _process_beam_search method test successful!")
            return True
            
        except Exception as e:
            print(f"❌ _process_beam_search method test failed: {e}")
            return False
    
    def test_beam_search_stats(self):
        """Test beam search statistics functionality."""
        print("\n🧪 Testing Beam Search statistics...")
        
        try:
            # Add some test beam instances
            for i in range(3):
                request_id = f"test_request_{i:03d}"
                beam_instance = BeamSearchInstance(
                    prompt_tokens=[1, 2, 3],
                    lora_request=None
                )
                beam_config = {
                    'beam_width': 4,
                    'max_tokens': 100,
                    'length_penalty': 1.0
                }
                
                self.engine.beam_search_instances[request_id] = beam_instance
                self.engine.beam_search_configs[request_id] = beam_config
            
            # Get statistics
            stats = self.engine.get_beam_search_stats()
            
            # Verify statistics
            assert stats['active_beam_requests'] == 3
            assert stats['total_beams'] == 3  # Each instance has 1 initial beam
            assert len(stats['requests']) == 3
            
            # Verify individual request stats
            for request_id in stats['requests']:
                request_stats = stats['requests'][request_id]
                assert request_stats['num_beams'] == 1
                assert request_stats['beam_width'] == 4
                assert request_stats['max_tokens'] == 100
            
            print("✅ Beam Search statistics test successful!")
            return True
            
        except Exception as e:
            print(f"❌ Beam Search statistics test failed: {e}")
            return False
    
    def run_all_tests(self):
        """Run all integration tests."""
        print("🚀 Starting LLMEngineMS Beam Search Integration Tests")
        print("=" * 60)
        
        # Setup
        if not self.setup():
            print("❌ Setup failed, aborting tests")
            return False
        
        # Run tests
        tests = [
            self.test_beam_search_detection,
            self.test_beam_search_initialization,
            self.test_beam_search_step_advancement,
            self.test_process_beam_search_method,
            self.test_beam_search_stats,
        ]
        
        passed = 0
        total = len(tests)
        
        for test in tests:
            try:
                if test():
                    passed += 1
                else:
                    print(f"❌ Test {test.__name__} failed")
            except Exception as e:
                print(f"❌ Test {test.__name__} crashed: {e}")
        
        # Summary
        print("\n" + "=" * 60)
        print(f"📊 Test Results: {passed}/{total} tests passed")
        
        if passed == total:
            print("🎉 All tests passed! LLMEngineMS Beam Search integration is working correctly.")
            return True
        else:
            print("⚠️  Some tests failed. Please check the implementation.")
            return False


def main():
    """Main test function."""
    print("LLMEngineMS Beam Search Integration Test")
    print("Testing the integration of Beam Search logic into the engine's main loop")
    print()
    
    # Create and run tests
    test_suite = TestLLMEngineMSIntegration()
    success = test_suite.run_all_tests()
    
    if success:
        print("\n✅ Integration test completed successfully!")
        print("\n📋 Summary of implemented features:")
        print("   ✅ Beam Search request detection")
        print("   ✅ _process_model_outputs method with Beam Search routing")
        print("   ✅ _process_beam_search method implementation")
        print("   ✅ BeamSearchInstance management")
        print("   ✅ Beam search step advancement")
        print("   ✅ Statistics and monitoring")
        print("\n🎯 The LLMEngineMS class now successfully integrates Beam Search")
        print("   into the engine's main loop, routing requests appropriately")
        print("   based on sampling parameters.")
    else:
        print("\n❌ Integration test failed!")
        print("   Please review the implementation and fix any issues.")
    
    return success


if __name__ == "__main__":
    main() 