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

import os
import sys
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import torch
import mindspore as ms
from transformers import AutoModelForCausalLM, AutoTokenizer

# 添加项目根目录到 Python 路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, project_root)

from vllm_mindspore.beam_search import BeamSearchSequence, BeamSearchInstance, get_beam_search_score
from vllm_mindspore.worker.cache_engine import CacheEngine
from vllm_mindspore.attention.layer import Attention
from vllm_mindspore.core.block_manager import BlockManager
from vllm_mindspore.sequence import Sequence

@dataclass
class KVCacheTestResult:
    """KV Cache Fork 测试结果"""
    test_name: str
    success: bool
    message: str
    execution_time: float

class KVCacheForkTester:
    """KV Cache Fork 机制测试器"""
    
    def __init__(self):
        self.results: List[KVCacheTestResult] = []
        
    def test_cache_engine_fork(self) -> KVCacheTestResult:
        """测试 CacheEngine 的 fork_kv_cache 方法"""
        start_time = time.time()
        
        try:
            # 创建 CacheEngine 实例
            cache_engine = CacheEngine(model_config=None)
            
            # 创建测试序列
            parent_seq = BeamSearchSequence(
                tokens=[1, 2, 3, 4],
                logprobs=[],
                cum_logprob=-1.5
            )
            
            child_seq = BeamSearchSequence(
                tokens=[1, 2, 3, 4, 5],
                logprobs=[],
                cum_logprob=-2.0
            )
            
            # 模拟 KV cache 数据
            request_id = "test_request_1"
            fake_kv_cache = [
                (ms.Tensor([[1.0, 2.0], [3.0, 4.0]]), ms.Tensor([[5.0, 6.0], [7.0, 8.0]])),
                (ms.Tensor([[9.0, 10.0], [11.0, 12.0]]), ms.Tensor([[13.0, 14.0], [15.0, 16.0]]))
            ]
            fake_slot_mapping = [0, 1]
            
            # 设置父序列的 KV cache
            cache_engine.update_kv_cache(request_id, fake_kv_cache, fake_slot_mapping)
            
            # 执行 fork 操作
            cache_engine.fork_kv_cache(request_id, parent_seq, child_seq)
            
            # 验证 fork 后的 KV cache
            forked_cache = cache_engine.get_kv_cache(request_id)
            
            if forked_cache is not None and len(forked_cache) == 2:
                execution_time = time.time() - start_time
                return KVCacheTestResult(
                    test_name="CacheEngine Fork",
                    success=True,
                    message="KV Cache fork 操作成功完成",
                    execution_time=execution_time
                )
            else:
                execution_time = time.time() - start_time
                return KVCacheTestResult(
                    test_name="CacheEngine Fork",
                    success=False,
                    message="KV Cache fork 后数据验证失败",
                    execution_time=execution_time
                )
                
        except Exception as e:
            execution_time = time.time() - start_time
            return KVCacheTestResult(
                test_name="CacheEngine Fork",
                success=False,
                message=f"测试过程中发生异常: {str(e)}",
                execution_time=execution_time
            )
    
    def test_attention_layer_fork(self) -> KVCacheTestResult:
        """测试 Attention 层的 fork_kv_cache 方法"""
        start_time = time.time()
        
        try:
            # 创建测试数据
            parent_key_cache = ms.Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
            parent_value_cache = ms.Tensor([[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]])
            parent_slot_mapping = ms.Tensor([0, 1])
            child_slot_mapping = ms.Tensor([0, 1])
            
            # 创建 Attention 实例（简化版本，仅用于测试）
            attention = Attention(
                num_heads=2,
                head_size=3,
                scale=1.0,
                num_kv_heads=2,
                alibi_slopes=None,
                cache_config=None,
                quant_config=None,
                blocksparse_params=None,
                logits_soft_cap=None
            )
            
            # 执行 fork 操作
            child_key_cache, child_value_cache = attention.fork_kv_cache(
                parent_key_cache, parent_value_cache, 
                parent_slot_mapping, child_slot_mapping
            )
            
            # 验证结果
            if child_key_cache.shape == parent_key_cache.shape and \
               child_value_cache.shape == parent_value_cache.shape:
                execution_time = time.time() - start_time
                return KVCacheTestResult(
                    test_name="Attention Layer Fork",
                    success=True,
                    message="Attention 层 KV Cache fork 操作成功完成",
                    execution_time=execution_time
                )
            else:
                execution_time = time.time() - start_time
                return KVCacheTestResult(
                    test_name="Attention Layer Fork",
                    success=False,
                    message="Attention 层 KV Cache fork 后形状验证失败",
                    execution_time=execution_time
                )
                
        except Exception as e:
            execution_time = time.time() - start_time
            return KVCacheTestResult(
                test_name="Attention Layer Fork",
                success=False,
                message=f"测试过程中发生异常: {str(e)}",
                execution_time=execution_time
            )
    
    def test_block_manager_fork(self) -> KVCacheTestResult:
        """测试 BlockManager 的 fork 方法"""
        start_time = time.time()
        
        try:
            # 创建 BlockManager 实例
            block_manager = BlockManager()
            
            # 创建测试序列
            parent_seq = Sequence(
                seq_id=1, 
                inputs={"type": "token", "prompt_token_ids": [1, 2, 3]}, 
                block_size=10
            )
            child_seq1 = Sequence(
                seq_id=2, 
                inputs={"type": "token", "prompt_token_ids": [1, 2, 3]}, 
                block_size=10
            )
            child_seq2 = Sequence(
                seq_id=3, 
                inputs={"type": "token", "prompt_token_ids": [1, 2, 3]}, 
                block_size=10
            )
            
            # 为父序列分配物理块
            block_manager.allocate(parent_seq.seq_id, [1, 2, 3])
            
            # 执行 fork 操作
            block_manager.fork(parent_seq, [child_seq1, child_seq2])
            
            # 验证结果
            parent_blocks = block_manager.get_block_table(parent_seq.seq_id)
            child1_blocks = block_manager.get_block_table(child_seq1.seq_id)
            child2_blocks = block_manager.get_block_table(child_seq2.seq_id)
            
            # 验证子序列的 block_table 与父序列相同
            if parent_blocks == child1_blocks == child2_blocks:
                # 验证引用计数
                ref_counts_correct = all(
                    block_manager.get_ref_count(block_num) == 3 
                    for block_num in parent_blocks
                )
                
                if ref_counts_correct:
                    execution_time = time.time() - start_time
                    return KVCacheTestResult(
                        test_name="BlockManager Fork",
                        success=True,
                        message="BlockManager fork 操作成功完成，引用计数正确",
                        execution_time=execution_time
                    )
                else:
                    execution_time = time.time() - start_time
                    return KVCacheTestResult(
                        test_name="BlockManager Fork",
                        success=False,
                        message="BlockManager fork 后引用计数验证失败",
                        execution_time=execution_time
                    )
            else:
                execution_time = time.time() - start_time
                return KVCacheTestResult(
                    test_name="BlockManager Fork",
                    success=False,
                    message="BlockManager fork 后 block_table 验证失败",
                    execution_time=execution_time
                )
                
        except Exception as e:
            execution_time = time.time() - start_time
            return KVCacheTestResult(
                test_name="BlockManager Fork",
                success=False,
                message=f"测试过程中发生异常: {str(e)}",
                execution_time=execution_time
            )
    
    def test_sequence_fork(self) -> KVCacheTestResult:
        """测试 Sequence 的 fork 方法"""
        start_time = time.time()
        
        try:
            # 创建原始序列
            original_seq = Sequence(
                seq_id=1,
                inputs={"type": "token", "prompt_token_ids": [1, 2, 3, 4]},
                block_size=10
            )
            
            # 执行 fork 操作
            forked_seq = original_seq.fork(new_seq_id=2)
            
            # 验证结果
            if (forked_seq.seq_id == 2 and 
                forked_seq.inputs["prompt_token_ids"] == original_seq.inputs["prompt_token_ids"] and
                forked_seq.seq_id != original_seq.seq_id):
                
                execution_time = time.time() - start_time
                return KVCacheTestResult(
                    test_name="Sequence Fork",
                    success=True,
                    message="Sequence fork 操作成功完成",
                    execution_time=execution_time
                )
            else:
                execution_time = time.time() - start_time
                return KVCacheTestResult(
                    test_name="Sequence Fork",
                    success=False,
                    message="Sequence fork 后数据验证失败",
                    execution_time=execution_time
                )
                
        except Exception as e:
            execution_time = time.time() - start_time
            return KVCacheTestResult(
                test_name="Sequence Fork",
                success=False,
                message=f"测试过程中发生异常: {str(e)}",
                execution_time=execution_time
            )
    
    def run_all_tests(self) -> None:
        """运行所有 KV Cache Fork 测试"""
        print("🧪 开始 KV Cache Fork 机制测试...")
        print("=" * 60)
        
        # 运行各项测试
        tests = [
            self.test_cache_engine_fork,
            self.test_attention_layer_fork,
            self.test_block_manager_fork,
            self.test_sequence_fork
        ]
        
        for test_func in tests:
            result = test_func()
            self.results.append(result)
            
            # 打印测试结果
            status = "✅ 通过" if result.success else "❌ 失败"
            print(f"{status} {result.test_name}")
            print(f"   消息: {result.message}")
            print(f"   执行时间: {result.execution_time:.4f}秒")
            print("-" * 60)
        
        # 打印总结
        self.print_summary()
    
    def print_summary(self) -> None:
        """打印测试总结"""
        total_tests = len(self.results)
        passed_tests = sum(1 for result in self.results if result.success)
        failed_tests = total_tests - passed_tests
        total_time = sum(result.execution_time for result in self.results)
        
        print("\n📊 KV Cache Fork 机制测试总结")
        print("=" * 60)
        print(f"总测试数: {total_tests}")
        print(f"通过测试: {passed_tests} ✅")
        print(f"失败测试: {failed_tests} ❌")
        print(f"成功率: {(passed_tests/total_tests)*100:.1f}%")
        print(f"总执行时间: {total_time:.4f}秒")
        
        if failed_tests > 0:
            print("\n❌ 失败的测试:")
            for result in self.results:
                if not result.success:
                    print(f"  - {result.test_name}: {result.message}")
        
        print("\n🎯 KV Cache Fork 机制实现状态:")
        
        # 分析实现状态
        cache_engine_ok = any(r.test_name == "CacheEngine Fork" and r.success for r in self.results)
        attention_ok = any(r.test_name == "Attention Layer Fork" and r.success for r in self.results)
        block_manager_ok = any(r.test_name == "BlockManager Fork" and r.success for r in self.results)
        sequence_ok = any(r.test_name == "Sequence Fork" and r.success for r in self.results)
        
        components = [
            ("CacheEngine Fork", cache_engine_ok),
            ("Attention Layer Fork", attention_ok),
            ("BlockManager Fork", block_manager_ok),
            ("Sequence Fork", sequence_ok)
        ]
        
        for component, status in components:
            status_icon = "✅" if status else "❌"
            print(f"  {status_icon} {component}")
        
        # 总体评估
        if all(status for _, status in components):
            print("\n🎉 KV Cache Fork 机制已完整实现！")
        elif any(status for _, status in components):
            print("\n⚠️  KV Cache Fork 机制部分实现，需要进一步完善。")
        else:
            print("\n🚨 KV Cache Fork 机制尚未实现或存在严重问题。")

def main():
    """主函数"""
    print("🚀 vLLM-MindSpore KV Cache Fork 机制测试")
    print("=" * 60)
    
    # 创建测试器并运行测试
    tester = KVCacheForkTester()
    tester.run_all_tests()

if __name__ == "__main__":
    main() 