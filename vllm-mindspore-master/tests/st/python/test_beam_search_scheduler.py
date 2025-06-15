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


@dataclass
class BeamSearchRequest:
    """Beam Search 请求的数据结构"""
    request_id: str
    prompt: str
    prompt_tokens: List[int]
    beam_width: int
    max_tokens: int
    length_penalty: float
    temperature: float
    beam_instance: BeamSearchInstance
    completed: bool = False
    start_time: float = 0.0
    completion_time: Optional[float] = None


class BeamSearchScheduler:
    """Beam Search 调度器，用于管理多个 Beam Search 请求"""
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.requests: Dict[str, BeamSearchRequest] = {}
        self.eos_token_id = tokenizer.eos_token_id
        
    def add_request(
        self,
        request_id: str,
        prompt: str,
        beam_width: int = 4,
        max_tokens: int = 10,
        length_penalty: float = 1.0,
        temperature: float = 1.0
    ) -> None:
        """添加新的 Beam Search 请求"""
        prompt_tokens = self.tokenizer.encode(prompt)
        beam_instance = BeamSearchInstance(prompt_tokens=prompt_tokens)
        
        request = BeamSearchRequest(
            request_id=request_id,
            prompt=prompt,
            prompt_tokens=prompt_tokens,
            beam_width=beam_width,
            max_tokens=max_tokens,
            length_penalty=length_penalty,
            temperature=temperature,
            beam_instance=beam_instance,
            start_time=time.time()
        )
        
        self.requests[request_id] = request
        print(f"[调度器] 添加请求 {request_id}: '{prompt}' (beam_width={beam_width}, max_tokens={max_tokens})")
    
    def get_active_requests(self) -> List[BeamSearchRequest]:
        """获取所有活跃的（未完成的）请求"""
        return [req for req in self.requests.values() if not req.completed]
    
    def step_all_requests(self) -> int:
        """推进所有活跃请求的一步 Beam Search"""
        active_requests = self.get_active_requests()
        if not active_requests:
            return 0
        
        print(f"\n[调度器] 推进 {len(active_requests)} 个活跃请求...")
        
        # 批量处理所有活跃请求
        for request in active_requests:
            self._step_single_request(request)
        
        # 检查完成状态
        completed_count = 0
        for request in active_requests:
            if self._is_request_completed(request):
                request.completed = True
                request.completion_time = time.time()
                completed_count += 1
                print(f"[调度器] 请求 {request.request_id} 已完成")
        
        return len(active_requests) - completed_count
    
    def _step_single_request(self, request: BeamSearchRequest) -> None:
        """推进单个请求的一步 Beam Search"""
        current_beams = request.beam_instance.beams
        
        # 对每个 beam 进行前向推理
        all_logits = []
        for beam in current_beams:
            input_ids = torch.tensor([beam.tokens], dtype=torch.int32)
            outputs = self.model(input_ids)
            logits = outputs.logits[:, -1, :]  # 取最后一个 token 的 logits
            all_logits.append(logits)
        
        # 合并所有 logits
        combined_logits = torch.cat(all_logits, dim=0)
        
        # 对每个 beam 的 logits 进行 topk 采样
        topk_values, topk_indices = torch.topk(combined_logits, k=request.beam_width)
        
        # 更新 beam 状态
        new_beams = []
        for i, (beam, topk_value, topk_index) in enumerate(zip(current_beams, topk_values, topk_indices)):
            for j in range(request.beam_width):
                new_token = topk_index[j].item()
                logprob = topk_value[j].item()
                new_sequence = BeamSearchSequence(
                    tokens=beam.tokens + [new_token],
                    logprobs=beam.logprobs + [{new_token: logprob}],
                    cum_logprob=beam.cum_logprob + logprob
                )
                new_beams.append(new_sequence)
        
        # 按累积对数概率排序，保留 top-k 个序列
        new_beams.sort(
            key=lambda x: get_beam_search_score(x.tokens, x.cum_logprob, self.eos_token_id, request.length_penalty),
            reverse=True
        )
        request.beam_instance.beams = new_beams[:request.beam_width]
    
    def _is_request_completed(self, request: BeamSearchRequest) -> bool:
        """检查请求是否已完成"""
        # 检查是否达到最大 token 数
        max_length = len(request.prompt_tokens) + request.max_tokens
        if any(len(beam.tokens) >= max_length for beam in request.beam_instance.beams):
            return True
        
        # 检查是否所有 beam 都生成了 EOS
        if all(beam.tokens[-1] == self.eos_token_id for beam in request.beam_instance.beams):
            return True
        
        return False
    
    def get_request_results(self, request_id: str) -> Optional[List[Tuple[str, float]]]:
        """获取指定请求的结果"""
        if request_id not in self.requests:
            return None
        
        request = self.requests[request_id]
        results = []
        
        for i, beam in enumerate(request.beam_instance.beams):
            text = self.tokenizer.decode(beam.tokens)
            score = get_beam_search_score(beam.tokens, beam.cum_logprob, self.eos_token_id, request.length_penalty)
            results.append((text, score))
        
        return results
    
    def print_status(self) -> None:
        """打印调度器状态"""
        total_requests = len(self.requests)
        active_requests = len(self.get_active_requests())
        completed_requests = total_requests - active_requests
        
        print(f"\n[调度器状态] 总请求: {total_requests}, 活跃: {active_requests}, 已完成: {completed_requests}")
        
        for request_id, request in self.requests.items():
            status = "已完成" if request.completed else "进行中"
            elapsed_time = (request.completion_time or time.time()) - request.start_time
            print(f"  - {request_id}: {status} (耗时: {elapsed_time:.2f}s)")


def main():
    """主测试函数"""
    print("=== Beam Search 调度器测试 ===")
    
    # 设置 MindSpore 上下文
    ms.set_context(mode=ms.GRAPH_MODE, device_target="CPU")
    
    # 加载本地模型和分词器
    model_path = r"D:\huggingface_cache\models--Qwen--Qwen2.5-1.5B-Instruct\snapshots\989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
    print(f"加载模型: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True, trust_remote_code=True)
    
    # 创建调度器
    scheduler = BeamSearchScheduler(model, tokenizer)
    
    # 添加多个 Beam Search 请求
    test_prompts = [
        ("req_1", "你好，", 3, 8),
        ("req_2", "今天天气", 4, 10),
        ("req_3", "人工智能", 2, 6),
        ("req_4", "编程语言", 3, 7),
    ]
    
    for request_id, prompt, beam_width, max_tokens in test_prompts:
        scheduler.add_request(
            request_id=request_id,
            prompt=prompt,
            beam_width=beam_width,
            max_tokens=max_tokens,
            length_penalty=1.0,
            temperature=1.0
        )
    
    # 调度器循环推进所有请求
    max_steps = 15
    step = 0
    
    while step < max_steps:
        active_count = scheduler.step_all_requests()
        step += 1
        
        scheduler.print_status()
        
        if active_count == 0:
            print(f"\n[调度器] 所有请求已完成，总共执行了 {step} 步")
            break
        
        print(f"[调度器] 第 {step} 步完成，剩余 {active_count} 个活跃请求")
    
    # 输出所有请求的最终结果
    print("\n=== 最终结果 ===")
    for request_id, _, _, _ in test_prompts:
        results = scheduler.get_request_results(request_id)
        if results:
            print(f"\n请求 {request_id}:")
            for i, (text, score) in enumerate(results):
                print(f"  Beam {i}: {text} (score: {score:.4f})")
        else:
            print(f"\n请求 {request_id}: 未找到结果")
    
    print("\n=== 测试完成 ===")


if __name__ == "__main__":
    main() 