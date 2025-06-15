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
import mindspore as ms
from transformers import AutoModelForCausalLM, AutoTokenizer

# 添加项目根目录到 Python 路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, project_root)

from vllm_mindspore.beam_search import BeamSearchSequence, BeamSearchInstance, get_beam_search_score


def main():
    # 设置 MindSpore 上下文
    ms.set_context(mode=ms.GRAPH_MODE, device_target="CPU")
    
    # 加载本地模型和分词器
    model_path = r"D:\huggingface_cache\models--Qwen--Qwen2.5-1.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True, trust_remote_code=True)
    
    # 准备输入 prompt
    prompt = "你好，"
    prompt_tokens = tokenizer.encode(prompt, return_tensors="ms")
    
    # 初始化 beam search 实例
    beam_width = 4
    max_tokens = 10
    length_penalty = 1.0
    eos_token_id = tokenizer.eos_token_id
    
    beam_instance = BeamSearchInstance(prompt_tokens=prompt_tokens[0].tolist())
    
    # 手动 beam search 生成
    for step in range(max_tokens):
        # 获取当前所有 beam 序列
        current_beams = beam_instance.beams
        print(f"Step {step}:")
        for i, beam in enumerate(current_beams):
            print(f"  Beam {i}: tokens={beam.tokens}, cum_logprob={beam.cum_logprob}")
        
        # 对每个 beam 进行前向推理，获取 logits
        all_logits = []
        for beam in current_beams:
            input_ids = ms.Tensor([beam.tokens], dtype=ms.int32)
            outputs = model(input_ids)
            logits = outputs.logits[:, -1, :]  # 取最后一个 token 的 logits
            all_logits.append(logits)
        
        # 合并所有 logits
        combined_logits = ms.ops.concat(all_logits, axis=0)
        
        # 对每个 beam 的 logits 进行 topk 采样
        topk_values, topk_indices = ms.ops.topk(combined_logits, k=beam_width)
        
        # 更新 beam 状态
        new_beams = []
        for i, (beam, topk_value, topk_index) in enumerate(zip(current_beams, topk_values, topk_indices)):
            for j in range(beam_width):
                new_token = topk_index[j].item()
                logprob = topk_value[j].item()
                new_sequence = BeamSearchSequence(
                    tokens=beam.tokens + [new_token],
                    logprobs=beam.logprobs + [{new_token: logprob}],
                    cum_logprob=beam.cum_logprob + logprob
                )
                new_beams.append(new_sequence)
        
        # 按累积对数概率排序，保留 top-k 个序列
        new_beams.sort(key=lambda x: get_beam_search_score(x.tokens, x.cum_logprob, eos_token_id, length_penalty), reverse=True)
        beam_instance.beams = new_beams[:beam_width]
        
        # 检查是否所有序列都生成了 EOS
        if all(beam.tokens[-1] == eos_token_id for beam in beam_instance.beams):
            break
    
    # 输出最终结果
    print("\nFinal beams:")
    for i, beam in enumerate(beam_instance.beams):
        text = tokenizer.decode(beam.tokens)
        print(f"  Beam {i}: {text}, score={get_beam_search_score(beam.tokens, beam.cum_logprob, eos_token_id, length_penalty)}")


if __name__ == "__main__":
    main() 