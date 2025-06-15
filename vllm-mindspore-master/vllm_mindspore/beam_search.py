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

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional, Union, Dict

from vllm_mindspore.lora.request import LoRARequest
print('LoRARequest import ok')
from vllm_mindspore.sequence import Logprob
print('Logprob import ok')

if TYPE_CHECKING:
    from vllm.multimodal.inputs import BatchedTensorInputs
    MultiModalDataDict = BatchedTensorInputs
else:
    MultiModalDataDict = Dict[str, Any]

print('准备定义BeamSearchSequence')
@dataclass
class BeamSearchSequence:
    """A sequence for beam search.
    It keeps track of the tokens and the log probability of the sequence.
    The text field is optional and will only be filled when the sequence is
    about to be returned to the user.
    """
    print('开始定义BeamSearchSequence字段')
    # The tokens includes the prompt.
    tokens: list[int]
    print('tokens字段定义完成')
    logprobs: list[dict[int, Logprob]]
    print('logprobs字段定义完成')
    lora_request: Optional[LoRARequest] = None
    print('lora_request字段定义完成')
    cum_logprob: float = 0.0
    print('cum_logprob字段定义完成')
    text: Optional[str] = None
    print('text字段定义完成')
    finish_reason: Optional[str] = None
    print('finish_reason字段定义完成')
    stop_reason: Union[int, str, None] = None
    print('stop_reason字段定义完成')
    multi_modal_data: Optional["MultiModalDataDict"] = None
    print('multi_modal_data字段定义完成')
    mm_processor_kwargs: Optional[dict[str, Any]] = None
    print('mm_processor_kwargs字段定义完成')


print('准备定义BeamSearchOutput')
@dataclass
class BeamSearchOutput:
    """The output of beam search.
    It contains the list of the best beam search sequences.
    The length of the list is equal to the beam width.
    """
    sequences: list[BeamSearchSequence]


print('准备定义BeamSearchInstance')
class BeamSearchInstance:

    def __init__(
        self,
        prompt_tokens: list[int],
        lora_request: Optional[LoRARequest] = None,
        logprobs: Optional[list[dict[int, Logprob]]] = None,
        **kwargs,
    ):
        self.beams: list[BeamSearchSequence] = [
            BeamSearchSequence(
                tokens=prompt_tokens,
                logprobs=[] if logprobs is None else list(logprobs),
                lora_request=lora_request,
                **kwargs,
            )
        ]
        self.completed: list[BeamSearchSequence] = []


def get_beam_search_score(
    tokens: list[int],
    cumulative_logprob: float,
    eos_token_id: int,
    length_penalty: float = 1.0,
) -> float:
    """Calculate the beam search score with length penalty.

    Adapted from

    https://github.com/huggingface/transformers/blob/ccb92be23def445f2afdea94c31286f84b89eb5b/src/transformers/generation/beam_search.py#L938
    """
    seq_len = len(tokens)
    if tokens[-1] == eos_token_id:
        seq_len -= 1

    return cumulative_logprob / (seq_len**length_penalty)


def create_sort_beams_key_function(eos_token_id: int, length_penalty: float):

    def sort_beams_key(x: BeamSearchSequence) -> float:
        return get_beam_search_score(x.tokens, x.cum_logprob, eos_token_id,
                                     length_penalty)

    return sort_beams_key 