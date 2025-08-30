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
"""NPU-optimized beam search module for vllm-mindspore."""

from .npu_beam_search import (
    BeamState,
    KVCacheBeamTracker,
    NPUBeamScoreCalculator,
    NPUBeamSearchSampler,
    create_npu_beam_search_sampler,
    integrate_beam_search_with_model_runner,
    extend_attention_metadata_for_beams
)

__all__ = [
    "BeamState",
    "KVCacheBeamTracker",
    "NPUBeamScoreCalculator",
    "NPUBeamSearchSampler",
    "create_npu_beam_search_sampler",
    "integrate_beam_search_with_model_runner",
    "extend_attention_metadata_for_beams"
]