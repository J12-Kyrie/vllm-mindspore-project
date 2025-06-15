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

# 打印当前工作目录
print("当前工作目录:", os.getcwd())

# 添加项目根目录到 Python 路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
print("项目根目录:", project_root)
sys.path.insert(0, project_root)
print("sys.path:", sys.path)

try:
    from vllm_mindspore.beam_search import BeamSearchSequence
    print("导入成功：BeamSearchSequence")
except Exception as e:
    print("导入失败：", e) 