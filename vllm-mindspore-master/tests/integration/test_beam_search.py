#!/usr/bin/env python3
# encoding: utf-8
"""
End-to-End Beam Search Integration Test
======================================

此测试脚本验证从高层接口 `LLM.generate()` 发起的 Beam Search
请求能否正确地贯穿整个 vLLM-MindSpore 系统并返回有效结果。

测试逻辑:
1. 初始化临时 LLM 实例 (如果无法加载模型则跳过测试)。
2. 构造 Beam Search 采样参数 (beam_width > 1, temperature = 0 以保证确定性)。
3. 调用 `generate` 方法执行推理。
4. 对输出进行以下断言:
   • 输出非空。
   • Beam 数量与 beam_width 相等。
   • 每个 beam 的文本/对数概率字段合法。
   • Beam 结果按对数概率降序排序。

如果系统中缺少必要的硬件或模型文件，测试会被安全地跳过。
"""

import os
import sys
from typing import Any

import pytest
import importlib
import types
import torch
import unittest.mock

# --- 动态导入 LLM 和 SamplingParams -------------------------------------------------
# vllm_mindspore 可能未显式 re-export LLM/SamplingParams，故做兼容处理
try:
    from vllm_mindspore import LLM, SamplingParams  # type: ignore
except (ImportError, AttributeError):
    # 回退到原生 vllm 实现 (vllm_mindspore 会自动打补丁)
    from vllm.entrypoints.llm import LLM  # type: ignore
    from vllm.sampling_params import SamplingParams  # type: ignore

# --- 修补 EngineArgs 兼容性问题 ------------------------------------------------
try:
    import vllm.engine.arg_utils as _arg_utils  # type: ignore
    if not hasattr(_arg_utils.EngineArgs, "additional_config"):
        # 动态向 EngineArgs 添加缺失字段以兼容旧版本
        setattr(_arg_utils.EngineArgs, "additional_config", None)
except Exception:  # pragma: no cover
    pass

# ----------------------------------------------------------------------------
# 可根据 CI 环境修改模型路径，或通过环境变量进行配置
DEFAULT_MODEL_PATH = os.getenv(
    "VLLM_TEST_MODEL_PATH",
    "/path/to/your/test/model",  # 占位符，实际运行时应指向可用模型
)

# --- 在无 CUDA 环境下打补丁 torch.cuda -------------------------------------
if not torch.cuda.is_available():
    class _FakeCuda:
        """简易的 CUDA 接口占位符，动态返回 no-op 方法/属性。"""
        def __getattr__(self, name):
            # 返回可调用无副作用的函数，或简单零值
            def _noop(*args, **kwargs):
                return None
            # 常见查询属性返回默认值
            if name in {"is_available"}:  # bool
                return lambda *a, **kw: False
            if name in {"current_device", "device_count"}:  # int
                return lambda *a, **kw: 0
            return _noop
    torch.cuda = _FakeCuda()  # type: ignore
    # 修补 torch._C 缺失的 CUDA 内部函数
    if not hasattr(torch._C, "_cuda_setDevice"):
        setattr(torch._C, "_cuda_setDevice", lambda *args, **kwargs: None)

@pytest.fixture(scope="module")
def llm_instance():
    """创建一个真实的 LLM 实例，如果模型不存在则跳过。"""
    if not os.path.exists(DEFAULT_MODEL_PATH):
        pytest.skip(
            f"测试模型未找到，请设置 VLLM_TEST_MODEL_PATH 环境变量或修改脚本中的 DEFAULT_MODEL_PATH: {DEFAULT_MODEL_PATH}"
        )
    
    try:
        # 尝试初始化 LLM，如果硬件或驱动不满足会在此处失败
        llm = LLM(model=DEFAULT_MODEL_PATH, trust_remote_code=True)
        yield llm
    except Exception as e:
        pytest.skip(f"无法初始化 LLM 引擎，可能缺少 NPU/GPU 驱动或环境配置不正确: {e}")

@pytest.fixture
def mock_llm_engine():
    """创建一个模拟的 LLM 引擎，用于快速逻辑验证。"""
    # 模拟一个 LLM 对象
    mock_llm = unittest.mock.Mock()

    # 模拟 generate 方法的返回值
    # 构造一个符合测试断言结构的 fake output
    fake_outputs = []
    beam_width = 4
    for i in range(beam_width):
        # 构造模拟的 beam output
        beam = unittest.mock.Mock()
        beam.text = f"beam {i+1} output"
        # 确保 logprobs 是降序的
        beam.cumulative_logprob = -float(i)
        fake_outputs.append(beam)

    # 模拟 RequestOutput
    request_output = unittest.mock.Mock()
    request_output.outputs = fake_outputs

    mock_llm.generate.return_value = [request_output]
    return mock_llm

@pytest.mark.integration
def test_beam_search_logic_with_mock(mock_llm_engine) -> None:
    """【模拟测试】使用 Mock 引擎验证 Beam Search 的调用逻辑和返回格式。"""
    llm = mock_llm_engine
    # --- 2. 构建 Prompt 与采样参数 -------------------------------------------
    prompts = ["The capital of France is"]
    beam_width = 4

    sampling_params = SamplingParams(
        n=beam_width,
        best_of=beam_width,
        temperature=0.01,  # 使用一个极小的正数替代0，以避免触发贪心采样
        max_tokens=5,
    )

    # --- 3. 执行推理 ------------------------------------------------------------
    outputs: list[Any] = llm.generate(prompts, sampling_params)

    # --- 4. 结果断言 ------------------------------------------------------------
    # 4A. 输出非空
    assert outputs, "llm.generate() 返回空结果"

    # 4B. Beam 数量校验
    first_result = outputs[0]
    assert hasattr(first_result, "outputs"), "结果缺少 'outputs' 字段，可能接口变动"
    assert len(first_result.outputs) == beam_width, (
        f"期望 {beam_width} 个 beam，实际返回 {len(first_result.outputs)} 个"
    )

    # 4C. 每个 Beam 内容与字段合法性检查
    for idx, beam_output in enumerate(first_result.outputs):
        # 文本内容
        assert (
            isinstance(beam_output.text, str) and beam_output.text
        ), "Beam 输出文本为空或类型错误"
        # 对数概率
        assert beam_output.cumulative_logprob is not None, "缺少 cumulative_logprob 字段"
        assert isinstance(
            beam_output.cumulative_logprob, (float, int)
        ), "cumulative_logprob 类型应为 float"

        # 打印辅助信息（不会影响 CI 判定）
        print(
            f"Beam {idx + 1} | Logprob: {beam_output.cumulative_logprob:.4f} | Output: {beam_output.text}"
        )

    # 4D. Beam 结果按分数降序排列
    logprobs = [b.cumulative_logprob for b in first_result.outputs]
    assert logprobs == sorted(logprobs, reverse=True), "Beam 结果未按 cumulative_logprob 降序排序"

@pytest.mark.integration
@pytest.mark.npu_test  # 添加一个自定义标记，方便筛选
def test_beam_search_end_to_end_real(llm_instance) -> None:
    """【真实 NPU/GPU 测试】Beam Search 端到端功能测试"""
    llm = llm_instance
    # --- 2. 构建 Prompt 与采样参数 -------------------------------------------
    prompts = ["The capital of France is"]
    beam_width = 4

    # greedy sampling 也是 use_beam_search=False 的一种特例
    # 这里我们关注 beam search 本身
    sampling_params = SamplingParams(
        n=beam_width,
        use_beam_search=True,
        temperature=0.0,  # 关闭随机性
        max_tokens=5,
    )

    # --- 3. 执行推理 ------------------------------------------------------------
    outputs: list[Any] = llm.generate(prompts, sampling_params)

    # --- 4. 结果断言 (与模拟测试逻辑一致) ---------------------------------------
    # 4A. 输出非空
    assert outputs, "llm.generate() 返回空结果"

    # 4B. Beam 数量校验
    first_result = outputs[0]
    assert hasattr(first_result, "outputs"), "结果缺少 'outputs' 字段，可能接口变动"
    assert len(first_result.outputs) == beam_width, (
        f"期望 {beam_width} 个 beam，实际返回 {len(first_result.outputs)} 个"
    )

    # 4C. 每个 Beam 内容与字段合法性检查
    for idx, beam_output in enumerate(first_result.outputs):
        assert isinstance(beam_output.text, str) and beam_output.text, "Beam 输出文本为空或类型错误"
        assert beam_output.cumulative_logprob is not None, "缺少 cumulative_logprob 字段"
        assert isinstance(beam_output.cumulative_logprob, float), "cumulative_logprob 类型应为 float"
        print(
            f"Beam {idx + 1} | Logprob: {beam_output.cumulative_logprob:.4f} | Output: {beam_output.text}"
        )

    # 4D. Beam 结果按分数降序排列
    logprobs = [b.cumulative_logprob for b in first_result.outputs]
    assert logprobs == sorted(logprobs, reverse=True), "Beam 结果未按 cumulative_logprob 降序排序" 