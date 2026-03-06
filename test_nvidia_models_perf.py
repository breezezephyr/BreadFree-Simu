#!/usr/bin/env python3
"""
NVIDIA NIM 模型性能评估：首 token 时间(TTFT)、输出 TPS、适用场景
模型列表来自 .env 注释中的候选，测试结果供选型参考。
"""

import os
import sys
import time
from dotenv import load_dotenv

load_dotenv()

if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from openai import OpenAI

from breadfree.utils.llm_client import _load_llm_config

_, providers = _load_llm_config()
if "nvidia" not in providers:
    print("ERROR: nvidia provider not in config")
    sys.exit(1)
spec = providers["nvidia"]
base_url = spec.get("base_url")
api_key = os.environ.get("NVIDIA_API_KEY") or os.environ.get("LLM_API_KEY")

# 与 .env 中注释的候选模型一致
MODELS_TO_TEST = [
    "nvidia/llama-3.3-nemotron-super-49b-v1.5",
    "deepseek-ai/deepseek-v3.1",
    "deepseek-ai/deepseek-v3.2",
    "minimaxai/minimax-m2.5",
    "qwen/qwen3-coder-480b-a35b-instruct",
    "qwen/qwen3-235b-a22b",
    "qwen/qwen3-next-80b-a3b-thinking",
    "qwen/qwen3.5-397b-a17b",
    "moonshotai/kimi-k2-instruct-0905",
    "mistralai/mistral-large-3-675b-instruct-2512",
    "moonshotai/kimi-k2-instruct",
    "z-ai/glm4.7",
    "minimaxai/minimax-m2.1",
    "google/gemma-3-27b-it",
]

# 适用场景简要说明（按模型 ID 匹配）
MODEL_SCENARIOS = {
    "nvidia/llama-3.3-nemotron-super-49b-v1.5": "通用/指令、49B 级能力、NVIDIA 优化",
    "deepseek-ai/deepseek-v3.1": "通用/推理、长上下文、中高负载",
    "deepseek-ai/deepseek-v3.2": "推理增强、复杂推理与长链",
    "minimaxai/minimax-m2.5": "通用对话、多轮与多语言",
    "qwen/qwen3-coder-480b-a35b-instruct": "代码生成与理解、技术问答",
    "qwen/qwen3-235b-a22b": "通用强能力、复杂任务与多语言",
    "qwen/qwen3-next-80b-a3b-thinking": "深度推理、思考链、数学/逻辑",
    "qwen/qwen3.5-397b-a17b": "超大参数量、综合能力、高要求场景",
    "moonshotai/kimi-k2-instruct-0905": "通用对话、长上下文、Kimi K2",
    "mistralai/mistral-large-3-675b-instruct-2512": "超大参数量指令模型、通用/复杂任务",
    "moonshotai/kimi-k2-instruct": "通用对话、长上下文（Kimi K2）",
    "z-ai/glm4.7": "智谱 GLM-4.7、通用/多语言",
    "minimaxai/minimax-m2.1": "通用对话、多轮",
    "google/gemma-3-27b-it": "中小规模指令微调、快速推理",
}

TEST_QUERY = "What is quantitative trading? Answer in one sentence."
TEST_PROMPT = "You are a concise financial assistant."
MAX_TOKENS = 128
TIMEOUT = 10


def test_one_model_stream(model_id: str) -> dict:
    """
    流式调用，测量 TTFT（首 token 时间）与输出 TPS。
    返回: available, ttft_ms, output_tps, prompt_tokens, completion_tokens, error
    """
    if not api_key:
        return {"available": False, "error": "NVIDIA_API_KEY not set", "ttft_ms": None, "output_tps": None}
    client = OpenAI(api_key=api_key, base_url=base_url, timeout=TIMEOUT)
    messages = [
        {"role": "system", "content": TEST_PROMPT},
        {"role": "user", "content": TEST_QUERY},
    ]
    t_start = time.perf_counter()
    ttft_ms = None
    t_first = None
    t_first_chunk = None  # 首个 chunk 到达时间（部分模型不在 delta.content 里返回内容时用作 TTFT 备选）
    completion_tokens = None
    prompt_tokens = None
    total_content_len = 0
    try:
        stream = client.chat.completions.create(
            model=model_id,
            messages=messages,
            temperature=0.3,
            max_tokens=MAX_TOKENS,
            stream=True,
        )
        for chunk in stream:
            t_now = time.perf_counter()
            if t_first_chunk is None:
                t_first_chunk = t_now
            if ttft_ms is None and chunk.choices and chunk.choices[0].delta.content:
                ttft_ms = int((t_now - t_start) * 1000)
                t_first = t_now
            if chunk.choices and chunk.choices[0].delta.content:
                total_content_len += len(chunk.choices[0].delta.content or "")
            if chunk.usage:
                completion_tokens = getattr(chunk.usage, "completion_tokens", None) or getattr(chunk.usage, "output_tokens", None)
                prompt_tokens = getattr(chunk.usage, "prompt_tokens", None) or getattr(chunk.usage, "input_tokens", None)
        t_end = time.perf_counter()
        if completion_tokens is None and total_content_len > 0:
            completion_tokens = max(1, total_content_len // 4)
        if completion_tokens is None:
            completion_tokens = 0
        if prompt_tokens is None:
            prompt_tokens = 0
        # 部分模型不在 delta.content 里返回内容，用「首 chunk 时间」作为 TTFT 备选
        if ttft_ms is None and t_first_chunk is not None:
            ttft_ms = int((t_first_chunk - t_start) * 1000)
        # 输出 TPS：优先用首 token 到结束的时长，否则用首 chunk 到结束
        t_from = t_first if t_first is not None else t_first_chunk
        if t_from is not None and (t_end - t_from) > 0 and completion_tokens:
            output_tps = round(completion_tokens / (t_end - t_from), 1)
        else:
            output_tps = None
        return {
            "available": True,
            "ttft_ms": ttft_ms,
            "output_tps": output_tps,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "error": None,
        }
    except Exception as e:
        return {
            "available": False,
            "ttft_ms": None,
            "output_tps": None,
            "prompt_tokens": None,
            "completion_tokens": None,
            "error": str(e),
        }


def main():
    print("=" * 90)
    print("NVIDIA NIM 模型性能评估：首 token 时间(TTFT)、输出 TPS、适用场景")
    print("=" * 90)
    print(f"\nBase URL: {base_url}")
    print(f"API Key: {'*' * 8}...{api_key[-6:] if api_key and len(api_key) > 10 else 'NOT SET'}")
    print(f"Test: streaming, max_tokens={MAX_TOKENS}")
    print()

    results = []
    for i, model_id in enumerate(MODELS_TO_TEST, 1):
        scenario = MODEL_SCENARIOS.get(model_id, "通用")
        print(f"[{i}/{len(MODELS_TO_TEST)}] {model_id} ... ", end="", flush=True)
        r = test_one_model_stream(model_id)
        r["model_id"] = model_id
        r["scenario"] = scenario
        results.append(r)
        if r["available"]:
            ttft = f"{r['ttft_ms']}ms" if r["ttft_ms"] is not None else "-"
            tps = f"{r['output_tps']} tok/s" if r["output_tps"] is not None else "-"
            print(f"OK | TTFT={ttft} | 输出 TPS={tps}")
        else:
            err = (r["error"] or "")[:50]
            print(f"FAIL | {err}")
        if i < len(MODELS_TO_TEST):
            time.sleep(1)

    # 报告表（仅输出可用模型）
    available_results = [r for r in results if r["available"]]
    print()
    print("=" * 90)
    print("性能与适用场景报告（仅可用模型）")
    print("=" * 90)
    print(f"{'Model':<45} {'首Token(ms)':<12} {'输出TPS':<12} {'适用场景'}")
    print("-" * 90)
    for r in available_results:
        mid = r["model_id"][:43] + (".." if len(r["model_id"]) > 43 else "")
        ttft = str(r["ttft_ms"]) if r["ttft_ms"] is not None else "-"
        tps = str(r["output_tps"]) if r["output_tps"] is not None else "-"
        scenario = r["scenario"][:28] + (".." if len(r["scenario"]) > 28 else "")
        print(f"{mid:<45} {ttft:<12} {tps:<12} {scenario}")
    print("=" * 90)
    print(f"合计: 以上 {len(available_results)} 个模型可用（共测试 {len(results)} 个）")
    print()
    print("说明: TTFT=从请求发出到收到首个输出 token 的耗时；输出 TPS=生成阶段每秒输出 token 数。")
    print("      适用场景为基于模型类型的建议，实际表现以你账户下的实测为准。")
    print()


if __name__ == "__main__":
    main()
