#!/usr/bin/env python3
"""
NVIDIA NIM 多模型可用性 / 延时 / TPM 测试
参考: https://build.nvidia.com/models
输出: 模型 ID、可用性、延时(ms)、input/output tokens、TPM (tokens per minute)
"""

import asyncio
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

# 从项目配置读取 NVIDIA
from breadfree.utils.llm_client import _load_llm_config

_, providers = _load_llm_config()
if "nvidia" not in providers:
    print("ERROR: nvidia provider not in config")
    sys.exit(1)
spec = providers["nvidia"]
base_url = spec.get("base_url")
api_key = os.environ.get("NVIDIA_API_KEY") or os.environ.get("LLM_API_KEY")
default_model = spec.get("model")

# 待测模型（与 .env 注释中的候选一致，见 https://build.nvidia.com/models）
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
]

TEST_QUERY = "What is quantitative trading? Answer in one sentence."
TEST_PROMPT = "You are a concise financial assistant."
MAX_TOKENS = 150
TIMEOUT = 60


def test_one_model(model_id: str) -> dict:
    """同步调用一次，返回 available, latency_ms, prompt_tokens, completion_tokens, error."""
    if not api_key:
        return {"available": False, "error": "NVIDIA_API_KEY not set", "latency_ms": None, "prompt_tokens": None, "completion_tokens": None}
    client = OpenAI(api_key=api_key, base_url=base_url, timeout=TIMEOUT)
    messages = [
        {"role": "system", "content": TEST_PROMPT},
        {"role": "user", "content": TEST_QUERY},
    ]
    t0 = time.perf_counter()
    try:
        completion = client.chat.completions.create(
            model=model_id,
            messages=messages,
            temperature=0.3,
            max_tokens=MAX_TOKENS,
        )
        latency_ms = int((time.perf_counter() - t0) * 1000)
        usage = getattr(completion, "usage", None)
        prompt_tokens = getattr(usage, "prompt_tokens", None) or getattr(usage, "input_tokens", None) or 0
        completion_tokens = getattr(usage, "completion_tokens", None) or getattr(usage, "output_tokens", None) or 0
        return {
            "available": True,
            "latency_ms": latency_ms,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "error": None,
        }
    except Exception as e:
        latency_ms = int((time.perf_counter() - t0) * 1000)
        return {
            "available": False,
            "latency_ms": latency_ms,
            "prompt_tokens": None,
            "completion_tokens": None,
            "error": str(e),
        }


def main():
    print("=" * 80)
    print("NVIDIA NIM 模型可用性 / 延时 / TPM 测试")
    print("参考: https://build.nvidia.com/models")
    print("=" * 80)
    print(f"\nBase URL: {base_url}")
    print(f"API Key: {'*' * 8}...{api_key[-6:] if api_key and len(api_key) > 10 else 'NOT SET'}")
    print(f"Default model in config: {default_model}")
    print(f"Test query: \"{TEST_QUERY[:50]}...\"")
    print()

    results = []
    for i, model_id in enumerate(MODELS_TO_TEST, 1):
        print(f"[{i}/{len(MODELS_TO_TEST)}] Testing {model_id} ... ", end="", flush=True)
        r = test_one_model(model_id)
        r["model_id"] = model_id
        results.append(r)
        if r["available"]:
            total = (r["prompt_tokens"] or 0) + (r["completion_tokens"] or 0)
            tpm = int(total / (r["latency_ms"] / 60000)) if r["latency_ms"] else 0
            print(f"OK | latency={r['latency_ms']}ms | in={r['prompt_tokens']} out={r['completion_tokens']} | TPM≈{tpm}")
        else:
            err = (r["error"] or "")[:60]
            print(f"FAIL | {err}")
        # 避免请求过快
        if i < len(MODELS_TO_TEST):
            time.sleep(1)

    # 报告表格
    print()
    print("=" * 80)
    print("报告 (可用性 / 延时(ms) / Input Tokens / Output Tokens / TPM)")
    print("=" * 80)
    print(f"{'Model':<50} {'可用':<6} {'延时(ms)':<10} {'In':<6} {'Out':<6} {'TPM':<8}")
    print("-" * 80)
    for r in results:
        mid = r["model_id"][:48] + (".." if len(r["model_id"]) > 48 else "")
        ok = "是" if r["available"] else "否"
        lat = str(r["latency_ms"]) if r["latency_ms"] is not None else "-"
        pin = str(r["prompt_tokens"]) if r["prompt_tokens"] is not None else "-"
        pout = str(r["completion_tokens"]) if r["completion_tokens"] is not None else "-"
        if r["available"] and r["latency_ms"]:
            total = (r["prompt_tokens"] or 0) + (r["completion_tokens"] or 0)
            tpm = int(total / (r["latency_ms"] / 60000))
        else:
            tpm = "-"
        print(f"{mid:<50} {ok:<6} {lat:<10} {pin:<6} {pout:<6} {tpm}")
    print("=" * 80)
    ok_count = sum(1 for r in results if r["available"])
    print(f"合计: {ok_count}/{len(results)} 个模型可用")
    print()


if __name__ == "__main__":
    main()
