#!/usr/bin/env python3
"""
Test script for NVIDIA NIM API integration
测试 NVIDIA NIM API 集成
"""

import asyncio
import os
import sys
from dotenv import load_dotenv

# Configure UTF-8 encoding for Windows console
if sys.platform == 'win32':
    try:
        if hasattr(sys.stdout, 'reconfigure'):
            sys.stdout.reconfigure(encoding='utf-8')
        if hasattr(sys.stderr, 'reconfigure'):
            sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Load environment variables
load_dotenv()

from breadfree.utils.llm_client import async_hunyuan_chat, _load_llm_config

# 本脚本只测 NVIDIA，从配置中读取 nvidia 的 spec（调用时传 provider="nvidia" 强制走 NVIDIA）
_, _PROVIDERS = _load_llm_config()
if "nvidia" not in _PROVIDERS:
    raise RuntimeError("NVIDIA provider not found in config. Check breadfree/config.yaml llm.providers.nvidia")
_NVIDIA_SPEC = _PROVIDERS["nvidia"]
NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY") or os.environ.get("LLM_API_KEY")


async def test_nvidia_api():
    """Test NVIDIA NIM API connection and response"""
    
    print("=" * 60)
    print("[TEST] NVIDIA NIM API Connection Test")
    print("=" * 60)
    
    # Check configuration (NVIDIA only)
    print(f"\n[CONFIG] Configuration:")
    print(f"   Provider: nvidia (forced for this test)")
    print(f"   API Key: {NVIDIA_API_KEY[:10]}...{NVIDIA_API_KEY[-5:] if NVIDIA_API_KEY else 'NOT SET'}")
    print(f"   Base URL: {_NVIDIA_SPEC.get('base_url')}")
    print(f"   Default Model: {_NVIDIA_SPEC.get('model') or 'NOT SET'}")
    
    if not NVIDIA_API_KEY or NVIDIA_API_KEY == "your_nvidia_api_key_here":
        print("\n[ERROR] NVIDIA API Key not configured!")
        print("   Please set NVIDIA_API_KEY in .env file")
        print("   Get your key from: https://build.nvidia.com/settings/api-keys")
        return
    
    if not _NVIDIA_SPEC.get("model"):
        print("\n[ERROR] NVIDIA model not configured!")
        print("   Please set NVIDIA_MODEL in .env (e.g. meta/llama-3.1-8b-instruct)")
        return
    
    # Test query
    test_query = "What is quantitative trading? Answer in one sentence."
    print(f"\n[QUERY] Test Query:")
    print(f"   '{test_query}'")
    print("\n[WAIT] Sending request to NVIDIA NIM API...")
    
    try:
        response, tokens = await async_hunyuan_chat(
            query=test_query,
            prompt="You are a helpful assistant specializing in finance and trading.",
            provider="nvidia",
            temperature=0.7,
            max_tokens=200
        )

        if not (response and response.strip()) or tokens == 0:
            print("\n[FAIL] No response or 0 tokens — 当前模型 ID 可能不可用（404）。")
            _print_404_model_tip()
        else:
            print("\n[SUCCESS] API Response:")
            print("-" * 60)
            print(response)
            print("-" * 60)
            print(f"\n[STATS] Tokens used: {tokens}")
            print("\n[DONE] NVIDIA NIM API is working correctly!")

    except Exception as e:
        print(f"\n[ERROR] {e}")
        err_str = str(e).lower()
        if "404" in err_str or "not found" in err_str or "function" in err_str:
            _print_404_model_tip()
        else:
            print("\n[TIPS] Troubleshooting tips:")
            print("   1. Check your API key is correct")
            print("   2. Ensure you have internet connection")
            print("   3. Visit https://build.nvidia.com/ to verify your account")
            print("   4. Check logs in logs/ directory for details")

    print("\n" + "=" * 60)


def _print_404_model_tip():
    """当前配置的模型在 NVIDIA 上 404 时的说明与建议。"""
    print("\n[TIP] 当前 .env 中的 NVIDIA_MODEL 在该账户下不可用（404）。")
    print("      请在 .env 中改为 NVIDIA 官方目录中的模型 ID，例如：")
    print("        NVIDIA_MODEL=meta/llama-3.1-8b-instruct")
    print("        或  NVIDIA_MODEL=deepseek-ai/deepseek-v3.1")
    print("      可用性/延时可运行: uv run python test_nvidia_models_benchmark.py")


if __name__ == "__main__":
    asyncio.run(test_nvidia_api())
