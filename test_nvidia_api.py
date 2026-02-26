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

# Import the LLM client
from breadfree.utils.llm_client import async_hunyuan_chat, LLM_PROVIDER, LLM_API_KEY, PROVIDER_CONFIGS


async def test_nvidia_api():
    """Test NVIDIA NIM API connection and response"""
    
    print("=" * 60)
    print("[TEST] NVIDIA NIM API Connection Test")
    print("=" * 60)
    
    # Check configuration
    print(f"\n[CONFIG] Configuration:")
    print(f"   Provider: {LLM_PROVIDER}")
    print(f"   API Key: {LLM_API_KEY[:10]}...{LLM_API_KEY[-5:] if LLM_API_KEY else 'NOT SET'}")
    
    if LLM_PROVIDER in PROVIDER_CONFIGS:
        config = PROVIDER_CONFIGS[LLM_PROVIDER]
        print(f"   Base URL: {config['base_url']}")
        print(f"   Default Model: {config['default_model']}")
    
    if not LLM_API_KEY or LLM_API_KEY == "your_nvidia_api_key_here":
        print("\n[ERROR] API Key not configured!")
        print("   Please set LLM_API_KEY in .env file")
        print("   Get your key from: https://build.nvidia.com/settings/api-keys")
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
            temperature=0.7,
            max_tokens=200
        )
        
        print("\n[SUCCESS] API Response:")
        print("-" * 60)
        print(response)
        print("-" * 60)
        print(f"\n[STATS] Tokens used: {tokens}")
        print("\n[DONE] NVIDIA NIM API is working correctly!")
        
    except Exception as e:
        print(f"\n[ERROR] {e}")
        print("\n[TIPS] Troubleshooting tips:")
        print("   1. Check your API key is correct")
        print("   2. Ensure you have internet connection")
        print("   3. Visit https://build.nvidia.com/ to verify your account")
        print("   4. Check logs in logs/ directory for details")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    asyncio.run(test_nvidia_api())
