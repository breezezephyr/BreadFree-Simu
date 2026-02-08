#!/usr/bin/env python3
"""
Test script for multi-agent multi-model configuration
测试多智能体多模型配置
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

from breadfree.utils.llm_client import async_hunyuan_chat, LLM_PROVIDER, LLM_API_KEY

# Agent model configuration
AGENT_MODELS = {
    "Market Analyst": "qwen/qwq-32b",
    "Risk Manager": "minimaxai/minimax-m2.1",
    "Fund Manager": "deepseek-ai/deepseek-v3.2"
}

async def test_agent_model(agent_name, model_name, query):
    """Test a specific agent model"""
    print(f"\n{'='*60}")
    print(f"Testing {agent_name} ({model_name})")
    print(f"{'='*60}")
    
    try:
        response, _ = await async_hunyuan_chat(
            query=query,
            model=model_name,
            temperature=0.7,
            max_tokens=200
        )
        
        print(f"[OK] {agent_name} Response:")
        print(f"{response[:200]}..." if len(response) > 200 else response)
        return True
        
    except Exception as e:
        print(f"[ERROR] {agent_name} failed: {e}")
        return False

async def test_all_agents():
    """Test all agent models"""
    print("="*60)
    print("Multi-Agent Multi-Model Configuration Test")
    print("="*60)
    print(f"LLM Provider: {LLM_PROVIDER}")
    print(f"API Key Configured: {'Yes' if LLM_API_KEY else 'No'}")
    
    if not LLM_API_KEY:
        print("\n[ERROR] No LLM_API_KEY configured!")
        print("Please set LLM_API_KEY in your .env file")
        return
    
    # Test queries for each agent
    test_queries = {
        "Market Analyst": "分析A股市场当前趋势，考虑技术面和政策面因素。",
        "Risk Manager": "当前账户持仓比例60%，市场波动率25%，评估风险等级。",
        "Fund Manager": "综合市场趋势和风险评估，给出交易决策（BUY/SELL/HOLD）。"
    }
    
    results = {}
    
    # Test each agent model
    for agent_name, model_name in AGENT_MODELS.items():
        query = test_queries[agent_name]
        success = await test_agent_model(agent_name, model_name, query)
        results[agent_name] = success
    
    # Summary
    print(f"\n{'='*60}")
    print("Test Summary")
    print(f"{'='*60}")
    
    for agent_name, success in results.items():
        status = "[OK]" if success else "[FAIL]"
        model = AGENT_MODELS[agent_name]
        print(f"{status} {agent_name}: {model}")
    
    all_success = all(results.values())
    
    if all_success:
        print(f"\n[SUCCESS] All agent models are working correctly!")
        print(f"You can now run: python main.py --strategy AgentStrategy")
    else:
        print(f"\n[WARNING] Some models failed. Please check the errors above.")
        print(f"You may need to verify model availability at https://build.nvidia.com/models")

if __name__ == "__main__":
    asyncio.run(test_all_agents())
