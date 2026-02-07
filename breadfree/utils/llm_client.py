from openai import OpenAI
import os
import re
import json
from datetime import datetime
from typing import Dict, Any, Optional

from breadfree.utils.logger import get_logger
logging = get_logger(__name__, mode="file")

# LLM Provider Configuration
# Supports: "nvidia", "hunyuan"
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "nvidia").lower()
LLM_API_KEY = os.environ.get("LLM_API_KEY") or os.environ.get("NVIDIA_API_KEY") or os.environ.get("HUNYUAN_API_KEY")

# Provider-specific configurations
PROVIDER_CONFIGS = {
    "nvidia": {
        "base_url": "https://integrate.api.nvidia.com/v1",
        "default_model": "minimaxai/minimax-m2.1",  # MiniMax M2.1 model
    },
    "hunyuan": {
        "base_url": "https://api.lkeap.cloud.tencent.com/v1",
        "default_model": "deepseek-v3.2",  # deepseek-r1, deepseek-v3.1
    }
}


async def async_hunyuan_chat(
        query=None,
        prompt=None,
        model=None,
        temperature=0.2, 
        top_p=0.3, 
        max_tokens=4096,
        stream=False,
    ):
    """
    Universal LLM chat function supporting multiple providers (NVIDIA, Hunyuan, etc.)
    
    Args:
        query: User query/question
        prompt: System prompt
        model: Model name (uses provider default if not specified)
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter
        max_tokens: Maximum tokens in response
        stream: Whether to stream the response
    """
    try:
        # Check API key at runtime instead of import time
        if not LLM_API_KEY or LLM_API_KEY == "YOUR_API_KEY_HERE":
            raise RuntimeError(
                f"No valid LLM_API_KEY environment variable detected. "
                f"Please set LLM_API_KEY (or NVIDIA_API_KEY/HUNYUAN_API_KEY) in .env file and try again."
            )
        
        # Get provider configuration
        if LLM_PROVIDER not in PROVIDER_CONFIGS:
            raise ValueError(f"Unsupported LLM_PROVIDER: {LLM_PROVIDER}. Supported: {list(PROVIDER_CONFIGS.keys())}")
        
        config = PROVIDER_CONFIGS[LLM_PROVIDER]
        selected_model = model or config["default_model"]
        
        # Construct client
        client = OpenAI(
            api_key=LLM_API_KEY,
            base_url=config["base_url"]
        )
        messages=[]
        if prompt is not None:
            messages.append({"role": "system", "content": prompt})
        if query is not None:
            messages.append({"role": "user", "content": query})
        
        # Log the request
        logging.info(
            f"--- LLM Request ---\n"
            f"Provider: {LLM_PROVIDER}\n"
            f"Model: {selected_model}\n"
            f"Query: {query}\n"
            f"-------------------"
        )
        completion = client.chat.completions.create(
            model=selected_model,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stream=stream,
        )
        
        response_content = completion.choices[0].message.content
        total_tokens = completion.usage.total_tokens
        
        # Log the response
        logging.info(f"--- LLM Response ---\nContent: {response_content}\nTokens: {total_tokens}\n--------------------")

        return response_content, total_tokens
    except Exception as e:
        error_msg = f"LLM Call Error: {e}"
        print(error_msg)
        logging.error(error_msg)
        return "", 0

def parse_llm_response(response: str, fallback: Dict[str, Any]) -> Dict[str, Any]:
    """
    Parse JSON from LLM response string with markdown block support and fallback.
    """
    try:
        patterns = [
            r'```json\s*(.*?)\s*```',
            r'```\s*(.*?)\s*```',
            r'({.*})'
        ]
        for pattern in patterns:
            match = re.search(pattern, response, re.DOTALL)
            if match:
                return json.loads(match.group(1))
        return json.loads(response)
    except Exception:
        return fallback
