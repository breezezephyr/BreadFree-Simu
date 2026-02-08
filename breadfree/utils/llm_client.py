from openai import OpenAI
import os
import re
import json
from typing import Dict, Any, Optional

from breadfree.utils.logger import get_logger
logging = get_logger(__name__, mode="file")

# 默认配置（无 config.yaml 或未配置 llm 时使用）
_DEFAULT_PROVIDERS = {
    "nvidia": {
        "base_url": "https://integrate.api.nvidia.com/v1",
        "model": "minimaxai/minimax-m2.1",
        "env_key": "NVIDIA_API_KEY",
    },
    "volcano": {
        "base_url": "https://ark.cn-beijing.volces.com/api/v3",
        "model": os.environ.get("ARK_MODEL", "ep-20251208192433-wsbrk"),
        "env_key": "ARK_API_KEY",
    },
}


def _load_llm_config() -> tuple:
    """从 config.yaml 读取 llm.active 和 llm.providers，否则用环境变量 + 默认配置。"""
    config_path = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
    if not os.path.exists(config_path):
        provider = os.environ.get("LLM_PROVIDER", "volcano").lower()
        configs = _DEFAULT_PROVIDERS
        return provider, configs
    try:
        import yaml
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    except Exception:
        provider = os.environ.get("LLM_PROVIDER", "volcano").lower()
        return provider, _DEFAULT_PROVIDERS
    llm = cfg.get("llm") or {}
    providers = llm.get("providers") or _DEFAULT_PROVIDERS
    active = (llm.get("active") or os.environ.get("LLM_PROVIDER") or "volcano").lower()
    return active, providers


def _get_llm_client_config():
    """当前生效的 provider 名与配置（base_url, model, api_key）。"""
    active, providers = _load_llm_config()
    if active not in providers:
        active = "volcano" if "volcano" in providers else list(providers.keys())[0]
    spec = providers[active]
    env_key = spec.get("env_key", "LLM_API_KEY")
    api_key = os.environ.get(env_key) or os.environ.get("LLM_API_KEY")
    return active, {
        "base_url": spec.get("base_url"),
        "model": spec.get("model"),
        "api_key": api_key,
    }


# 兼容旧用法：导出“当前” provider 与 key（供测试脚本等用）
LLM_PROVIDER, _current_spec = _get_llm_client_config()
LLM_API_KEY = _current_spec.get("api_key")
PROVIDER_CONFIGS = {k: {"base_url": v.get("base_url"), "default_model": v.get("model")} for k, v in _load_llm_config()[1].items()}


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
    通用 LLM 对话接口，支持通过 config.yaml 配置多 provider（nvidia、volcano 等）。
    
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
        # 每次调用时从 config.yaml + 环境变量取当前 LLM 配置
        provider_name, client_spec = _get_llm_client_config()
        api_key = client_spec.get("api_key")
        if not api_key or api_key == "YOUR_API_KEY_HERE":
            env_key = (_load_llm_config()[1].get(provider_name) or {}).get("env_key", "LLM_API_KEY")
            raise RuntimeError(
                f"No valid API key for provider '{provider_name}'. Set {env_key} or LLM_API_KEY in .env"
            )
        base_url = client_spec.get("base_url")
        selected_model = model or client_spec.get("model")
        if not base_url or not selected_model:
            raise ValueError(f"Missing base_url or model for provider '{provider_name}' in config")
        
        client = OpenAI(api_key=api_key, base_url=base_url)
        messages=[]
        if prompt is not None:
            messages.append({"role": "system", "content": prompt})
        if query is not None:
            messages.append({"role": "user", "content": query})
        
        # Log the request
        logging.info(
            f"--- LLM Request ---\n"
            f"Provider: {provider_name}\n"
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
