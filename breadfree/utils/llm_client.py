from openai import OpenAI
import os
import re
import json
from typing import Dict, Any, Optional

from breadfree.utils.logger import get_logger
logging = get_logger(__name__, mode="all")

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
        timeout_seconds: int = 60,
        max_retries: int = 2,
    ):
    """
    通用 LLM 对话接口，支持通过 config.yaml 配置多 provider（nvidia、volcano 等）。
    
    实盘增强:
    - timeout_seconds: 单次调用超时（默认 60s，实盘中避免无限等待）
    - max_retries: 失败重试次数（默认 2 次，含首次共 3 次尝试）
    - 每次重试间隔 2s（指数退避可后续优化）

    Args:
        query: User query/question
        prompt: System prompt
        model: Model name (uses provider default if not specified)
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter
        max_tokens: Maximum tokens in response
        stream: Whether to stream the response
        timeout_seconds: Timeout per attempt in seconds
        max_retries: Number of retries on failure
    """
    import time as _time

    last_error = None
    for attempt in range(1 + max_retries):
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
            
            client = OpenAI(
                api_key=api_key,
                base_url=base_url,
                timeout=float(timeout_seconds),
            )
            messages=[]
            if prompt is not None:
                messages.append({"role": "system", "content": prompt})
            if query is not None:
                messages.append({"role": "user", "content": query})
            
            retry_tag = f" (retry {attempt})" if attempt > 0 else ""
            # Log the request（model 便于与下方 token 日志对应）
            logging.info(
                f"LLM request{retry_tag} | provider={provider_name} | model={selected_model} | query_len={sum(len(m.get('content', '') or '') for m in messages)}"
            )

            start_ms = int(_time.time() * 1000)
            completion = client.chat.completions.create(
                model=selected_model,
                messages=messages,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                stream=stream,
            )
            latency_ms = int(_time.time() * 1000) - start_ms
            
            response_content = completion.choices[0].message.content or ""
            usage = getattr(completion, "usage", None)
            prompt_tokens = getattr(usage, "prompt_tokens", None) or getattr(usage, "input_tokens", None)
            completion_tokens = getattr(usage, "completion_tokens", None) or getattr(usage, "output_tokens", None)
            total_tokens = getattr(usage, "total_tokens", None) or (prompt_tokens + completion_tokens if (prompt_tokens is not None and completion_tokens is not None) else None)
            if total_tokens is None:
                total_tokens = 0

            # 日志：model、input tokens、output tokens（便于排查与计费）
            logging.info(
                f"LLM call | model={selected_model} | input_tokens={prompt_tokens} | output_tokens={completion_tokens} | total_tokens={total_tokens} | latency={latency_ms}ms"
            )
            logging.info(f"--- LLM Response ---\nContent: {response_content[:500]}{'...' if len(response_content) > 500 else ''}\n--------------------")

            return response_content, total_tokens

        except Exception as e:
            last_error = e
            error_msg = f"LLM Call Error (attempt {attempt + 1}/{1 + max_retries}): {e}"
            logging.error(error_msg)
            if attempt < max_retries:
                wait = 2 * (attempt + 1)
                logging.info(f"LLM retrying in {wait}s...")
                _time.sleep(wait)

    # All attempts failed
    logging.error(f"LLM all {1 + max_retries} attempts failed. Last error: {last_error}")
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
