from openai import OpenAI
import os
import re
import json
from typing import Dict, Any, Optional

from breadfree.utils.logger import get_logger
logging = get_logger(__name__, mode="all")

# 默认结构（无 config.yaml 时使用），仅 base_url + env_key，模型全部从 .env 读取
_DEFAULT_PROVIDERS = {
    "nvidia": {
        "base_url": "https://integrate.api.nvidia.com/v1",
        "env_key": "NVIDIA_API_KEY",
    },
    "volcano": {
        "base_url": "https://ark.cn-beijing.volces.com/api/v3",
        "env_key": "ARK_API_KEY",
    },
}

# .env 变量名：主模型 + 可选分角色模型（不设则用主模型）
_ENV_MODEL_KEYS = {
    "nvidia": ("NVIDIA_MODEL", "NVIDIA_MODEL_ANALYST", "NVIDIA_MODEL_RISK", "NVIDIA_MODEL_FUND"),
    "volcano": ("ARK_MODEL", "ARK_MODEL_ANALYST", "ARK_MODEL_RISK", "ARK_MODEL_FUND"),
}


def _parse_model_list(val: Optional[str]) -> list:
    """解析 .env 中逗号分隔的多模型，返回列表（空串/空返回 []）。"""
    if not val or not str(val).strip():
        return []
    return [m.strip() for m in str(val).split(",") if m.strip()]


def _build_providers_from_env(providers_raw: dict) -> dict:
    """
    从 config 结构（base_url, env_key）与 .env 组装完整 providers。
    主模型、agent_models 全部来自 .env；主模型支持逗号分隔多模型，用于首 token 超时退阶。
    """
    out = {}
    for pname, spec in providers_raw.items():
        s = dict(spec)
        keys = _ENV_MODEL_KEYS.get(pname, (None, None, None, None))
        main_key, analyst_key, risk_key, fund_key = keys
        main_raw = os.environ.get(main_key) if main_key else None
        model_list = _parse_model_list(main_raw)
        s["model_list"] = model_list
        s["model"] = model_list[0] if model_list else None
        # 分角色：可选逗号分隔，取第一个
        def _first_model(env_key: Optional[str]) -> Optional[str]:
            raw = os.environ.get(env_key) if env_key else None
            lst = _parse_model_list(raw)
            return lst[0] if lst else None
        am = {
            "market_analyst": _first_model(analyst_key) or s["model"],
            "risk_manager": _first_model(risk_key) or s["model"],
            "fund_manager": _first_model(fund_key) or s["model"],
        }
        s["agent_models"] = am
        out[pname] = s
    return out


def _load_llm_config() -> tuple:
    """从 config.yaml 读取 llm.providers 结构（仅 base_url、env_key），模型与 agent_models 全部从 .env 读取。"""
    config_path = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
    if not os.path.exists(config_path):
        active = (os.environ.get("LLM_PROVIDER") or "volcano").lower()
        return active, _build_providers_from_env(_DEFAULT_PROVIDERS)
    try:
        import yaml
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    except Exception:
        active = (os.environ.get("LLM_PROVIDER") or "volcano").lower()
        return active, _build_providers_from_env(_DEFAULT_PROVIDERS)
    llm = cfg.get("llm") or {}
    providers_raw = llm.get("providers") or _DEFAULT_PROVIDERS
    providers = _build_providers_from_env(providers_raw)
    active = (os.environ.get("LLM_PROVIDER") or "volcano").lower()
    return active, providers


def _get_llm_client_config():
    """当前生效的 provider 名与配置（base_url, model, model_list, api_key）。"""
    active, providers = _load_llm_config()
    if active not in providers:
        active = "volcano" if "volcano" in providers else list(providers.keys())[0]
    spec = providers[active]
    env_key = spec.get("env_key", "LLM_API_KEY")
    api_key = os.environ.get(env_key) or os.environ.get("LLM_API_KEY")
    model_list = spec.get("model_list") or ([spec.get("model")] if spec.get("model") else [])
    return active, {
        "base_url": spec.get("base_url"),
        "model": spec.get("model"),
        "model_list": model_list,
        "api_key": api_key,
    }


# 兼容旧用法：导出“当前” provider 与 key（供测试脚本等用）
LLM_PROVIDER, _current_spec = _get_llm_client_config()
LLM_API_KEY = _current_spec.get("api_key")
PROVIDER_CONFIGS = {k: {"base_url": v.get("base_url"), "default_model": v.get("model")} for k, v in _load_llm_config()[1].items()}

# 当前进程内 LLM 跑测累计 token 消耗（每次成功调用后累加）
_llm_token_sum: int = 0
_llm_call_count: int = 0


def get_llm_token_sum() -> Dict[str, Any]:
    """返回当前进程内 LLM 调用的累计 token 消耗与调用次数。"""
    return {"total_tokens": _llm_token_sum, "call_count": _llm_call_count}


def reset_llm_token_sum() -> None:
    """重置累计 token 与调用次数（新一轮跑测前可调）。"""
    global _llm_token_sum, _llm_call_count
    _llm_token_sum = 0
    _llm_call_count = 0


def _add_llm_token_usage(total_tokens: int) -> None:
    global _llm_token_sum, _llm_call_count
    _llm_token_sum += total_tokens
    _llm_call_count += 1


def _log_llm_request_prompt(messages: list) -> None:
    """打印 LLM 请求的 prompt（system + user）。"""
    if not messages:
        logging.info("--- LLM Request Prompt --- (empty)\n--------------------")
        return
    parts = []
    for m in messages:
        role = m.get("role", "unknown")
        content = m.get("content") or ""
        parts.append(f"[{role}]\n{content}")
    full = "\n\n".join(parts)
    logging.info("--- LLM Request Prompt ---\n%s\n--------------------", full)


def _first_token_timeout_seconds() -> int:
    """首 token 超时阈值（秒），超过则退阶下一模型。.env: LLM_FIRST_TOKEN_TIMEOUT_SECONDS，默认 60。"""
    try:
        return max(5, int(os.environ.get("LLM_FIRST_TOKEN_TIMEOUT_SECONDS", "60")))
    except Exception:
        return 60


async def async_hunyuan_chat(
        query=None,
        prompt=None,
        model=None,
        provider=None,
        temperature=0.2,
        top_p=0.3,
        max_tokens=4096,
        stream=False,
        timeout_seconds: int = 60,
        max_retries: int = 2,
    ):
    """
    通用 LLM 对话接口，支持多 provider、多模型逗号分隔、首 token 超时退阶。

    .env 支持 ARK_MODEL / NVIDIA_MODEL 逗号分隔多模型；若当前模型首 token 响应超过
    LLM_FIRST_TOKEN_TIMEOUT_SECONDS（默认 60s），自动退阶使用下一个模型。
    """
    import time as _time

    last_error = None
    for attempt in range(1 + max_retries):
        try:
            if provider is not None:
                _, providers = _load_llm_config()
                if provider not in providers:
                    raise ValueError(f"Unknown provider '{provider}'. Available: {list(providers.keys())}")
                spec = providers[provider]
                env_key = spec.get("env_key", "LLM_API_KEY")
                api_key = os.environ.get(env_key) or os.environ.get("LLM_API_KEY")
                provider_name = provider
                client_spec = {
                    "base_url": spec.get("base_url"),
                    "model": spec.get("model"),
                    "model_list": spec.get("model_list") or ([spec.get("model")] if spec.get("model") else []),
                    "api_key": api_key,
                }
            else:
                provider_name, client_spec = _get_llm_client_config()
                # 确保有 model_list（单模型时为 [model]）
                if "model_list" not in client_spec:
                    m = client_spec.get("model")
                    client_spec["model_list"] = [m] if m else []

            api_key = client_spec.get("api_key")
            if not api_key or api_key == "YOUR_API_KEY_HERE":
                _, providers = _load_llm_config()
                env_key = (providers.get(provider_name) or {}).get("env_key", "LLM_API_KEY")
                raise RuntimeError(
                    f"No valid API key for provider '{provider_name}'. Set {env_key} or LLM_API_KEY in .env"
                )
            base_url = client_spec.get("base_url")
            model_list = list(client_spec.get("model_list") or [])
            if model:
                model_list = [model]
            if not base_url or not model_list:
                raise ValueError(f"Missing base_url or model for provider '{provider_name}' in config")

            client = OpenAI(
                api_key=api_key,
                base_url=base_url,
                timeout=float(timeout_seconds),
            )
            messages = []
            if prompt is not None:
                messages.append({"role": "system", "content": prompt})
            if query is not None:
                messages.append({"role": "user", "content": query})

            # 打印本次请求的 prompt（system + user）
            _log_llm_request_prompt(messages)

            ttft_limit = _first_token_timeout_seconds()
            response_content = ""
            total_tokens = 0
            used_model = None

            for selected_model in model_list:
                retry_tag = f" (retry {attempt})" if attempt > 0 else ""
                logging.info(
                    f"LLM request{retry_tag} | provider={provider_name} | model={selected_model} | ttft_limit={ttft_limit}s"
                )
                start_ms = int(_time.time() * 1000)
                first_token_ms = None
                chunks_content = []
                usage_info = None
                try:
                    stream_handle = client.chat.completions.create(
                        model=selected_model,
                        messages=messages,
                        temperature=temperature,
                        top_p=top_p,
                        max_tokens=max_tokens,
                        stream=True,
                    )
                    for chunk in stream_handle:
                        if first_token_ms is None:
                            delta = getattr(chunk.choices[0], "delta", None) if chunk.choices else None
                            content = getattr(delta, "content", None) if delta else None
                            if content:
                                first_token_ms = int(_time.time() * 1000) - start_ms
                                if first_token_ms > ttft_limit * 1000:
                                    logging.warning(
                                        f"LLM ttft={first_token_ms}ms > {ttft_limit}s for model={selected_model}, fallback to next model"
                                    )
                                    break
                        if chunk.choices:
                            delta = getattr(chunk.choices[0], "delta", None)
                            if delta and getattr(delta, "content", None):
                                chunks_content.append(delta.content)
                        usage = getattr(chunk, "usage", None)
                        if usage is not None:
                            usage_info = usage
                    else:
                        response_content = "".join(chunks_content)
                        used_model = selected_model
                        if usage_info:
                            prompt_tokens = getattr(usage_info, "prompt_tokens", None) or getattr(usage_info, "input_tokens", None)
                            completion_tokens = getattr(usage_info, "completion_tokens", None) or getattr(usage_info, "output_tokens", None)
                            total_tokens = getattr(usage_info, "total_tokens", None) or (0 if (prompt_tokens is None or completion_tokens is None) else prompt_tokens + completion_tokens)
                        else:
                            total_tokens = 0
                        latency_ms = int(_time.time() * 1000) - start_ms
                        logging.info(
                            f"LLM call | model={used_model} | ttft={first_token_ms or 0}ms | total_tokens={total_tokens} | latency={latency_ms}ms"
                        )
                        logging.info(f"--- LLM Response ---\nContent: {response_content[:500]}{'...' if len(response_content) > 500 else ''}\n--------------------")
                        _add_llm_token_usage(total_tokens)
                        return response_content, total_tokens
                except Exception as e:
                    logging.warning(f"LLM model={selected_model} failed: {e}, try next model")
                    last_error = e
                    continue

            if not response_content and last_error:
                raise last_error
            if not response_content:
                logging.error("LLM all models in list failed or exceeded ttft limit")
                return "", 0
            return response_content, total_tokens

        except Exception as e:
            last_error = e
            logging.error(f"LLM Call Error (attempt {attempt + 1}/{1 + max_retries}): {e}")
            if attempt < max_retries:
                wait = 2 * (attempt + 1)
                logging.info(f"LLM retrying in {wait}s...")
                _time.sleep(wait)

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
