"""页面共用工具"""
import os, yaml
from typing import Dict
import streamlit as st

@st.cache_data(ttl=3600)
def load_config() -> dict:
    config_path = os.path.join(os.path.dirname(__file__), "..", "..", "config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def get_pool() -> Dict[str, str]:
    return load_config().get("etf_pool", {})

def sym_name(code: str) -> str:
    pool = get_pool()
    name = pool.get(code, "")
    return f"{code}-{name}" if name else code
