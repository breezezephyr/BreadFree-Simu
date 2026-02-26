# AGENTS.md

## Cursor Cloud specific instructions

### Project overview

BreadFree is a Python-based quantitative trading research platform for the Chinese A-share market. It uses `uv` for dependency management and Python 3.13 (specified in `.python-version`).

### Running the application

Backtest entry point is `main.py`. Strategies are selected via `--strategy` flag. See `README.md` section 3 for full CLI usage.

```bash
uv run python main.py --strategy RotationStrategy --lookback_period 20 --hold_period 20 --top_n 3
uv run python main.py --strategy BenchmarkStrategy
```

### Key caveats

- **No linting/formatting config**: The project has no ruff, flake8, pylint, mypy, or black configuration. Use `python -m py_compile <file>` for basic syntax validation.
- **No formal test suite**: There is no pytest or unittest infrastructure. The two `test_*.py` files in the repo root (`test_nvidia_api.py`, `test_multi_agent_models.py`) require LLM API keys (`NVIDIA_API_KEY` or `ARK_API_KEY` in `.env`) and are not runnable without them.
- **Data fetching requires internet**: The first backtest run fetches market data from East Money / AkShare APIs over HTTP. Subsequent runs use CSV cache files in `breadfree/data/cache/`.
- **SQLite databases are file-based**: `breadfree.db` (market data) and `live_trading.db` (live trading) are auto-created. No external database server needed.
- **LLM strategies (AgentStrategy, EffiA) require API keys**: Configure in a `.env` file at project root. See `docs/nvidia_api_setup.md` or `docs/volcano_ark_setup.md`.
- **`uv` must be on PATH**: Installed to `~/.local/bin`. Ensure `export PATH="$HOME/.local/bin:$PATH"` is in your shell profile.
- **The Tsinghua PyPI mirror** is configured in `pyproject.toml` under `[tool.uv]` as the default index. This accelerates downloads from China but works globally.
