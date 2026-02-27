# AGENTS.md

## Cursor Cloud specific instructions

### Project overview

BreadFree is a Python-based quantitative trading research platform for the Chinese A-share market. It uses `uv` for dependency management and Python 3.13 (specified in `.python-version`).

The stock/ETF pool (25 symbols: 20 ETFs + 5 individual stocks) is configured in `breadfree/config.yaml` under `etf_pool`. Chinese names are auto-loaded from there for all logs and LLM prompts.

### Running the application

Backtest entry point is `main.py`. See `README.md` section 3 for full CLI usage.

```bash
# Pure quant strategies (no LLM needed)
uv run python main.py --strategy RotationStrategy --lookback_period 20 --hold_period 20 --top_n 3
uv run python main.py --strategy BenchmarkStrategy

# LLM strategies (require ARK_API_KEY + ARK_MODEL secrets)
uv run python main.py --strategy AgentStrategyV2 --lookback_period 20 --hold_period 20 --top_n 3
uv run python main.py --strategy EffiA --lookback_period 20 --hold_period 20
```

Available strategies: `RotationStrategy`, `BenchmarkStrategy`, `DoubleMAStrategy`, `TripleMomentumStrategy`, `AgentStrategyV2`, `EffiA`.

### Required secrets for LLM strategies

| Secret | Purpose |
|---|---|
| `ARK_API_KEY` | Volcano Ark API key |
| `ARK_MODEL` | Volcano Ark endpoint ID (e.g. `ep-xxxxx`) |
| `LLM_PROVIDER` | Set to `volcano` (default) |

Without these, `AgentStrategyV2` and `EffiA` will fail with 404. Pure quant strategies (`RotationStrategy`, `BenchmarkStrategy`, etc.) work without any secrets.

### Key caveats

- **No linting/formatting config**: Use `python -m py_compile <file>` for basic syntax validation.
- **No formal test suite**: The `test_*.py` files in repo root require LLM API keys.
- **Data fetching requires internet**: First run fetches from East Money / AkShare APIs. Subsequent runs use CSV cache in `breadfree/data/cache/`.
- **25-symbol pool includes individual stocks**: Some symbols (600276, 600900, 600938, 000333, 300124) are stocks not ETFs. The `DataFetcher` handles both via the same East Money K-line API.
- **Market intel data**: `breadfree/data/market_intel.py` fetches fund flow / northbound capital data from AkShare. First call is slow (~10s); results are cached in `breadfree/data/cache/intel/` for 12 hours.
- **LLM strategies are slow**: `AgentStrategyV2` makes 2-3 LLM calls per rebalance day (~30-50s per rebalance). For a full-year backtest with `hold_period=20`, expect ~12 rebalances = ~10 minutes total.
- **SQLite databases are file-based**: `breadfree.db` and `live_trading.db` are auto-created. No external database server needed.
- **`uv` must be on PATH**: Installed to `~/.local/bin`.
