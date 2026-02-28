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

### Strategy architecture (V2)

```
RotationStrategy (纯量化):  多因子效率分 → Top-N 选股 → 等权/风险平价加权
AgentStrategyV2  (LLM辩证): QuantEngine → Bull Analyst → Bear Challenger → PM 裁决
EffiA            (LLM轮动): QuantPrep → Analyst Agent → RiskMgr Agent
```

`RotationStrategy` 是核心量化策略, 多因子合成:
- `efficiency = (momentum / period_vol) * R²` — 风险调整后的趋势质量
- `accel = current_mom - lagged_mom` — 动量加速度 (趋势增强加分)
- `drawdown_penalty` — 远离近期高点的标的降权
- 支持 `retention_bonus` (降低换手摩擦)、`drawdown_circuit_breaker` (回撤熔断)

### Key caveats

- **No linting/formatting config**: Use `python -m py_compile <file>` for basic syntax validation.
- **No formal test suite**: The `test_*.py` files in repo root require LLM API keys.
- **Data fetching requires internet**: First run fetches from East Money / AkShare APIs. Subsequent runs use CSV cache in `breadfree/data/cache/`.
- **25-symbol pool includes individual stocks**: Some symbols (600276, 600900, 600938, 000333, 300124) are stocks not ETFs. The `DataFetcher` handles both via the same East Money K-line API.
- **Market intel data**: `breadfree/data/market_intel.py` fetches fund flow / northbound capital data from AkShare. First call is slow (~10s); results are cached in `breadfree/data/cache/intel/` for 12 hours.
- **LLM strategies are slow**: `AgentStrategyV2` makes 2-3 LLM calls per rebalance day (~30-50s per rebalance). For a full-year backtest with `hold_period=20`, expect ~12 rebalances = ~10 minutes total.
- **SQLite databases are file-based**: `breadfree.db` and `live_trading.db` are auto-created. No external database server needed.
- **`uv` must be on PATH**: Installed to `~/.local/bin`.
- **Deleted old strategy variants**: `double_ma_strategy.py`, `effi_agent_strategy_signal.py`, `effi_agent_strategy_hold20d.py` have been consolidated. `ma_strategy.py` is the canonical `DoubleMAStrategy`.
- **DB singleton preload cache**: `get_db_manager()` is a process-wide singleton. Calling `calc_top_n_scores()` preloads a narrow date window into `_mem_cache`; a subsequent `BacktestEngine.run()` in the same process may get stale data. Call `get_db_manager().clear_cache()` between independent data-loading flows.

### Daily email report

Sends Top-5 factor ranking + equity curve to configured recipients via QQ SMTP.

```bash
# Scheduled daemon (08:30 Asia/Shanghai daily)
uv run python daily_report_scheduler.py

# Immediate one-shot (testing)
uv run python daily_report_scheduler.py --now
```

| Secret | Purpose |
|---|---|
| `SMTP_USER` | QQ 邮箱地址 |
| `SMTP_PASSWORD` | QQ 邮箱授权码 (非登录密码) |
| `REPORT_RECIPIENTS` | 收件人, 逗号分隔多个 |

Config in `breadfree/config.yaml` under `daily_report` (schedule_time, top_n, lookback_period, backtest_days).
