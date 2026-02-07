# BreadFree - Flexible A-Share Quant Research Platform

BreadFree is a modular, event-driven quantitative trading research platform designed for the A-share market. It separates the **execution engine** from **strategy logic**, allowing researchers to flexibly switch between classic algorithmic strategies, simple rotation models, and complex LLM-driven agent systems.

## 1. Project Architecture (核心架构)

The system is built around a central event loop in `main.py` -> `breadfree/engine/backtest_engine.py` that feeds data to pluggable strategy modules.

```mermaid
graph TD
    CLI["main.py CLI / config.yaml"] --> Engine[BacktestEngine]
    Data["DataFetcher (AkShare)"] -->|"Cache & Warmup"| Engine
    
    Engine -->|"On Bar (Daily)"| Strategy[Strategy Interface]
    
    subgraph Pluggable Strategies
        Strategy --> Rotation[RotationStrategy]
        Strategy --> Agent["AgentStrategy (LLM)"]
        Strategy --> DMA[DoubleMAStrategy]
        Strategy --> Bench[BenchmarkStrategy]
    end
    
    Rotation -->|"Signals"| Broker["Broker & Execution"]
    Agent -->|"Signals"| Broker
    DMA -->|"Signals"| Broker
    
    Broker -->|"Account State"| Engine
    Engine -->|"Logs & Metrics"| Output["Visualization & Reports"]
```

### Core Components used in `main.py`
- **Engine (`breadfree/engine`)**: Handles data feeding, time-stepping, and interaction between strategy and broker.
- **Strategies (`breadfree/strategies`)**: Contains logic implementations. All inherit from `BaseStrategy`.
- **Data (`breadfree/data`)**: Wraps AkShare for fetching and caching local CSVs to speed up backtests.
- **Utils**: Configuration, logging, and plotting (Pyecharts).

## 2. Supported Strategies (支持的策略)

You can switch strategies using the `--strategy` flag in `main.py`.

### A. RotationStrategy (Core Focus)
A momentum and efficiency-based rotation strategy for ETFs or stocks.
- **Logic**: Selects `top_n` assets based on recent returns (`lookback_period`) and holds them for `hold_period`.
- **Key Params**: `--lookback_period`, `--hold_period`, `--top_n`, `--use_efficiency`.
- **Use Case**: Sector rotation, ETF momentum.

### B. AgentStrategy (LLM Powered)
A multi-agent system simulating an investment committee using LangGraph.
- **Roles**: Analyst -> Risk Manager -> Fund Manager.
- **Features**: Reads news, reasons about market sentiment, and adheres to distinct personas.
- **Configuration**: Requires LLM API keys (supports NVIDIA NIM or Tencent Hunyuan).
- **Setup Guide**: See [docs/nvidia_api_setup.md](docs/nvidia_api_setup.md) for detailed configuration instructions.

### C. Classic Strategies
- **DoubleMAStrategy**: Golden cross/Dead cross logic with short and long windows.
- **BenchmarkStrategy**: Simple Buy-and-Hold on a target asset (e.g., HS300) to baseline performance.

## 3. Workflow & Usage (使用流程)

### 1. Installation
```bash
pip install uv
uv sync
source .venv/bin/activate
```

### 2. Configuration
**A. Global Settings**  
Configure global settings in `breadfree/config.yaml` (e.g., initial cash, default stock pool).

**B. LLM API Configuration (Optional - Only for AgentStrategy)**  
If you want to use the LLM-powered AgentStrategy:
1. Get a free API key from [NVIDIA Build](https://build.nvidia.com/settings/api-keys)
2. Configure in `.env` file:
   ```bash
   LLM_PROVIDER=nvidia
   LLM_API_KEY=nvapi-xxxxxxxxxxxxx
   ```
3. Test the connection: `python test_nvidia_api.py`
4. See detailed guide: [docs/nvidia_api_setup.md](docs/nvidia_api_setup.md)

### 3. Running Backtests
The entry point `main.py` unifies the execution.

**Run Rotation Strategy (Example):**
```bash
python main.py --strategy RotationStrategy \
    --lookback_period 20 --hold_period 20 --top_n 3 \
    --output_file output/rotation_result.png
```

**Run Benchmark (Buy & Hold):**
```bash
python main.py --strategy BenchmarkStrategy --output_file output/benchmark_result.png
```

**Run LLM Agent Strategy:**
```bash
python main.py --strategy AgentStrategy
```

### 4. Analysis
- **HTML Reports**: Interactive charts generated in the root directory (e.g., `backtest_result.html`).
- **Logs**: Detailed execution logs in `logs/`.
- **Grid Search**: Use `grid_search.sh` to optimize parameters for RotationStrategy.

## 4. Development Status

- **Completed**:
    - [x] Unified `BacktestEngine` supporting T+1 simulation.
    - [x] Modular Strategy Interface (`BaseStrategy`).
    - [x] Robust Data Fetching with caching (AkShare).
    - [x] Strategy Implementations: Rotation, MA, Agent (LLM).
    - [x] Visualization with Pyecharts.

- **Roadmap**:
    - [ ] Real-time data integration (Tick/Min bars).
    - [ ] Notification system (Push messages).
    - [ ] Vector DB for news-based historical analogy.
    - [ ] Web Dashboard.

## 5. Disclaimer
This project is for **simulation and research purposes only**. It does not constitute financial advice.