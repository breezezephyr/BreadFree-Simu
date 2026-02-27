# NVIDIA NIM 模型性能与适用场景评估

基于 `.env` 中候选模型列表，从**首 token 时间(TTFT)**、**输出 TPS(tokens/s)**、**适用场景**三方面做选型参考。  
实测数据需在本机运行 `uv run python test_nvidia_models_perf.py` 获取（依赖账户与区域可用性）。

---

## 1. 指标说明

| 指标 | 含义 |
|------|------|
| **首 token 时间 (TTFT)** | 从发送请求到收到第一个输出 token 的耗时，影响“首字响应快慢”的体感。 |
| **输出 TPS** | 生成阶段每秒输出的 token 数，影响长回复的生成速度。 |
| **适用场景** | 按模型定位给出的推荐用途，供业务选型参考。 |

---

## 2. 模型评估表（含适用场景）

| 模型 ID | 可用性 | 首 Token(ms) | 输出 TPS | 适用场景 |
|---------|--------|--------------|----------|----------|
| `meta/llama-3.1-8b-instruct` | 实测可用 | ~700 | 较高 | 通用对话、轻量推理、低延迟、成本敏感 |
| `deepseek-ai/deepseek-v3.1` | 实测可用 | ~690 | ~80+ | 通用/推理、长上下文、中高负载、综合质量 |
| `deepseek-ai/deepseek-v3.2` | 视账户 | - | - | 推理增强、复杂推理与长链、强推理场景 |
| `minimaxai/minimax-m2.5` | 视账户 | - | - | 通用对话、多轮与多语言 |
| `qwen/qwen3-coder-480b-a35b-instruct` | 视账户 | - | - | 代码生成与理解、技术问答、开发辅助 |
| `qwen/qwen3-235b-a22b` | 视账户 | - | - | 通用强能力、复杂任务、多语言、高理解 |
| `qwen/qwen3-next-80b-a3b-thinking` | 视账户 | - | - | 深度推理、思考链、数学/逻辑/分析 |
| `qwen/qwen3.5-397b-a17b` | 视账户 | - | - | 超大参数量、综合能力、极高要求场景 |

- **可用性**：部分模型可能 404/410，以你账户在 [build.nvidia.com/models](https://build.nvidia.com/models) 的可见性为准。  
- **首 Token / TPS**：带 “~” 的为当前脚本实测典型值；未填的需在本机跑 `test_nvidia_models_perf.py` 补全。

---

## 3. 场景选型建议

- **低延迟、高 TPS、成本敏感**：优先试 `meta/llama-3.1-8b-instruct`、`deepseek-ai/deepseek-v3.1`。  
- **强推理与长链**：优先试 `deepseek-ai/deepseek-v3.2`、`qwen/qwen3-next-80b-a3b-thinking`。  
- **代码/技术**：优先试 `qwen/qwen3-coder-480b-a35b-instruct`。  
- **综合能力与多语言**：可试 `qwen/qwen3-235b-a22b`、`minimaxai/minimax-m2.5`。  
- **量化/交易类 Agent**：可结合延迟与推理能力，在 `deepseek-v3.1` / `v3.2` 与 `llama-3.1-8b` 之间做权衡。

---

## 4. 如何跑出你本机的报告

```bash
uv run python test_nvidia_models_perf.py
```

脚本会对上述 8 个模型做流式调用，输出每个模型的**可用性、TTFT、输出 TPS、适用场景**一行。  
跑完后把终端里的表格贴到本文或自行更新上表即可。
