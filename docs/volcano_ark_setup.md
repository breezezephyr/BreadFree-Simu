# 火山方舟 ARK API 配置（国内较快）

Agent 策略支持使用火山引擎方舟 ARK 作为 LLM，国内访问通常比 NVIDIA 更快。

## 配置步骤

1. 在 [火山方舟控制台](https://console.volcengine.com/ark) 创建模型端点，获取 **API Key** 和 **端点 ID**（形如 `ep-xxxxxxxx`）。

2. 在项目根目录创建 `.env`（可复制 `.env.example`），配置 **API Key** 和 **端点 ID**（真实值仅放在 .env，勿提交到 git）：

```bash
ARK_API_KEY=你的方舟API_Key
ARK_MODEL=ep-xxxxxxxx
```

`config.yaml` 中保持占位符 `YOUR_ARK_ENDPOINT_ID` 即可，运行时会用 `.env` 中的 `ARK_MODEL` 覆盖。

3. 在 `breadfree/config.yaml` 的 `llm` 中选用火山方舟（通常已默认）：

```yaml
llm:
  active: volcano
  providers:
    volcano:
      base_url: "https://ark.cn-beijing.volces.com/api/v3"
      model: "YOUR_ARK_ENDPOINT_ID"  # 由 .env 的 ARK_MODEL 覆盖
      env_key: ARK_API_KEY
```

4. 运行 Agent 策略：

```bash
uv run python main.py --strategy AgentStrategy
```

## 可选

- 使用其他端点：在 `.env` 中设置 `ARK_MODEL=ep-另一个端点ID`，勿把真实端点 ID 写进 config 提交。
- 改用 NVIDIA：在 `config.yaml` 中设置 `llm.active: nvidia`，并在 `.env` 中配置 `NVIDIA_API_KEY` 与 `NVIDIA_MODEL`。

## 支持的 LLM 提供商（在 config.yaml 的 llm.providers 中配置）

| llm.active | 说明 |
|------------|------|
| `nvidia` | NVIDIA NIM，需在 .env 设置 `NVIDIA_API_KEY` |
| `volcano` | 火山方舟 ARK（国内推荐），需在 .env 设置 `ARK_API_KEY` |

切换方式：在 `breadfree/config.yaml` 中修改 `llm.active` 为 `nvidia` 或 `volcano` 即可。
