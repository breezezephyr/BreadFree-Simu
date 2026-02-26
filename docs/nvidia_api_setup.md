# NVIDIA NIM API 配置指南

本项目已支持使用 NVIDIA NIM API 作为 LLM 提供商，用于驱动 AgentStrategy（智能交易代理）。

## 🚀 快速开始

### 1. 获取 NVIDIA API Key

1. 访问 [NVIDIA Build 平台](https://build.nvidia.com/)
2. 点击右上角 **Sign in** 登录（或创建免费账户）
3. 进入 [API Keys 页面](https://build.nvidia.com/settings/api-keys)
4. 点击 **Generate API Key** 按钮
5. 复制生成的 API Key（格式类似：`nvapi-xxxxxxxxxxxxxxxxxxxxxx`）

### 2. 配置项目

在项目根目录创建 `.env`（可复制 `.env.example`），配置 API Key 与模型 ID（真实值仅放在 .env，勿提交到 git）：

```bash
# 设置 LLM 提供商为 NVIDIA
LLM_PROVIDER=nvidia

# 填入您的 NVIDIA API Key（与 config 中 env_key 一致时也可用 NVIDIA_API_KEY）
NVIDIA_API_KEY=nvapi-xxxxxxxxxxxxxxxxxxxxxx

# 模型 ID，会覆盖 config 中的占位符
NVIDIA_MODEL=meta/llama-3.1-8b-instruct
```

### 3. 运行 LLM 驱动的交易策略

```bash
python main.py --strategy AgentStrategy
```

## 📚 可用模型

NVIDIA NIM 提供多种开源模型，您可以在 [Models 页面](https://build.nvidia.com/models) 浏览：

### 推荐模型

| 模型名称 | 说明 | 适用场景 |
|---------|------|---------|
| `meta/llama-3.1-8b-instruct` | 默认模型，快速高效 | 日常交易决策 |
| `nvidia/llama-3.1-nemotron-70b-instruct` | 更强大的推理能力 | 复杂市场分析 |
| `deepseek/deepseek-r1` | 专注推理的模型 | 深度策略分析 |
| `google/gemma-2-9b-it` | Google 的轻量级模型 | 快速响应场景 |
| `microsoft/phi-3-medium-128k-instruct` | 超长上下文 | 大量历史数据分析 |

### 自定义模型

要使用其他模型，请在 `.env` 中设置 `NVIDIA_MODEL=你的模型ID`（如 `nvidia/llama-3.1-nemotron-70b-instruct`），会覆盖 config 中的占位符。勿把真实模型名写入 config.yaml 提交到 git。详见 `.env.example`。

## 🔧 高级配置

### 多提供商支持

项目同时支持 NVIDIA 和腾讯混元，可以通过 `LLM_PROVIDER` 环境变量切换：

**使用 NVIDIA:**
```bash
LLM_PROVIDER=nvidia
LLM_API_KEY=nvapi-xxxxx
```

**使用腾讯混元:**
```bash
LLM_PROVIDER=hunyuan
LLM_API_KEY=sk-xxxxx
```

### API 限制

NVIDIA NIM 免费层级提供：
- ✅ 免费云托管推理
- ✅ 访问所有开源模型
- ⚠️ 有请求速率限制（具体限制请查看 [NVIDIA 文档](https://docs.api.nvidia.com/)）

## 🌐 相关链接

- [NVIDIA Build 主页](https://build.nvidia.com/)
- [API Keys 管理](https://build.nvidia.com/settings/api-keys)
- [模型浏览](https://build.nvidia.com/models)
- [API 文档](https://docs.api.nvidia.com/)
- [快速入门指南](https://docs.api.nvidia.com/nim/docs/api-quickstart)

## ❓ 常见问题

### Q: API Key 免费吗？
A: 是的，NVIDIA NIM 提供免费的云托管推理服务用于开发和原型设计。

### Q: 支持哪些模型？
A: 支持 Meta Llama、Google Gemma、Microsoft Phi、DeepSeek 等众多开源模型。

### Q: 如何切换模型？
A: 修改 `breadfree/utils/llm_client.py` 中的 `default_model` 配置，或在调用时传入 `model` 参数。

### Q: 遇到 API 错误怎么办？
A: 
1. 确认 API Key 正确配置
2. 检查网络连接
3. 查看日志文件 `logs/` 目录
4. 参考 [NVIDIA API 文档](https://docs.api.nvidia.com/)

## 📝 示例代码

查看项目中的 AgentStrategy 实现：
- `breadfree/strategies/agent_strategy.py` - 智能交易代理策略
- `breadfree/utils/llm_client.py` - LLM 客户端封装

## 🎯 下一步

配置完成后，尝试运行智能交易代理：

```bash
# 使用默认配置运行
python main.py --strategy AgentStrategy

# 查看详细日志
tail -f logs/breadfree.log
```
