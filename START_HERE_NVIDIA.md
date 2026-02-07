# 🚀 开始使用 NVIDIA API - 完整指南

## 🎯 您现在拥有的功能

项目已成功集成 **NVIDIA NIM API**，支持免费使用强大的 LLM 模型来驱动您的量化交易策略！

---

## ⚡ 三步快速开始

### 步骤 1️⃣：获取 NVIDIA API Key（2分钟）

1. 访问：https://build.nvidia.com/
2. 点击右上角 **Sign in**（支持 Google/GitHub 登录）
3. 进入：https://build.nvidia.com/settings/api-keys
4. 点击 **Generate API Key**
5. 复制 API Key（格式：`nvapi-xxxxxxxxxxxxx`）

### 步骤 2️⃣：配置项目（1分钟）

打开项目根目录的 `.env` 文件，编辑：

```bash
LLM_PROVIDER=nvidia
LLM_API_KEY=nvapi-这里粘贴你的密钥
```

### 步骤 3️⃣：测试和运行（1分钟）

```bash
# 测试连接
python test_nvidia_api.py

# 如果看到 "Success!" 就可以运行 AI 策略了
python main.py --strategy AgentStrategy
```

---

## 📚 完整文档导航

### 🌟 必读文档

| 文档 | 说明 | 适合人群 |
|------|------|---------|
| [NVIDIA_API_快速配置.md](NVIDIA_API_快速配置.md) | **最简洁的中文指南** | 🔥 推荐首读 |
| [docs/nvidia_api_setup.md](docs/nvidia_api_setup.md) | 详细的配置说明和模型列表 | 需要深入了解 |
| [docs/llm_provider_comparison.md](docs/llm_provider_comparison.md) | NVIDIA vs 腾讯混元对比 | 选择提供商 |
| [CHANGELOG_NVIDIA_API.md](CHANGELOG_NVIDIA_API.md) | 技术更新日志 | 开发者 |

### 🔧 实用工具

| 工具 | 用途 |
|------|------|
| `test_nvidia_api.py` | 测试 API 连接是否正常 |
| `main.py --strategy AgentStrategy` | 运行 LLM 驱动的交易策略 |

---

## 🤖 可用的 AI 模型

项目默认使用 `meta/llama-3.1-8b-instruct`，快速且高效。

### 其他推荐模型

浏览完整列表：https://build.nvidia.com/models

**推荐试试：**
- `nvidia/llama-3.1-nemotron-70b-instruct` - 更强大
- `deepseek/deepseek-r1` - 专注推理
- `google/gemma-2-9b-it` - Google 出品
- `microsoft/phi-3-medium-128k-instruct` - 超长上下文

**切换模型：** 编辑 `breadfree/utils/llm_client.py` 第 20 行

---

## ✅ 验证清单

在开始之前，确保：

- [x] ✅ Python 依赖已安装（`pip install -e .`）
- [x] ✅ 已获取 NVIDIA API Key
- [x] ✅ 已配置 `.env` 文件
- [ ] ⏳ 运行 `test_nvidia_api.py` 测试成功
- [ ] ⏳ 成功运行 `python main.py --strategy AgentStrategy`

---

## 🎯 快速命令参考

```bash
# 1. 测试 API 连接
python test_nvidia_api.py

# 2. 运行 LLM 策略
python main.py --strategy AgentStrategy

# 3. 运行传统策略（不需要 API）
python main.py --strategy RotationStrategy --lookback_period 20 --hold_period 20 --top_n 3

# 4. 查看日志
type logs\breadfree.log
# 或 Linux/Mac: tail -f logs/breadfree.log

# 5. 查看回测结果
start output\backtest_result.html
# 或直接在浏览器中打开该文件
```

---

## 💡 重要提示

### ✅ 优势
- **完全免费**：NVIDIA 提供免费的云端推理
- **无需信用卡**：注册即用
- **模型丰富**：访问最新的开源 LLM
- **快速响应**：1-3秒内获得结果

### ⚠️ 注意事项
- 需要国际网络访问
- 免费版有速率限制（通常够用）
- 不要在 prompt 中包含敏感信息

---

## 🆚 NVIDIA vs 腾讯混元

| 特性 | NVIDIA | 腾讯混元 |
|------|--------|---------|
| 价格 | 免费 ✅ | 付费 💰 |
| 注册 | 简单 ⭐ | 需要手机号 ⭐⭐ |
| 网络 | 国际 🌍 | 国内友好 🇨🇳 |
| 模型 | 丰富 🎯 | 中等 |

**建议：** 新手从 NVIDIA 开始！

---

## ❓ 遇到问题？

### 常见问题

**Q: 测试失败，提示 API Key 错误**
```bash
# 检查：
1. API Key 是否正确复制（无多余空格）
2. .env 文件是否保存
3. 格式是否正确：LLM_API_KEY=nvapi-xxxxx
```

**Q: 提示网络连接错误**
```bash
# 确保可以访问国际网络
# 测试：ping integrate.api.nvidia.com
```

**Q: 想用腾讯混元怎么办？**
```bash
# 编辑 .env：
LLM_PROVIDER=hunyuan
LLM_API_KEY=sk-你的混元密钥
```

### 获取帮助

1. 查看日志：`logs/breadfree.log`
2. 阅读文档：[docs/nvidia_api_setup.md](docs/nvidia_api_setup.md)
3. NVIDIA 官方文档：https://docs.api.nvidia.com/

---

## 🎓 学习资源

- **NVIDIA Build 平台**：https://build.nvidia.com/
- **API 文档**：https://docs.api.nvidia.com/
- **模型浏览器**：https://build.nvidia.com/models
- **快速入门**：https://docs.api.nvidia.com/nim/docs/api-quickstart

---

## 🎉 完成！

配置完成后，您就可以：

✨ 使用 AI 驱动的量化交易策略  
✨ 让 LLM 分析市场趋势  
✨ 构建智能投资组合  
✨ 零成本使用最新的开源模型  

**祝您交易顺利！** 📈

---

## 📞 需要帮助？

- 🐛 发现 Bug？查看项目 Issues
- 💬 使用问题？阅读文档
- 🚀 功能建议？欢迎讨论

---

*最后更新：2026-02-07*
*项目版本：v0.1.0 with NVIDIA NIM Support*
