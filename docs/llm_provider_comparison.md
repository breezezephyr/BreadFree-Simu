# LLM 提供商对比

本项目支持多个 LLM 提供商，您可以根据需求选择最适合的平台。

## 📊 提供商对比

| 特性 | NVIDIA NIM | 腾讯混元 |
|------|-----------|---------|
| **免费额度** | ✅ 免费云端推理 | ⚠️ 需要付费 |
| **注册难度** | ⭐ 简单（支持 Google/GitHub） | ⭐⭐ 需要中国手机号 |
| **网络要求** | 🌍 国际网络 | 🇨🇳 国内网络友好 |
| **模型选择** | 🎯 丰富（Llama, Gemma, DeepSeek等） | 🎯 DeepSeek 系列 |
| **API 兼容性** | ✅ OpenAI 兼容 | ✅ OpenAI 兼容 |
| **文档质量** | ⭐⭐⭐ 完善的英文文档 | ⭐⭐ 中文文档 |
| **响应速度** | ⚡ 快速 | ⚡ 快速 |
| **推荐场景** | 国际用户、开发测试 | 中国用户、生产环境 |

## 🚀 NVIDIA NIM（推荐）

### 优势
- ✅ **完全免费**：无需信用卡，注册即用
- ✅ **模型丰富**：Meta Llama、Google Gemma、DeepSeek、Microsoft Phi 等
- ✅ **快速注册**：支持 Google、GitHub 账号登录
- ✅ **文档完善**：详细的 API 文档和示例
- ✅ **社区活跃**：开源生态支持

### 劣势
- ⚠️ 需要国际网络访问
- ⚠️ 免费版有速率限制（通常足够开发使用）

### 适用场景
- 快速原型开发
- 学习和实验
- 国际化项目
- 预算有限的个人开发者

### 配置方法
```bash
LLM_PROVIDER=nvidia
LLM_API_KEY=nvapi-xxxxxxxxxxxxx
```

获取 API Key: https://build.nvidia.com/settings/api-keys

## 🇨🇳 腾讯混元

### 优势
- ✅ **国内访问**：无需特殊网络
- ✅ **中文友好**：中文文档和支持
- ✅ **稳定可靠**：腾讯云基础设施
- ✅ **企业级**：适合生产环境

### 劣势
- ⚠️ 需要付费（按调用量计费）
- ⚠️ 需要实名认证
- ⚠️ 模型选择相对较少

### 适用场景
- 生产环境部署
- 需要中文优化的场景
- 企业级应用
- 国内网络环境

### 配置方法
```bash
LLM_PROVIDER=hunyuan
LLM_API_KEY=sk-xxxxxxxxxxxxx
```

## 🎯 如何选择？

### 选择 NVIDIA NIM 如果：
- 👨‍💻 你是个人开发者或学生
- 🔬 用于学习、实验或原型开发
- 🌍 有稳定的国际网络
- 💰 预算有限或想零成本开始
- 🚀 想快速开始，无需复杂注册

### 选择腾讯混元如果：
- 🏢 用于企业生产环境
- 🇨🇳 主要在中国大陆使用
- 💼 有预算支持
- 🔐 需要更高的 SLA 保证
- 📝 需要中文技术支持

## 🔄 切换提供商

项目支持在任何时候切换 LLM 提供商，只需修改 `.env` 文件：

```bash
# 切换到 NVIDIA
LLM_PROVIDER=nvidia
LLM_API_KEY=nvapi-xxxxxxxxxxxxx

# 或切换到混元
LLM_PROVIDER=hunyuan
LLM_API_KEY=sk-xxxxxxxxxxxxx
```

无需修改代码，切换后重新运行即可。

## 📈 性能对比（参考）

| 指标 | NVIDIA Llama-3.1-8B | 腾讯混元 DeepSeek-V3 |
|------|-------------------|---------------------|
| 响应速度 | ~1-3秒 | ~1-3秒 |
| 上下文长度 | 128K tokens | 64K tokens |
| 推理能力 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| 金融领域 | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| 成本 | 免费 | 按量付费 |

*注：性能因模型和任务而异，以上仅供参考*

## 🌟 推荐配置

### 开发环境
```bash
LLM_PROVIDER=nvidia
LLM_API_KEY=nvapi-xxxxxxxxxxxxx
```

### 生产环境（中国）
```bash
LLM_PROVIDER=hunyuan
LLM_API_KEY=sk-xxxxxxxxxxxxx
```

## 📚 相关链接

### NVIDIA NIM
- 官网: https://build.nvidia.com/
- API Keys: https://build.nvidia.com/settings/api-keys
- 文档: https://docs.api.nvidia.com/
- 模型库: https://build.nvidia.com/models

### 腾讯混元
- 官网: https://cloud.tencent.com/product/hunyuan
- 控制台: https://console.cloud.tencent.com/hunyuan
- 文档: https://cloud.tencent.com/document/product/1729

## ❓ FAQ

**Q: 可以同时配置两个提供商吗？**  
A: 可以，通过修改 `LLM_PROVIDER` 环境变量即可切换。

**Q: NVIDIA 免费版够用吗？**  
A: 对于开发、学习和小规模回测完全够用。

**Q: 数据会被提供商存储吗？**  
A: 请参考各提供商的隐私政策。建议避免在 prompt 中包含敏感信息。

**Q: 可以添加其他提供商吗？**  
A: 可以！只需在 `breadfree/utils/llm_client.py` 的 `PROVIDER_CONFIGS` 中添加配置即可。

---

💡 **建议**：初学者推荐从 NVIDIA NIM 开始，快速上手零成本。如果需要部署到生产环境或在国内使用，再考虑切换到腾讯混元。
