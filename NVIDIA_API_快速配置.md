# 🚀 NVIDIA API 快速配置指南

## 第一步：获取 NVIDIA API Key（免费）

1. 打开 [NVIDIA Build 平台](https://build.nvidia.com/)
2. 点击右上角 **Sign in** 登录（支持 Google/GitHub 账号）
3. 进入 [API Keys 管理页面](https://build.nvidia.com/settings/api-keys)
4. 点击 **Generate API Key** 生成密钥
5. **复制 API Key**（格式：`nvapi-xxxxxxxxx`）

## 第二步：配置项目

编辑项目根目录的 `.env` 文件：

```bash
# 1. 设置提供商为 NVIDIA
LLM_PROVIDER=nvidia

# 2. 粘贴你的 API Key
LLM_API_KEY=nvapi-你的密钥这里
```

## 第三步：测试连接

运行测试脚本验证配置：

```bash
python test_nvidia_api.py
```

如果看到 ✅ Success! 说明配置成功！

## 第四步：使用 LLM 策略

运行 AI 驱动的交易策略：

```bash
python main.py --strategy AgentStrategy
```

## 🎯 可用的 NVIDIA 模型

项目默认使用 `meta/llama-3.1-8b-instruct`，您也可以选择：

| 模型 | 特点 |
|------|------|
| `meta/llama-3.1-8b-instruct` | 快速、高效（默认）|
| `nvidia/llama-3.1-nemotron-70b-instruct` | 更强大的推理能力 |
| `deepseek/deepseek-r1` | 专注推理和分析 |
| `google/gemma-2-9b-it` | Google 轻量级模型 |

浏览更多模型：https://build.nvidia.com/models

## 🔧 切换模型

在 `breadfree/config.yaml` 的 `llm.providers.nvidia.model` 中填写你的模型 ID，例如：

```yaml
llm:
  providers:
    nvidia:
      model: "你的模型ID"  # 如 meta/llama-3.1-8b-instruct
```

## ❓ 常见问题

**Q: API Key 是免费的吗？**  
A: 是的！NVIDIA 提供免费的云端推理服务。

**Q: 有请求限制吗？**  
A: 免费版有速率限制，通常足够开发和测试使用。

**Q: 如果不想用 NVIDIA，可以用其他的吗？**  
A: 可以！项目也支持腾讯混元，在 `.env` 中设置：
```bash
LLM_PROVIDER=hunyuan
LLM_API_KEY=your_hunyuan_key
```

**Q: 测试失败怎么办？**  
A: 检查：
1. API Key 是否正确复制（不要有多余空格）
2. 网络连接是否正常
3. 查看 `logs/breadfree.log` 获取详细错误信息

## 📚 详细文档

- 完整配置指南：[docs/nvidia_api_setup.md](docs/nvidia_api_setup.md)
- NVIDIA 官方文档：https://docs.api.nvidia.com/

---

**配置完成后，享受 AI 驱动的量化交易吧！** 🎉
