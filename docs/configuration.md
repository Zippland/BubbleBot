# 配置与扩展

[返回 README](../README.md) · [渠道接入](channels.md)

全局配置位于 `~/.bubbles/config.json`。下面的 JSON 都是配置片段，请合并到对应字段，不要覆盖已有凭据和渠道设置。支持 camelCase 和 snake_case；完整字段以[配置 schema](../bubbles/config/schema.py)为准。

## 模型服务

`agents.defaults.provider` 选择服务商，`agents.defaults.model` 选择该服务提供的模型。首次配置建议显式指定 provider，避免自动识别带来的歧义。

| 接入方式 | provider 字段 | 凭据位置 |
| --- | --- | --- |
| OpenAI 兼容 API、自托管服务 | `custom` | `providers.custom` |
| Anthropic / OpenAI / Gemini | `anthropic` / `openai` / `gemini` | 同名 `providers` 项 |
| DeepSeek / 智谱 / 通义 | `deepseek` / `zhipu` / `dashscope` | 同名 `providers` 项 |
| Moonshot / MiniMax / Groq | `moonshot` / `minimax` / `groq` | 同名 `providers` 项 |
| OpenRouter / SiliconFlow / AiHubMix / VolcEngine | `openrouter` / `siliconflow` / `aihubmix` / `volcengine` | 同名 `providers` 项 |

其余 provider 与登录方式见[注册表](../bubbles/providers/registry.py)和 `uv run bubbles provider --help`。

### 使用服务商适配器

例如使用智谱，将模型 ID 和密钥换成自己账户可用的值：

```json
{
  "agents": {
    "defaults": {
      "provider": "zhipu",
      "model": "YOUR_MODEL_ID"
    }
  },
  "providers": {
    "zhipu": {
      "apiKey": "YOUR_API_KEY"
    }
  }
}
```

智谱适配器默认请求 BigModel 中国区接口。国际区或代理服务可设置对应 provider 的 `apiBase`；不要把不同区域的 key 与 endpoint 混用。

### 使用兼容端点

README 中的 `custom` 直接使用服务提供的 OpenAI 兼容 Chat Completions 接口。`apiBase` 是基础地址，而不是完整的 `/chat/completions` 路径。

自托管服务也用同样的配置：填写自己的服务地址和实际加载的模型 ID。若服务不校验 API key，可用非空占位值 `no-key`。模型需要支持工具调用；图片理解还需要模型和适配器都支持视觉输入。

### 上下文与请求参数

| 字段 | 用途 |
| --- | --- |
| `agents.defaults.contextLimit` | 按所用模型设置上下文上限，用于历史压缩判断。 |
| `agents.defaults.maxTokens` | 单次模型响应的输出上限。 |
| `agents.defaults.maxToolIterations` | 每轮工具循环次数上限。 |
| `agents.defaults.apiTimeout` | 单次模型请求超时，单位秒。 |
| `agents.defaults.maxApiRetries` | 可重试请求失败后的重试次数。 |

这些字段的默认值不是对所有模型都适用的能力声明；请按供应商文档调整。API 调用与账号额度由所选服务提供，不包含在本项目中。

## 会话配置与记忆

在聊天中使用：

```text
/session work
/config
/config model YOUR_MODEL_ID
/config sandbox local_isolated
/compact
```

- `/session work` 创建或绑定一个命名工作空间。不同入口使用相同名称，会共享该 session 的历史、记忆和配置。
- `/config` 查看会话配置；`model`、`system_prompt`、`sandbox` 可以按会话覆盖。若全局固定了 provider，更换 model 仍使用该 provider。
- `SOUL.md` 用于描述协作方式，`MEMORY.md` 用于长期信息。可以自己编辑，也可以在聊天中明确请助手写入，随后检查文件。
- `/compact` 压缩当前上下文，不等于删除完整历史。

`local_isolated` 为会话提供独立的 HOME/CLI 凭证环境，**不是文件系统或网络的强隔离**。安全边界见 [SECURITY.md](../SECURITY.md)。

## 搜索与图片生成

Web 搜索使用 Tavily，需要配置搜索 key：

```json
{
  "tools": {
    "web": {
      "search": {
        "apiKey": "YOUR_TAVILY_API_KEY"
      }
    }
  }
}
```

网页正文抓取与搜索是不同工具。生成图片则需要启用 `tools.imageGeneration.enabled`，配置该后端使用的 provider key，并选择后端支持的模型；字段见[配置 schema](../bubbles/config/schema.py)，后端实现见 [image_generation](../bubbles/image_generation/)。生成后得到本地文件，再由消息工具发送。

## 接入 MCP 工具

支持 stdio 与 Streamable HTTP。以自己管理的 HTTP 服务为例：

```json
{
  "tools": {
    "mcpServers": {
      "my-tools": {
        "url": "https://mcp.example.com/mcp",
        "headers": {
          "Authorization": "Bearer YOUR_MCP_TOKEN"
        },
        "toolTimeout": 30
      }
    }
  }
}
```

替换示例地址和令牌；stdio 服务则使用 `command`、`args` 和可选 `env`。添加服务意味着允许助手调用其工具，请先确认权限与数据去向。

## 添加 Skills

将带有 `SKILL.md` 的目录放入 `~/.bubbles/sessions/<session>/skills/`。技能描述进入索引，详细说明按需读取。

格式、内置清单及模板同步方式见[技能说明](../bubbles/templates/session/skills/README.md)。Skills 描述工作方法，不会自动安装所有外部程序，也不会替代模型或服务权限。

## 常用检查

```bash
uv run bubbles status
uv run bubbles channels status
uv run bubbles cron list --all
uv run bubbles agent --help
```

全局配置修改后重启相应进程。已在运行网关时，不要再启动第二个使用同一份数据的网关，尤其不要重复连接微信。
