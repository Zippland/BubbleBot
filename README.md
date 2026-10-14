<div align="center">
  <h1>🫧 BubbleBot</h1>
  <p><strong>聊天就是入口，工作不止于聊天。</strong></p>
  <p>可自托管的个人 AI 助手 · 多渠道对话 · 独立工作空间 · Skills & MCP</p>
  <p>
    <a href="pyproject.toml"><img src="https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white" alt="Python 3.11+"></a>
    <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-22A06B" alt="License: MIT"></a>
  </p>
  <p>
    <a href="#快速开始">快速开始</a> ·
    <a href="docs/channels.md">接入聊天平台</a> ·
    <a href="docs/configuration.md">配置与扩展</a> ·
    <a href="https://github.com/Zippland/BubbleBot/issues">反馈问题</a>
  </p>
</div>

BubbleBot 把 AI 助手接进微信、飞书、Telegram 等聊天工具，也可以直接在终端里使用。
你用自然语言提出需求，它可以搜索资料、读写文件、执行命令、创建提醒，把结果发回聊天窗口。

每个 session 都有自己的工作目录、对话历史、长期记忆和技能。工作可以继续，偏好可以保留，不必每次从头交代。

## 你可以用它做什么

下面是可以交给助手的请求示例，而不是预设工作流；执行方式由模型结合工具和当前上下文决定。

> **资料整理**：「读一下这几个链接，比较它们的方案，把结论保存成 Markdown。」
>
> **项目协作**：「检查工作目录里的脚本，解释报错原因，改完后跑一下测试。」
>
> **日常提醒**：「明天上午九点提醒我交材料，发到这个聊天里。」
>
> **后台研究**：「另开一个子任务调研这个问题，完成后把结论告诉我。」

搜索需要配置搜索服务，定时任务和后台子任务需要进程持续运行；可执行的具体工作取决于模型、工具权限和已安装的依赖。

## 为什么是 BubbleBot

| 能力 | 使用方式 |
| --- | --- |
| **在聊天里办事** | 从微信、飞书等入口发起请求，接收文字、文件和图片；终端同样可用。 |
| **按项目组织上下文** | 用 `/session work` 创建或切换工作空间；不同入口也可以绑定到同一个 session。 |
| **留下有用的记忆** | 用 `SOUL.md` 定义沟通方式，用 `MEMORY.md` 保留长期信息；长对话支持自动压缩。 |
| **从回答到执行** | 内置文件、Shell、网页、定时任务和后台子任务工具；生图后端可按需开启。 |
| **用自己的模型** | 接入多家模型服务，或通过 OpenAI 兼容 API 使用自托管模型。 |
| **把能力继续接进来** | 用 Skills 固化工作方法，用 MCP 连接外部工具，无需改写核心循环。 |

配置、会话和工作文件保存在自己的机器上。使用云模型或聊天平台时，相应内容仍会发送给你配置的服务。

## 快速开始

先准备 **Python 3.11+、Git、[uv](https://docs.astral.sh/uv/getting-started/installation/)**，以及一个支持工具调用的模型 API。
可以先在终端完成一次对话，再接聊天平台。

### 1. 安装并初始化

```bash
git clone https://github.com/Zippland/BubbleBot.git
cd BubbleBot
uv sync --locked
uv run bubbles onboard
```

项目名称是 BubbleBot，命令行入口是 `bubbles`。初始化会生成 `~/.bubbles/config.json`；Windows 上位于 `%USERPROFILE%\.bubbles\config.json`。

### 2. 配置模型

编辑生成的配置文件。下面以 **OpenAI 兼容 API** 为例，把这些字段合并进去，保留其他设置：

```json
{
  "agents": {
    "defaults": {
      "provider": "custom",
      "model": "YOUR_MODEL_ID"
    }
  },
  "providers": {
    "custom": {
      "apiBase": "https://api.example.com/v1",
      "apiKey": "YOUR_API_KEY"
    }
  }
}
```

将 `apiBase` 换成服务商给出的 API 基础地址，`model` 换成该服务的模型 ID，`apiKey` 换成自己的密钥。不要把示例值原样用于真实请求。
支持 Anthropic、OpenAI、DeepSeek、智谱、通义、Gemini、OpenRouter 等 provider，详见[配置与扩展](docs/configuration.md)。

### 3. 开始对话

```bash
uv run bubbles agent -s personal
```

试着说：「在工作目录里写一份 `hello.md`，介绍你能帮我做什么。」
生成的文件位于 `~/.bubbles/sessions/personal/` 下；检查文件即可确认工具确实执行了。

也可以用于脚本中的单次调用：

```bash
uv run bubbles agent -s personal -m "读取 hello.md，概括其中的内容"
```

### 4. 接入聊天平台

按[渠道接入指南](docs/channels.md)配置凭据和访问白名单，然后启动网关：

```bash
uv run bubbles gateway
```

在对应私聊中发送 `/session personal` 绑定工作空间；群聊中先 @ 机器人再发送命令。
网关运行期间，已启用的渠道和定时任务会一起工作。

## 聊天平台

| 平台 | 接入方式 |
| --- | --- |
| 微信 | Windows 微信客户端 + WCFerry；见 [Windows 微信部署](docs/windows-wechat.md)。 |
| 飞书 | 应用机器人，WebSocket 长连接，无需配置公网回调地址。 |
| Telegram | Bot Token，可配置代理。 |
| Discord | Bot Token，通过 Gateway 连接。 |
| Slack | Bot Token + App Token，Socket Mode。 |
| 钉钉 | Client ID + Client Secret，Stream 模式。 |
| QQ | 官方机器人 App ID + Secret。 |
| WhatsApp | Node.js 网桥 + 扫码登录。 |
| Email | IMAP 收信 + SMTP 发信。 |
| Matrix | Access Token，需安装 `matrix` 可选依赖。 |
| Mochat | Token 与会话配置。 |

渠道配置、群聊触发方式和依赖各不相同；接入前请阅读[渠道指南](docs/channels.md)并设置白名单。

## 一个 session，一份工作空间

```text
~/.bubbles/sessions/personal/
├── SOUL.md          # 人设、语气与协作方式
├── MEMORY.md        # 长期记忆
├── session.jsonl    # 对话历史
├── config.json      # 会话级配置
├── skills/          # 这个会话的技能
└── data/            # 临时附件与工作产物
```

用 `/session writing` 开始另一项工作，用 `/config` 查看当前会话配置。
需要跨渠道共享上下文时，将这些聊天绑定到同一个 session；需要分开时，使用不同名称。

**`data/` 是临时目录**：超过 3 天未修改的文件会被清理。需要长期保存的产物请及时导出或放在 session 下其他目录，长期信息写入 `MEMORY.md`。

## 扩展你的助手

- **Skills**：在 `skills/<name>/SKILL.md` 中写清适用场景和操作步骤。内置海报设计、摘要、tmux 和文件分享流程，见[技能说明](bubbles/templates/session/skills/README.md)。
- **MCP**：通过 `tools.mcpServers` 接入 stdio 或 Streamable HTTP 工具服务。
- **定时任务**：在聊天中描述提醒，或用 `uv run bubbles cron --help` 查看命令。创建后可用 `uv run bubbles cron list --all` 核对。
- **模型与工具**：配置搜索、图片生成和执行环境，详见[配置与扩展](docs/configuration.md)。

想了解模型如何使用工具、切换消息目标或结束一轮对话，见 [Agent 与消息协议](docs/agent-protocol.md)。

## 部署与安全

普通渠道可在自己的电脑或服务器上运行；微信渠道需要兼容 WCFerry 的 Windows 微信环境。
仓库提供 [Dockerfile](Dockerfile)、[Compose 配置](docker-compose.yml)和 [Windows 常驻运行脚本](docs/windows-wechat.md)。

连接真实聊天前，请确认：

- **限制访问者**：配置渠道白名单；不要把空白名单当成“拒绝所有人”。
- **控制宿主机权限**：助手能执行 Shell。路径检查和 `local_isolated` 凭证环境不等于操作系统级隔离；建议使用低权限专用账号，接入不可信用户时另做容器或 OS 隔离。
- **保护凭据和数据**：不要提交配置、密钥或聊天记录；第三方模型、MCP、分享服务各有自己的数据边界。

完整说明见 [SECURITY.md](SECURITY.md)。

## 开发与贡献

```bash
uv sync --locked --extra dev
uv run pytest
```

欢迎提交问题、修复和新的集成。报告问题时请附上操作系统、复现步骤和脱敏日志，不要附 API key 或私人聊天内容。
涉及安全漏洞，请按 [SECURITY.md](SECURITY.md) 中的方式报告。

| 想了解什么 | 从这里开始 |
| --- | --- |
| 产品行为与约定 | [SPEC.md](SPEC.md) |
| 配置字段与默认值 | [配置指南](docs/configuration.md) · [配置 schema](bubbles/config/schema.py) |
| Agent 循环与工具 | [消息协议](docs/agent-protocol.md) · [Agent 源码](bubbles/agent/) |
| 新增模型或渠道 | [Providers](bubbles/providers/) · [Channels](bubbles/channels/) |

## 致谢与许可证

BubbleBot 基于 [nanobot](https://github.com/HKUDS/nanobot) 发展，感谢上游项目及开源社区。

项目代码采用 [MIT License](LICENSE)。附带的第三方技能和字体遵循各自许可证，见[第三方材料说明](THIRD_PARTY_NOTICES.md)。
