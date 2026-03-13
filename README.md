<div align="center">
  <h1>🫧 Bubbles</h1>
  <p>
    <img src="https://img.shields.io/badge/python-≥3.11-blue" alt="Python">
    <img src="https://img.shields.io/badge/license-MIT-green" alt="License">
  </p>
</div>

🫧 **Bubbles** is a lightweight personal AI assistant.

> ✉️ 致来访者
>
> 我一直在尝试做一个全面的的生态助手（个人助理），能够连接我使用的任何工具、日程表、数据库、资料库。并基于数据资料，实时地和我交流，以及帮我安排日程、提醒、规划时间和出行，安排我的任务计划。甚至于查资料、做研究、处理工作、回老板微信（帮我上班）。
>
> 之前做了 [LifeSync-AI](https://github.com/Zippland/LifeSync-AI) 这个项目，帮我进行每天的任务规划。但定时任务是被动的，而且缺少一个统一的数据处理中心，所以无法通过与用户交流进行实时的任务调度。后来又做了 [bubbles](https://github.com/Zippland/Bubbles) 这个项目，但是总归还是不够 AI —— 然后 OpenClaw 就横空出世了。所以我最近在复盘这玩意儿和 OpenCalw 的差距，为什么同一个项目，OpenClaw 做成了，我没做成：
> 
> 1. Timing 很重要，做太早了，会没有 sota 范式积累
> 2. 更多重构的勇气和决心，探索期的产物，两个月就需要全盘推翻
> 3. 已经足够好、功能足够多、边际效应低，我没有超过这个项目的痛点
> 4. Vibe coding 只需要用最好的模型，用次优解会导致代码难以维护
> 
> 然后诞生了 Bubblebot，本质上只是把 bubbles 的内核换成了 coding agent。
>
> 玩得开心，
>
> Zylan
>

## 📦 Install

```bash
cd BubbleBot
pip install -e .
```

## 🚀 Quick Start

**1. Initialize**

```bash
bubbles onboard
```

**2. Configure** (`~/.bubbles/config.json`)

```json
{
  "providers": {
    "openrouter": {
      "apiKey": "sk-or-v1-xxx"
  },
  "agents": {
    "defaults": {
      "model": "anthropic/claude-opus-4-5",
      "provider": "openrouter"
    }
  }
}
```

**3. CLI Chat**

```bash
bubbles agent
```

## 💬 Chat Apps

Connect Bubbles to your favorite chat platform.

| Channel | What you need |
|---------|---------------|
| **Telegram** | Bot token from @BotFather |
| **Discord** | Bot token + Message Content intent |
| **WhatsApp** | QR code scan |
| **Feishu** | App ID + App Secret |
| **DingTalk** | App Key + App Secret |
| **Slack** | Bot token + App-Level token |
| **Email** | IMAP/SMTP credentials |
| **QQ** | App ID + App Secret |
| **WeChat** | wcferry (**Windows only**) |

### Telegram Example

```json
{
  "channels": {
    "telegram": {
      "enabled": true,
      "token": "YOUR_BOT_TOKEN",
      "allowFrom": ["YOUR_USER_ID"]
    }
  }
}
```

```bash
bubbles gateway
```

### WeChat Example

```json
{
  "channels": {
    "wechat": {
      "enabled": true,
      "groups": ["group_id_1", "group_id_2"]
    }
  }
}
```

- **Private chats**: Direct reply
- **Group chats**: Only reply when @mentioned

## CLI Reference

### Core Commands

| Command | Description |
|---------|-------------|
| `bubbles onboard` | Initialize config & workspace |
| `bubbles agent` | Interactive chat mode |
| `bubbles agent -m "..." -s <session>` | Single message mode |
| `bubbles gateway` | Start the gateway (all channels) |
| `bubbles status` | Show status |
| `bubbles sync-skills` | Sync skills to all sessions |

### Channel Commands

| Command | Description |
|---------|-------------|
| `bubbles channels status` | Show channel status |
| `bubbles channels login` | Link WhatsApp via QR code |

### Cron Commands

| Command | Description |
|---------|-------------|
| `bubbles cron list` | List scheduled jobs |
| `bubbles cron add` | Add a scheduled job |
| `bubbles cron remove <id>` | Remove a job |
| `bubbles cron enable <id>` | Enable/disable a job |
| `bubbles cron run <id>` | Manually run a job |

### Provider Commands

| Command | Description |
|---------|-------------|
| `bubbles provider login <name>` | OAuth login (e.g. `openai-codex`) |

## 🐳 Docker

```bash
docker build -t bubbles .
docker run -v ~/.bubbles:/root/.bubbles --rm bubbles onboard
docker run -v ~/.bubbles:/root/.bubbles -p 18790:18790 bubbles gateway
```

## 📁 Project Structure

```
bubbles/
├── agent/          # Core agent logic
├── channels/       # Chat channel integrations
├── bus/            # Message routing
├── providers/      # LLM providers
├── config/         # Configuration
└── cli/            # Commands
```

## Acknowledgments

Thanks to the [nanobot](https://github.com/HKUDS/nanobot) team for the solid foundation.

## License

[MIT](LICENSE)
