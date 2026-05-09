<div align="center">
  <h1>🫧 Bubbles</h1>
  <p>
    <img src="https://img.shields.io/badge/python-≥3.11-blue" alt="Python">
    <img src="https://img.shields.io/badge/license-MIT-green" alt="License">
  </p>
</div>

🫧 **Bubbles** 是一个本地优先、以 coding agent 为内核的个人 AI 助手框架。
跨渠道触达、有持久记忆与工作空间、能主动执行任务。

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

---

## 🚀 上手

```bash
# 1. 安装
pip install -e .

# 2. 初始化 ~/.bubbles/ （配置文件 + 会话目录）
bubbles onboard

# 3. 编辑 ~/.bubbles/config.json，至少配一个 provider 的 API key 与默认 model

# 4. CLI 聊天
bubbles agent

# 5. 或者启动网关，把所有已 enable 的渠道接进来，长期常驻
bubbles gateway
```

CLI 子命令一律支持 `--help`：

```bash
bubbles --help
bubbles agent --help
bubbles channels --help
bubbles cron --help
bubbles provider --help
```

Docker：

```bash
docker build -t bubbles .
docker run -v ~/.bubbles:/root/.bubbles --rm bubbles onboard
docker run -v ~/.bubbles:/root/.bubbles -p 18790:18790 bubbles gateway
```

---

## 📖 想了解什么？看哪里？

| 我想知道                                                       | 看这里                              |
| -------------------------------------------------------------- | ----------------------------------- |
| **Bubbles 是什么、做什么、不做什么**                          | [`SPEC.md`](./SPEC.md) — 产品层唯一事实源 |
| **支持哪些渠道、哪些 LLM provider、哪些 Skill**               | [`SPEC.md`](./SPEC.md) §5           |
| **CLI 命令完整能力 / 配置顶级字段语义**                        | [`SPEC.md`](./SPEC.md) §4 / §6      |
| **配置文件里每一个字段叫什么、是什么类型、默认值是多少**        | `bubbles/config/schema.py`          |
| **安全与隐私模型详解**                                        | [`SECURITY.md`](./SECURITY.md)      |
| **预装的 Skill 都做什么 / 怎么写一个新的 Skill**              | `bubbles/templates/session/skills/` |

> **改产品行为之前，请先读并修改 [`SPEC.md`](./SPEC.md)**。
> 它声明了 Bubbles 对用户承诺的所有能力与边界；任何用户可见的改动必须先反映在那里。

---

## License

[MIT](LICENSE)
