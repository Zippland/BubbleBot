# Skills

Skills 把可复用的工作方法交给助手：什么时候使用、如何执行、需要哪些依赖和权限。
新 session 会从此目录初始化技能，用户可在自己的 session 中增删改，不必修改 Agent 核心。

## 内置技能

| Skill | 用途 |
| --- | --- |
| [canvas-design](canvas-design/SKILL.md) | 视觉设计工作流，附字体资源。 |
| [summarize](summarize/SKILL.md) | URL、文件和视频内容的摘要流程，需相应外部工具。 |
| [tmux](tmux/SKILL.md) | 通过 tmux 管理终端会话，需先安装 tmux。 |
| [file-share](file-share/SKILL.md) | 原生附件发送不可用时，用外部分享服务提供链接；不要上传敏感文件。 |

技能说明不等于依赖已经安装；执行前请查看对应 `SKILL.md` 的要求。

## 添加自己的技能

在 `~/.bubbles/sessions/<session>/skills/<name>/SKILL.md` 中写入：

```markdown
---
name: release-notes
description: 将项目变更整理成面向用户的发布说明。
---

# 发布说明

1. 确认需要比较的版本与读者。
2. 阅读变更，区分新功能、修复和兼容性变化。
3. 在工作目录保存一份 Markdown 草稿，请用户检查。
```

模型会看到技能索引，再按需读取详细说明。附属脚本或资源放在同一技能目录中。

## 同步模板

```bash
uv run bubbles sync-skills --session personal --dry-run
uv run bubbles sync-skills --session personal
```

先预览，再同步。**同步会替换模板中同名的整个技能目录**，包括你对这些技能的本地修改；请先备份。自建且不与模板同名的技能不受影响。省略 `--session` 会同步所有 session。
添加或分发第三方技能前，请核对许可和依赖；附带材料的许可见[第三方材料说明](../../../../THIRD_PARTY_NOTICES.md)。
