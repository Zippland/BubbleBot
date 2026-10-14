# 接入聊天平台

[返回 README](../README.md) · [配置与扩展](configuration.md) · [Windows 微信](windows-wechat.md)

先在终端验证模型配置，再接入一个聊天平台。以下 JSON 都应合并进 `~/.bubbles/config.json`；示例 ID、token 和 secret 必须替换。

## 接入步骤

1. 在平台创建机器人或准备客户端，取得凭据。
2. 配置 `channels.<channel>`，启用渠道，并填写访问白名单。
3. 运行 `uv run bubbles gateway`，检查连接日志。
4. 私聊机器人发送 `/session personal`；群聊先 @ 机器人，再发送 `/session work`。
5. 发送一句普通请求，确认模型回复。需要文件、图片时，再分别验证对应权限。

**能连接平台，不等于模型调用、媒体下载和消息发送都已验证。** 按上面的顺序检查，可以更快定位问题。

## Telegram

从 BotFather 获取 Bot Token，填写自己账户的用户 ID：

```json
{
  "channels": {
    "telegram": {
      "enabled": true,
      "token": "YOUR_BOT_TOKEN",
      "allowFrom": ["YOUR_TELEGRAM_USER_ID"]
    }
  }
}
```

需要网络代理时，可设置 `channels.telegram.proxy`。加入群聊后，先确认机器人可以接收发给它的消息，再 @ 它绑定 session。

## 飞书

在[飞书开放平台](https://open.feishu.cn/)创建应用并启用机器人能力，配置消息接收与发送权限，使用长连接接收事件，订阅 `im.message.receive_v1`。按组织要求完成版本发布，并把机器人添加到目标群。

```json
{
  "channels": {
    "feishu": {
      "enabled": true,
      "appId": "YOUR_APP_ID",
      "appSecret": "YOUR_APP_SECRET",
      "allowFrom": ["YOUR_OPEN_ID"]
    }
  }
}
```

`allowFrom` 填用户 `open_id`，不是群 `chat_id`。事件通过 WebSocket 接收，不要求为机器人暴露一个公网 webhook。

飞书推送哪些群消息取决于应用的权限与订阅。只希望被 @ 时工作，可以从群内 @ 消息接收能力开始；不要假设已自动获得整群历史。图片、文件、引用消息和群成员查询还需对应资源权限，缺失时查看平台错误码和网关日志。

## 微信

见 [Windows 微信部署](windows-wechat.md)，其中包括渠道配置、客户端要求、常驻运行和受控升级。

微信使用 WCFerry，依赖兼容的 Windows 微信客户端。它不是官方 Bot API；不要把其他平台的 token 接入方式套用到微信，也不要在未评估账号风险时直接连接重要账号。

## 其他平台

| 渠道配置项 | 所需配置或额外步骤 |
| --- | --- |
| `channels.discord` | `token`；确认机器人权限及 Gateway intents。 |
| `channels.slack` | `botToken`、`appToken`；启用 Socket Mode。 |
| `channels.dingtalk` | `clientId`、`clientSecret`；使用 Stream 模式。 |
| `channels.qq` | 官方机器人 `appId`、`secret`。 |
| `channels.whatsapp` | Node.js 网桥；运行 `uv run bubbles channels login` 按提示扫码。 |
| `channels.email` | IMAP、SMTP 凭据；检查收信、自动回复和 consent 相关开关。 |
| `channels.matrix` | 先 `uv sync --locked --extra matrix`，配置 `homeserver`、`accessToken` 等字段。 |
| `channels.mochat` | `clawToken` 等平台连接配置。 |

各渠道的完整字段、默认值和群聊策略见[配置 schema](../bubbles/config/schema.py)。

## 多个聊天如何共享一个助手

同一个聊天发送 `/session work` 后，后续消息绑定到 `work`。另一个入口也绑定到 `work`，即可共享历史、记忆和工作文件；解绑用 `/session unbind`，不会删除已有工作空间。

共享 session 也共享当前发送目标。目标首次由触发聊天初始化，之后由模型通过 `switch_message_target` 显式切换，不会因换了入口就自动重置；详见[消息协议](agent-protocol.md)。

不同人或不相关群聊建议使用不同 session。渠道白名单是入口控制，不应把 session 名称当成多租户权限边界。

## 排查收不到回复

- **渠道未连接**：检查 `enabled`、凭据、网络与平台事件配置。
- **提示未绑定或无权限**：确认发件人在 `allowFrom` 中，并在正确的聊天里发送 `/session <name>`。
- **私聊可以、群聊不行**：检查 @、群策略，以及平台允许机器人接收哪些事件。
- **文字可以、图片不行**：检查资源权限和模型视觉支持；收到附件不代表模型能够识图。
- **回复去了别处**：检查是否将多个入口绑到了同一个 session，以及当前消息发送目标。

大多数渠道的空 `allowFrom` 表示不限制发件人，微信空 `groups` 表示不限制群。正式接入前请显式设置访问范围。
