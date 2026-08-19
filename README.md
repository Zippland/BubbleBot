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

### Windows 常驻运行与受控升级

微信渠道需要在登录了微信的 Windows 用户会话中运行。仓库提供一个 PowerShell
supervisor，并用当前用户的“登录时”计划任务启动它；任务不会使用 `SYSTEM`，也不会在
用户尚未登录时启动。完成一次安装后，日常升级不再需要登录远程桌面。

安装前确保：

- 当前目录是 Bubblebot clone 根目录；
- 当前分支是准备部署的干净分支（下例为 `main`）；
- Git remote `bubblebot` 指向要更新的仓库；
- `git` 与 `uv` 已在当前用户的 `PATH` 中。

在 PowerShell 中执行：

> 使用 `-StartNow` 前，先在原来手工启动 gateway 的窗口按 `Ctrl+C`，等待它完整退出；不要
> 让旧版手工进程与计划任务同时连接微信。新版会在创建 `Wcf()` 前同时检查 Windows 命名锁、
> 持久安全标记和 WCFerry RPC 监听端口，但旧进程不一定参与这些协议。
> 安装脚本必须用 Windows 自带的 **64 位 Windows PowerShell 5.1** 运行，不要用 `pwsh`
> 或 `SysWOW64` 下的 32 位 PowerShell；安装和
> 后续稳定副本刷新都会先用同一版本的 Parser API 校验脚本语法与参数协议；若新版新增了
> 旧计划任务不会传入的必填参数，本次升级会拒绝替换并回滚。

```powershell
Set-Location C:\path\to\Bubblebot
.\scripts\install-bubbles-gateway-task.ps1 `
  -RepoPath (Get-Location).Path `
  -Remote bubblebot `
  -Branch main `
  -Port 18790 `
  -StartNow
```

安装脚本会先执行一次 `uv sync --locked`，再把 supervisor 原子复制到
`%USERPROFILE%\.bubbles\control\bubbles-supervisor.ps1`。计划任务始终运行这份仓库外的稳定
副本，并显式传入 clone 根目录；因此 `git pull` 不会改写正在执行或下次登录要执行的
supervisor。安装时解析到的 `git.exe` 和 `uv.exe` 绝对路径也会固化到任务参数，避免计划
任务的 `PATH` 与交互式 PowerShell 不同。安装脚本还会记录 remote fetch URL 的 SHA-256
摘要（任务参数中不保存 URL 本身）；每次升级前重新比对，remote 被改指后会拒绝拉取。

升级成功后，当前稳定 supervisor 会把新仓库里的 supervisor 原子刷新到上述仓库外路径，
供下次计划任务或登录使用；当前正在运行的 PowerShell 实例不会热替换自身，仍按 v1 控制
协议完成本次运行，也始终不会直接从正在被更新的仓库脚本启动。

计划任务会覆盖同名的 `Bubblebot Gateway` 任务，采用 `Interactive`、`Limited` 当前用户
身份和隐藏窗口，忽略重复启动，且没有 72 小时执行上限。移除任务：

```powershell
Unregister-ScheduledTask -TaskName "Bubblebot Gateway" -Confirm:$false
```

supervisor 始终从 clone 根目录执行以下命令，启动参数**不带** `-v`：

```powershell
uv run --no-sync bubbles gateway --port 18790
```

每次新的 supervisor 实例启动时，都会先确认 checkout 仍在安装时固定的分支且工作树完全
干净，并在 `uv sync --locked` 前后静态读取 `bubbles/gateway_control.py`，确认控制协议仍是
当前 supervisor 支持的 v1；因此用户手工 pull 后重新登录，也不会直接用不兼容的新 gateway
替换旧进程。真正创建进程前还会再做一次同样校验。循环内普通崩溃重启不重复同步，但仍保留
最后这道启动前协议栅栏。

注意参数位置不同，含义也不同：`bubbles -v` 是打印版本；`bubbles gateway -v` 才是详细
日志。计划任务默认不启用详细日志。

gateway 正常退出（退出码 `0`）时 supervisor 一并结束。普通非零退出最多重启 5 次，
默认按 2、4、8、16、32 秒退避；连续运行 5 分钟后会重置失败计数。每个 Windows 用户在
整台机器上只允许一个 Bubblebot supervisor：它使用带用户 SID 的 `Global` 命名锁，因此
控制台、RDP 会话、不同 clone 和不同计划任务名也不能同时操作共享控制文件。整台机器的
控制台、RDP 会话和所有 Bubblebot clone 还会共用一个 `Global` WCFerry 命名锁，不能同时
构造两个 `Wcf()`。这个锁独立于 supervisor：即使有人绕过计划任务手工启动新版 gateway，
也必须先取得同一把跨会话锁。
升级事务同时记录所属 clone 的绝对路径；异常恢复只允许原 clone 接管，其他计划任务即使随后
取得 supervisor 锁也会失败关闭，不会替别的 clone 停候选进程或执行回滚。

WCFerry 会把 `spy.dll` 注入已经运行的 `WeChat.exe`，所以“Python 进程已经退出”并不等于
“微信注入已经卸载”。gateway 在注入前创建
`%USERPROFILE%\.bubbles\control\wcferry-lease.json`。停止时由 Bubblebot 自己按固定顺序
best effort 停止消息接收并立即清除运行/接收标记，先关闭消息 socket 并等待唯一捕获的
WCFerry 内部 `GetMessage` 线程退出，再关闭命令 socket、只调用一次 `WxDestroySDK`，并确认
返回值严格等于 `0`、
`10086-10087` 端口都已释放后，才删除 lease。这里不调用内部还会再次执行 native destroy 的
`Wcf.cleanup()`，避免同一次停止重复卸载 SDK。所有由 supervisor 启动的 gateway 也会在完成
上述证明后写入带唯一 instance ID 和控制协议版本的 `gateway-stopped.json`；supervisor 只有
看到当前实例的匹配证明才允许自动重启。退出码 `76` 表示“停止 supervisor，不得自动重启”。

如果 native 清理卡住，gateway 会以 `76` 失败关闭并保留 lease；候选版本回滚、断电恢复和
supervisor 自身退出都使用 `gateway-stop-request.json` / `gateway-stopped.json` 握手，绝不
使用 `taskkill /F`。握手、lease 或进程身份无法验证时宁可停服，也不会再连接第二个 WCFerry。

要允许微信管理员触发升级，在 `~/.bubbles/config.json` 中配置独立于普通聊天白名单的
升级管理员，例如。这里必须填写稳定的微信 `wxid`；如果本人已经在
`channels.wechat.allow_from` 中，可以直接复用同一个值。

```json
{
  "gateway": {
    "update": {
      "enabled": true,
      "remote": "bubblebot",
      "branch": "main",
      "admins": {
        "wechat": ["wxid_owner"]
      }
    }
  }
}
```

#### 退出码 75 升级协议

请求方必须先原子写入 `%USERPROFILE%\.bubbles\control\upgrade-request.json`，再让 gateway
以退出码 `75` 结束。请求格式：

```json
{
  "schema_version": 1,
  "action": "upgrade",
  "request_id": "a-unique-request-id",
  "channel": "wechat",
  "chat_id": "wxid_owner",
  "sender_id": "wxid_owner",
  "remote": "bubblebot",
  "branch": "main",
  "requested_at": "2026-08-19T12:00:00Z"
}
```

收到退出码 `75` 后，supervisor 会依次：

1. 校验请求目标与安装时固定的 remote/branch 完全一致；
2. 校验 clone 当前就在该 branch，并且已跟踪和未跟踪文件都为空；
3. 写入持久事务标记，再执行 `git pull --ff-only <remote> <branch>`；
4. 校验新 checkout 的 gateway 控制协议仍与当前 supervisor 兼容，再执行 `uv sync --locked`；
5. 注入本次请求 ID，用 `uv run --no-sync bubbles gateway --port <port>` 启动候选 gateway；
6. 等候候选进程同时完成 Agent 启动，并确认唯一的 WCFerry 内部 `GetMessage` 线程、消息
   socket pipe 和 Bubblebot 消费线程都存活且接收标记为真，再通过
   `%USERPROFILE%\.bubbles\control\gateway-ready.json` 回报同一请求 ID，再连续存活 5 秒；
7. 只有就绪校验通过，才删除旧请求、把事务标记切换为 `committing`、原子写入匹配的
   `%USERPROFILE%\.bubbles\control\upgrade-result.json`，最后删除事务标记作为唯一提交点；
   supervisor 随后继续监督这一个已经启动的进程，不会再拉起第二个 WCFerry 实例。

候选 gateway 在看到 supervisor 写入同一请求 ID 的成功结果前处于 `startup_pending`，此时
再次发送 `/upgrade` 只会得到“正在确认上一次升级”的回复，不会覆盖升级请求。微信接收线程
若在就绪验证期间退出，gateway 会以非零状态结束，supervisor 会把本次升级判定为失败。

结果文件使用同一请求 ID 和消息目标，包含 `ok`、`old_revision`、`new_revision`、失败阶段
和退出码；不会包含可能回发到微信的 Git/uv 原始输出。详细诊断只进入 Windows 本机的
`%USERPROFILE%\.bubbles\control\supervisor.log`，其中 URL 凭据和常见 token 形式会被脱敏。

只有 supervisor 已取得可信的旧 revision，且失败后仓库仍能通过分支、洁净状态和 revision
复核时，预检或 pull 失败才会写失败结果并重启原版本；在取得可信旧 revision 前发生的请求、
目标或仓库校验失败会直接停止 supervisor。pull 成功但
`uv sync --locked` 失败、候选进程提前退出，或候选进程在 120 秒内没有完成 Agent + 微信
就绪时，会先向候选实例写入受控停止请求；只有候选完成 native WCFerry 清理、写入匹配确认并
退出后，才利用升级前“工作树完全干净”的前提执行
`git reset --hard <old_revision>`，再按旧 `uv.lock` 执行 `uv sync --locked`；回滚成功后写
失败结果并重启原版本。停止握手失败、回滚依赖失败、事务标记无法安全清理，或结果文件不能
可靠持久化时都会停止；不会强杀候选后冒险启动原版本。
如果断电或 supervisor 异常退出留下
`%USERPROFILE%\.bubbles\control\upgrade-in-progress.json`，下一次登录会先读取其中的
`old_revision` / `new_revision`。只有当前分支仍匹配、工作树完全干净，而且 HEAD 正是本次
事务记录的旧或新 revision 时，才会自动执行 `git reset --hard` 和旧 lock 的
`uv sync --locked`，恢复稳定 supervisor、清理事务标记并启动原版本；检测到人工修改、
切换分支或其他提交时会停止，绝不静默覆盖。只有该自动恢复也失败时才需要管理员本机处理。
如果异常发生在候选进程启动后，事务标记还会保存候选 instance ID、`uv.exe` 的 PID、UTC
启动时间和绝对路径。进程仍存在时四者完全匹配才会发送停止请求；进程已消失时必须已有同一
instance ID 的 native-cleanup 确认。身份或确认不匹配，或者进程刚启动但身份尚未来得及
持久化时，supervisor 会安全停止并要求人工检查，不会冒险再启动第二个微信连接。

supervisor 禁止 Git/Git Credential Manager 弹出交互式凭据提示；SSH 使用 BatchMode 和
15 秒连接超时，Git HTTP 在持续低速 30 秒后失败，uv HTTP 超时为 60 秒。它们约束网络和
凭据等待，但不是整个升级事务的硬性总时限。

启用 `gateway.update` 并配置微信管理员后，管理员可在微信私聊中单独发送精确命令
`/upgrade`（不接受参数）。gateway 会先发送受理回执；只有确认回执已经送达，才会以退出码
`75` 优雅退出并交给 supervisor。WCFerry 永远先于 Agent、cron 和其他渠道清理：native
清理超过 30 秒会以 `76` 失败关闭；只有 native 清理已经被证明成功后，剩余第三方任务仍拒绝
退出时，30 秒 watchdog 才可直接以 `75` 完成本次升级交接。

异常恢复时不要直接删除 `wcferry-lease.json` 或未配对的 stop 文件。先停止计划任务，完全
退出微信并确认没有遗留 gateway/`uv.exe`，重新启动微信，再根据
`%USERPROFILE%\.bubbles\control\supervisor.log` 处理这些安全标记。WCFerry 39.5.1 的 RPC
监听地址是 `0.0.0.0:10086-10087` 且没有 Bubblebot 侧鉴权；必须确保 Windows 防火墙和云安全组
不允许其他主机访问这两个端口。

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
