---
name: file-share
description: 上传文件到临时分享服务，返回下载链接。当你需要把生成的文件发送给用户，但`message`无法使用时，使用该 skill 生成分享链接。
---

# file-share

将本地文件上传到临时文件分享服务，获取公开下载链接。

## 使用方法

### tmpfiles.org（推荐，1小时有效）

```bash
curl -s -F "file=@<文件路径>" https://tmpfiles.org/api/v1/upload
```

示例：
```bash
curl -s -F "file=@./data/report.pdf" https://tmpfiles.org/api/v1/upload
# 返回: {"status":"success","data":{"url":"http://tmpfiles.org/12345/report.pdf"}}
# 注意：需要把 url 中的 tmpfiles.org 改成 tmpfiles.org/dl 才能直接下载
# 即: http://tmpfiles.org/dl/12345/report.pdf
```

### catbox.moe（永久有效）

```bash
curl -s -F "reqtype=fileupload" -F "fileToUpload=@<文件路径>" https://catbox.moe/user/api.php
```

示例：
```bash
curl -s -F "reqtype=fileupload" -F "fileToUpload=@./data/chart.png" https://catbox.moe/user/api.php
# 返回: https://files.catbox.moe/abc123.png
```

### litterbox（24小时临时）

```bash
curl -s -F "reqtype=fileupload" -F "time=24h" -F "fileToUpload=@<文件路径>" https://litterbox.catbox.moe/resources/internals/api.php
```

示例：
```bash
curl -s -F "reqtype=fileupload" -F "time=24h" -F "fileToUpload=@./data/data.xlsx" https://litterbox.catbox.moe/resources/internals/api.php
# 返回: https://litter.catbox.moe/abc123.xlsx
```

## 使用流程

1. 生成/处理文件，保存到 `data/` 目录
2. 使用上述任一命令上传
3. 将返回的链接发送给用户
4. 用户点击链接即可下载

## 服务对比

| 服务 | 有效期 | 大小限制 |
|------|--------|----------|
| tmpfiles.org | 1小时 | 100MB |
| catbox.moe | 永久 | 200MB |
| litterbox | 24小时 | 1GB |

## 注意事项

- 不要上传敏感信息（这些是公开链接）
- 尽量不要使用永久有效的方案
- 上传前确认文件存在且路径正确
- tmpfiles.org 返回的链接需要加 `/dl` 才能直接下载
