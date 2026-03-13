---
name: file-share
description: Upload files to temporary sharing services and get download links. Use this skill only when the `message` tool cannot send files directly.
---

# file-share

Upload local files to temporary file sharing services to get public download links.

## Important

Always prefer using the channel's native file sending capability via the `message` tool first - this provides the best user experience. Only use this as a fallback when native file sending is unavailable or fails.

## Available Services

### tmpfiles.org (Recommended, 1 hour expiry, 100MB)

```bash
curl -s -F "file=@<file_path>" https://tmpfiles.org/api/v1/upload
# Returns: {"status":"success","data":{"url":"http://tmpfiles.org/12345/file.pdf"}}
# Note: Add /dl to URL for direct download: http://tmpfiles.org/dl/12345/file.pdf
```

### uguu.se (48 hours expiry, 128MB)

```bash
curl -s -F "files[]=@<file_path>" https://uguu.se/upload.php
# Returns: {"success":true,"files":[{"url":"https://d.uguu.se/xxxxx.pdf",...}]}
```

## Workflow

1. Generate/process file and save to `data/` directory
2. Upload using one of the commands above
3. Extract the download URL from response
4. Send the download link to the user

## Notes

- Do not upload sensitive information (public links)
- Verify the file exists before uploading
- For tmpfiles.org, add `/dl` to URL for direct download

## Unavailable Services (Do NOT use)

The following services are blocked or unreliable:
- transfer.sh - timeout
- x0.at - timeout
- litterbox.catbox.moe - timeout
- catbox.moe - timeout
- 0x0.st - blocks automated uploads
- file.io - unreliable
