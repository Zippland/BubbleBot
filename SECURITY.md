# Security Policy

## Reporting a Vulnerability

If you discover a security vulnerability in bubbles, please report it by:

1. **DO NOT** open a public GitHub issue
2. Create a private security advisory on GitHub or contact the repository maintainers (maintainers)
3. Include:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Suggested fix (if any)

We aim to respond to security reports within 48 hours.

## Security Best Practices

### 1. API Key Management

**CRITICAL**: Never commit API keys to version control.

```bash
# ✅ Good: Store in config file with restricted permissions
chmod 600 ~/.bubbles/config.json

# ❌ Bad: Hardcoding keys in code or committing them
```

**Recommendations:**
- Store API keys in `~/.bubbles/config.json` with file permissions set to `0600`
- Consider using environment variables for sensitive keys
- Use OS keyring/credential manager for production deployments
- Rotate API keys regularly
- Use separate API keys for development and production

### 2. Channel Access Control

**IMPORTANT**: Always configure `allowFrom` lists for production use.

```json
{
  "channels": {
    "telegram": {
      "enabled": true,
      "token": "YOUR_BOT_TOKEN",
      "allowFrom": ["123456789", "987654321"]
    },
    "whatsapp": {
      "enabled": true,
      "allowFrom": ["+1234567890"]
    }
  }
}
```

**Security Notes:**
- Empty `allowFrom` list will **ALLOW ALL** users (open by default for personal use)
- Get your Telegram user ID from `@userinfobot`
- Use full phone numbers with country code for WhatsApp
- Review access logs regularly for unauthorized access attempts

### 3. Shell Command Execution

The `exec` tool can execute shell commands. While dangerous command patterns are blocked, you should:

- ✅ Review all tool usage in agent logs
- ✅ Understand what commands the agent is running
- ✅ Use a dedicated user account with limited privileges
- ✅ Never run bubbles as root
- ❌ Don't disable security checks
- ❌ Don't run on systems with sensitive data without careful review

**Blocked patterns:**
- `rm -rf /` - Root filesystem deletion
- Fork bombs
- Filesystem formatting (`mkfs.*`)
- Raw disk writes
- Other destructive operations

### 4. File System Access

File operations have path traversal protection, but:

- ✅ Run bubbles with a dedicated user account
- ✅ Use filesystem permissions to protect sensitive directories
- ✅ Regularly audit file operations in logs
- ❌ Don't give unrestricted access to sensitive files

### 4.1 Cross-Session Isolation

Bubbles uses two cooperating layers inside the framework. The "absolutely
cannot cross sessions" property is **not** delivered by application code
alone — to get it, you also need OS / container-level isolation in your
deployment.

| Layer | What it blocks |
|-------|----------------|
| **L1: Application path sandbox** | `read_file` / `write_file` / `edit_file` / `list_dir` paths outside the session directory (incl. via symlinks, absolute paths, `~`). `exec` `working_dir` resolved through `..` or symlinks. Common shell tricks caught: `cat /other-session/...`, `cd /...`, `cd ..`, `find / ...`, variable indirection `P=/...; cat $P/...`, command substitution `$(...)`. |
| **L2: Shell-trick best-effort** | Static-analyzable bypass attempts in the `exec` tool's command string. Caught: absolute paths in quotes, `cd` to absolute targets outside session, traversal patterns. **NOT caught**: base64-decoded paths, paths constructed at runtime in subshells, write+chmod+exec workflows — anything turing-complete enough to evade static analysis. |

**For the hard guarantee, layer your deployment with OS-level isolation.**
Any of these work; pick what fits:

- **Per-session UNIX user**: each session under its own `bubbles-<key>`
  account, with `~/.bubbles/sessions/<key>` owned mode 0700 by that user.
  Filesystem permissions then block cross-session reads at the kernel level.
- **`bubblewrap` / `firejail` / `unshare`** wrapping the `exec` tool: confine
  the subprocess to a mount-namespace where only the current session's
  directory is visible.
- **`sandbox-exec`** (macOS): kernel-enforced policy that denies file-read /
  file-write on other sessions while allowing the current one.
- **One container per session**: each bubbles process bind-mounts only its
  own session's working tree.

These are deployment-side hardenings, not framework features. L1 + L2 cover
the common cases out of the box; the absolute guarantee is something you
opt into by choosing the right deployment shape.

### 5. Network Security

**API Calls:**
- All external API calls use HTTPS by default
- Timeouts are configured to prevent hanging requests
- Consider using a firewall to restrict outbound connections if needed

**WhatsApp Bridge:**
- The bridge binds to `127.0.0.1:3001` (localhost only, not accessible from external network)
- Set `bridgeToken` in config to enable shared-secret authentication between Python and Node.js
- Keep authentication data in `~/.bubbles/whatsapp-auth` secure (mode 0700)

### 6. Dependency Security

**Critical**: Keep dependencies updated!

```bash
# Check for vulnerable dependencies
pip install pip-audit
pip-audit

# Update to latest secure versions
pip install --upgrade bubbles-ai
```

For Node.js dependencies (WhatsApp bridge):
```bash
cd bridge
npm audit
npm audit fix
```

**Important Notes:**
- Keep `litellm` updated to the latest version for security fixes
- We've updated `ws` to `>=8.17.1` to fix DoS vulnerability
- Run `pip-audit` or `npm audit` regularly
- Subscribe to security advisories for bubbles and its dependencies

### 7. Production Deployment

For production use:

1. **Isolate the Environment**
   ```bash
   # Run in a container or VM
   docker run --rm -it python:3.11
   pip install bubbles-ai
   ```

2. **Use a Dedicated User**
   ```bash
   sudo useradd -m -s /bin/bash bubbles
   sudo -u bubbles bubbles gateway
   ```

3. **Set Proper Permissions**
   ```bash
   chmod 700 ~/.bubbles
   chmod 600 ~/.bubbles/config.json
   chmod 700 ~/.bubbles/whatsapp-auth
   ```

4. **Enable Logging**
   ```bash
   # Configure log monitoring
   tail -f ~/.bubbles/logs/bubbles.log
   ```

5. **Use Rate Limiting**
   - Configure rate limits on your API providers
   - Monitor usage for anomalies
   - Set spending limits on LLM APIs

6. **Regular Updates**
   ```bash
   # Check for updates weekly
   pip install --upgrade bubbles-ai
   ```

### 8. Development vs Production

**Development:**
- Use separate API keys
- Test with non-sensitive data
- Enable verbose logging
- Use a test Telegram bot

**Production:**
- Use dedicated API keys with spending limits
- Restrict file system access
- Enable audit logging
- Regular security reviews
- Monitor for unusual activity

### 9. Data Privacy

- **Logs may contain sensitive information** - secure log files appropriately
- **LLM providers see your prompts** - review their privacy policies
- **Chat history is stored locally** - protect the `~/.bubbles` directory
- **API keys are in plain text** - use OS keyring for production

### 10. Incident Response

If you suspect a security breach:

1. **Immediately revoke compromised API keys**
2. **Review logs for unauthorized access**
   ```bash
   grep "Access denied" ~/.bubbles/logs/bubbles.log
   ```
3. **Check for unexpected file modifications**
4. **Rotate all credentials**
5. **Update to latest version**
6. **Report the incident** to maintainers

## Security Features

### Built-in Security Controls

✅ **Input Validation**
- Path traversal protection on file operations
- Dangerous command pattern detection
- Input length limits on HTTP requests

✅ **Authentication**
- Allow-list based access control
- Failed authentication attempt logging
- Open by default (configure allowFrom for production use)

✅ **Resource Protection**
- Command execution timeouts (60s default)
- Output truncation (10KB limit)
- HTTP request timeouts (10-30s)

✅ **Secure Communication**
- HTTPS for all external API calls
- TLS for Telegram API
- WhatsApp bridge: localhost-only binding + optional token auth

## Known Limitations

⚠️ **Current Security Limitations:**

1. **No Rate Limiting** - Users can send unlimited messages (add your own if needed)
2. **Plain Text Config** - API keys stored in plain text (use keyring for production)
3. **No Session Management** - No automatic session expiry
4. **Limited Command Filtering** - Only blocks obvious dangerous patterns
5. **No Audit Trail** - Limited security event logging (enhance as needed)

## Security Checklist

Before deploying bubbles:

- [ ] API keys stored securely (not in code)
- [ ] Config file permissions set to 0600
- [ ] `allowFrom` lists configured for all channels
- [ ] Running as non-root user
- [ ] File system permissions properly restricted
- [ ] Dependencies updated to latest secure versions
- [ ] Logs monitored for security events
- [ ] Rate limits configured on API providers
- [ ] Backup and disaster recovery plan in place
- [ ] Security review of custom skills/tools

## Updates

**Last Updated**: 2026-02-03

For the latest security updates and announcements, check:
- GitHub Security Advisories: https://github.com/Zippland/BubbleBot/security/advisories
- Release Notes: https://github.com/Zippland/BubbleBot/releases

## License

See LICENSE file for details.
