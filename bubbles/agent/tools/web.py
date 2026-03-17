"""Web tools: web_search and web_fetch."""

from __future__ import annotations

import html
import ipaddress
import logging
import os
import re
import time
from typing import Any, Optional
from urllib.parse import urlparse

import httpx
import trafilatura

from bubbles.agent.tools.base import Tool

logger = logging.getLogger(__name__)

# ============ 共享常量 ============
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
)

# ============ WebFetch 缓存配置 ============
DEFAULT_CACHE_TTL_SECONDS = 15 * 60  # 15 分钟
DEFAULT_CACHE_MAX_ENTRIES = 100

# ============ WebFetch 响应体限制 ============
DEFAULT_MAX_RESPONSE_BYTES = 2_000_000  # 2MB
DEFAULT_MAX_CHARS = 10_000

# ============ 缓存存储 ============
_cache: dict[str, dict[str, Any]] = {}


def _cache_key(url: str) -> str:
    """生成缓存键（规范化 URL）。"""
    return url.strip().lower()


def _read_cache(url: str) -> Optional[dict[str, Any]]:
    """读取缓存，过期则删除。"""
    key = _cache_key(url)
    entry = _cache.get(key)
    if not entry:
        return None
    if time.time() > entry["expires_at"]:
        del _cache[key]
        return None
    return entry


def _write_cache(url: str, content: str, title: Optional[str], extractor: str) -> None:
    """写入缓存，超过最大条目数时删除最旧的。"""
    if len(_cache) >= DEFAULT_CACHE_MAX_ENTRIES:
        oldest_key = min(_cache.keys(), key=lambda k: _cache[k]["inserted_at"])
        del _cache[oldest_key]

    _cache[_cache_key(url)] = {
        "content": content,
        "title": title,
        "extractor": extractor,
        "expires_at": time.time() + DEFAULT_CACHE_TTL_SECONDS,
        "inserted_at": time.time(),
    }


# ============ SSRF 防护 ============
def _is_safe_url(url: str) -> tuple[bool, str]:
    """
    检查 URL 是否安全（非内网地址）。

    Returns:
        (is_safe, error_message)
    """
    try:
        parsed = urlparse(url)
    except Exception:
        return False, "Invalid URL format"

    if parsed.scheme not in ("http", "https"):
        return False, "Only http/https allowed"

    host = parsed.hostname
    if not host:
        return False, "Missing hostname"

    # 检查常见的内网域名
    if host in ("localhost", "127.0.0.1", "::1"):
        return False, "localhost not allowed"

    # 检查 IP 地址
    try:
        ip = ipaddress.ip_address(host)
        if ip.is_private:
            return False, f"Private IP not allowed: {host}"
        if ip.is_loopback:
            return False, f"Loopback not allowed: {host}"
        if ip.is_reserved:
            return False, f"Reserved IP not allowed: {host}"
        if ip.is_link_local:
            return False, f"Link-local not allowed: {host}"
        # 云服务元数据地址
        if str(ip) == "169.254.169.254":
            return False, "Cloud metadata address not allowed"
    except ValueError:
        # 不是 IP 地址，是域名，允许
        pass

    return True, ""


# ============ 响应体限制读取 ============
async def _read_response_limited(
    response: httpx.Response,
    max_bytes: int = DEFAULT_MAX_RESPONSE_BYTES,
) -> tuple[str, bool]:
    """
    限制响应体大小读取。

    Returns:
        (content, truncated)
    """
    chunks = []
    bytes_read = 0
    truncated = False

    async for chunk in response.aiter_bytes():
        if bytes_read + len(chunk) > max_bytes:
            remaining = max_bytes - bytes_read
            if remaining > 0:
                chunks.append(chunk[:remaining])
            truncated = True
            break
        chunks.append(chunk)
        bytes_read += len(chunk)

    content = b"".join(chunks).decode("utf-8", errors="replace")
    return content, truncated


# ============ 基础 HTML 清理（trafilatura 后备） ============
def _strip_tags(text: str) -> str:
    """去除 HTML 标签。"""
    return re.sub(r"<[^>]+>", "", text)


def _decode_entities(text: str) -> str:
    """解码 HTML 实体。"""
    text = html.unescape(text)
    text = text.replace("\xa0", " ")  # 非断行空格 → 普通空格
    return text


def _normalize_whitespace(text: str) -> str:
    """规范化空白字符。"""
    text = re.sub(r"\r", "", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def _basic_html_to_markdown(html_content: str) -> tuple[str, Optional[str]]:
    """
    基础 HTML 清理，作为 trafilatura 的后备。

    Returns:
        (markdown_text, title)
    """
    # 提取标题
    title_match = re.search(r"<title[^>]*>([\s\S]*?)</title>", html_content, re.I)
    title = _normalize_whitespace(_strip_tags(title_match.group(1))) if title_match else None

    text = html_content

    # 删除脚本、样式、noscript
    text = re.sub(r"<script[\s\S]*?</script>", "", text, flags=re.I)
    text = re.sub(r"<style[\s\S]*?</style>", "", text, flags=re.I)
    text = re.sub(r"<noscript[\s\S]*?</noscript>", "", text, flags=re.I)

    # 转换标题标签
    for i in range(1, 7):
        pattern = rf"<h{i}[^>]*>([\s\S]*?)</h{i}>"
        text = re.sub(
            pattern,
            lambda m, level=i: f"\n{'#' * level} {_normalize_whitespace(_strip_tags(m.group(1)))}\n",
            text,
            flags=re.I,
        )

    # 转换链接
    text = re.sub(
        r'<a\s+[^>]*href=["\']([^"\']+)["\'][^>]*>([\s\S]*?)</a>',
        lambda m: f"[{_normalize_whitespace(_strip_tags(m.group(2)))}]({m.group(1)})",
        text,
        flags=re.I,
    )

    # 转换列表项
    text = re.sub(
        r"<li[^>]*>([\s\S]*?)</li>",
        lambda m: f"\n- {_normalize_whitespace(_strip_tags(m.group(1)))}",
        text,
        flags=re.I,
    )

    # 换行标签
    text = re.sub(r"<(br|hr)\s*/?>", "\n", text, flags=re.I)
    text = re.sub(r"</(p|div|section|article|header|footer|table|tr|ul|ol)>", "\n", text, flags=re.I)

    # 去除剩余标签
    text = _strip_tags(text)

    # 解码实体
    text = _decode_entities(text)

    # 规范化空白
    text = _normalize_whitespace(text)

    return text, title


class WebSearchTool(Tool):
    """Search the web using Tavily Search API."""

    def __init__(self, api_key: str | None = None, max_results: int = 5):
        self._init_api_key = api_key
        self.max_results = max_results

    @property
    def name(self) -> str:
        return "web_search"

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "count": {"type": "integer", "description": "Results (1-10)", "minimum": 1, "maximum": 10},
            },
            "required": ["query"],
        }

    @property
    def api_key(self) -> str:
        """Resolve API key at call time so env/config changes are picked up."""
        return self._init_api_key or os.environ.get("TAVILY_API_KEY", "")

    async def execute(self, query: str, count: int | None = None, **kwargs: Any) -> str:
        if not self.api_key:
            return (
                "Error: Tavily API key not configured. "
                "Set TAVILY_API_KEY environment variable or configure in ~/.bubbles/config.json"
            )

        try:
            n = min(max(count or self.max_results, 1), 10)
            async with httpx.AsyncClient() as client:
                r = await client.post(
                    "https://api.tavily.com/search",
                    json={
                        "api_key": self.api_key,
                        "query": query,
                        "max_results": n,
                        "search_depth": "basic",
                    },
                    timeout=15.0
                )
                r.raise_for_status()

            data = r.json()
            results = data.get("results", [])
            if not results:
                return f"No results for: {query}"

            lines = [f"Results for: {query}\n"]
            for i, item in enumerate(results[:n], 1):
                lines.append(f"{i}. {item.get('title', '')}\n   {item.get('url', '')}")
                if content := item.get("content"):
                    snippet = content[:300] + "..." if len(content) > 300 else content
                    lines.append(f"   {snippet}")
            return "\n".join(lines)
        except Exception as e:
            return f"Error: {e}"


class WebFetchTool(Tool):
    """从指定 URL 提取网页正文内容。

    使用 trafilatura 库提取正文，自动过滤导航栏、广告等噪音。
    支持分页读取长文章、缓存、SSRF 防护、响应体限制。
    """

    @property
    def name(self) -> str:
        return "web_fetch"

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "HTTP or HTTPS URL to fetch.",
                },
                "offset": {
                    "type": "integer",
                    "description": "Character offset to start reading from (default: 0). Use for pagination.",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum characters to return (default: 10000). Use offset to continue reading.",
                },
                "include_links": {
                    "type": "boolean",
                    "description": "Whether to preserve links in output (default: false).",
                },
                "include_tables": {
                    "type": "boolean",
                    "description": "Whether to extract tables (default: false).",
                },
            },
            "required": ["url"],
        }

    async def execute(
        self,
        url: str,
        offset: int = 0,
        limit: int = DEFAULT_MAX_CHARS,
        include_links: bool = False,
        include_tables: bool = False,
        **kwargs: Any,
    ) -> str:
        logger.info("WebFetchTool: url=%s, offset=%d, limit=%d", url, offset, limit)

        # 1. SSRF 防护
        is_safe, error_msg = _is_safe_url(url)
        if not is_safe:
            return f"Error: {error_msg}"

        # 2. 检查缓存
        cached = _read_cache(url)
        if cached:
            logger.info("WebFetchTool: cache hit for %s (extractor=%s)", url, cached["extractor"])
            return self._format_output(
                cached["content"],
                offset,
                limit,
                url,
                cached["extractor"],
                cached=True,
            )

        # 3. 发起请求
        try:
            async with httpx.AsyncClient(
                follow_redirects=True,
                timeout=30.0,
            ) as client:
                response = await client.get(
                    url,
                    headers={
                        "User-Agent": USER_AGENT,
                        # 优先请求 markdown（Cloudflare Markdown for Agents）
                        "Accept": "text/markdown, text/html;q=0.9, */*;q=0.1",
                        "Accept-Language": "en-US,en;q=0.9",
                    },
                )
                response.raise_for_status()

                # 4. 响应体大小限制
                body, truncated = await _read_response_limited(response)
                content_type = response.headers.get("content-type", "")

                if truncated:
                    logger.warning("WebFetchTool: response truncated at %d bytes", DEFAULT_MAX_RESPONSE_BYTES)

        except httpx.HTTPStatusError as e:
            return f"Error: HTTP {e.response.status_code} - {url}"
        except httpx.RequestError as e:
            return f"Error: Request failed - {type(e).__name__}: {e}"

        # 5. 内容提取
        extracted: Optional[str] = None
        title: Optional[str] = None
        extractor = "unknown"

        # 5a. Cloudflare Markdown for Agents
        if "text/markdown" in content_type:
            extracted = body
            extractor = "cf-markdown"
            logger.info("WebFetchTool: using Cloudflare Markdown for Agents")

        # 5b. HTML 内容
        elif "text/html" in content_type or body.strip().startswith(("<!doctype", "<!DOCTYPE", "<html", "<HTML")):
            # 首选：trafilatura
            extracted = trafilatura.extract(
                body,
                include_links=include_links,
                include_tables=include_tables,
                include_comments=False,
                output_format="markdown",
                url=url,
            )
            if extracted:
                extractor = "trafilatura"
            else:
                # 后备：基础 HTML 清理
                logger.info("WebFetchTool: trafilatura failed, falling back to basic HTML cleanup")
                extracted, title = _basic_html_to_markdown(body)
                extractor = "basic-html"

        # 5c. JSON 内容
        elif "application/json" in content_type:
            import json
            try:
                parsed = json.loads(body)
                extracted = json.dumps(parsed, indent=2, ensure_ascii=False)
                extractor = "json"
            except json.JSONDecodeError:
                extracted = body
                extractor = "raw"

        # 5d. 其他：原样返回
        else:
            extracted = body
            extractor = "raw"

        if not extracted:
            return f"Error: Failed to extract content from {url} (possibly dynamic rendering or anti-crawl)"

        # 6. 写入缓存
        _write_cache(url, extracted, title, extractor)

        # 7. 格式化输出（支持分页）
        return self._format_output(extracted, offset, limit, url, extractor, cached=False)

    def _format_output(
        self,
        content: str,
        offset: int,
        limit: int,
        url: str,
        extractor: str,
        cached: bool,
    ) -> str:
        """格式化输出，支持分页。"""
        total_chars = len(content)

        if offset >= total_chars:
            return f"offset ({offset}) exceeds content length ({total_chars}), no more content."

        sliced = content[offset : offset + limit]
        end_pos = offset + len(sliced)

        result = sliced

        # 添加元信息
        meta_parts = []
        if offset > 0 or end_pos < total_chars:
            meta_parts.append(f"chars {offset}-{end_pos} / {total_chars}")
            if end_pos < total_chars:
                meta_parts.append(f"continue with offset={end_pos}")
        if cached:
            meta_parts.append("cached")
        meta_parts.append(f"extractor={extractor}")

        if meta_parts:
            result += f"\n\n---\n[{' | '.join(meta_parts)}]"

        logger.info(
            "WebFetchTool: returned chars %d-%d of %d from %s (extractor=%s, cached=%s)",
            offset, end_pos, total_chars, url, extractor, cached,
        )
        return result
