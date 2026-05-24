"""Provider routing helpers shared by gateway / agent / cron CLI commands.

Resolves an arbitrary `model` string to a provider name (strict, no
fallback) and builds a concrete LLMProvider instance. Pulled out of
`commands.py` so subcommand modules can route models without re-importing
the entire CLI module tree.
"""

from __future__ import annotations

import typer

from bubbles.cli.commands import console
from bubbles.config.schema import Config


def _configured_provider_names(config: Config) -> list[str]:
    """Provider names with valid credentials (api_key or OAuth)."""
    from bubbles.providers.registry import PROVIDERS
    return [
        spec.name for spec in PROVIDERS
        if (p := getattr(config.providers, spec.name, None))
        and (spec.is_oauth or p.api_key)
    ]


def _suggest_provider_names(prefix: str, config: Config, n: int = 3) -> list[str]:
    """Up to `n` close-spelling provider names, filtered to ones the user has configured.

    Returns empty list when no configured provider scores above the similarity
    threshold — caller should then just state the prefix is unknown.
    """
    import difflib
    candidates = _configured_provider_names(config)
    return difflib.get_close_matches(prefix.lower(), candidates, n=n, cutoff=0.5)


def _strict_match_provider(config: Config, model: str) -> tuple[str, str | None]:
    """Strictly resolve which provider should handle `model`. No fallback.

    Returns (provider_name, error). On success: (name, None). On failure:
    ("", user_facing_chinese_error). Distinguishes between "unknown prefix"
    (the name doesn't map to anything) and "known but unconfigured" (the
    provider exists in the registry but has no api_key / OAuth token).
    """
    from bubbles.providers.registry import PROVIDERS, find_by_name

    if not model:
        return "", "model 不能为空"

    # Respect global forced provider (config.agents.defaults.provider != "auto")
    forced = config.agents.defaults.provider
    if forced != "auto":
        p = getattr(config.providers, forced, None)
        spec = find_by_name(forced)
        if p and ((spec and spec.is_oauth) or p.api_key):
            return forced, None
        return "", f"全局强制 provider '{forced}' 未配置（api_key 缺失）"

    model_lower = model.lower()
    model_normalized = model_lower.replace("-", "_")

    # Step 1: explicit `<prefix>/<model>` → look up spec by exact name
    candidate = None
    if "/" in model_lower:
        prefix = model_lower.split("/", 1)[0].replace("-", "_")
        candidate = find_by_name(prefix)
        if not candidate:
            suggestions = _suggest_provider_names(prefix, config)
            msg = f"未知的 provider 前缀 '{prefix}'。"
            if suggestions:
                msg += f"可能想用：{', '.join(suggestions)}"
            return "", msg

    # Step 2: no prefix → keyword match (e.g. bare "gpt-5.5" → openai)
    if not candidate:
        for spec in PROVIDERS:
            for kw in spec.keywords:
                kwl = kw.lower()
                if kwl in model_lower or kwl.replace("-", "_") in model_normalized:
                    candidate = spec
                    break
            if candidate:
                break

    if not candidate:
        return "", f"无法识别模型 '{model}' 对应的 provider。建议写成 `<provider>/<model>` 形式。"

    # Step 3: candidate is a known spec — check it has credentials
    p = getattr(config.providers, candidate.name, None)
    if p and (candidate.is_oauth or p.api_key):
        return candidate.name, None

    return "", (
        f"Provider '{candidate.name}' 还没配。"
        f"在 ~/.bubbles/config.json 的 providers 段加：\n"
        f'  "{candidate.name}": {{ "api_key": "..." }}'
    )


def _make_provider_for_model(config: Config, model: str):
    """Create an LLM provider instance for an arbitrary model string.

    Returns (provider_name, provider). Raises ValueError with a user-facing
    message when the model can't be routed (unknown prefix / not configured).
    """
    from bubbles.providers.litellm_provider import LiteLLMProvider
    from bubbles.providers.openai_codex_provider import OpenAICodexProvider
    from bubbles.providers.custom_provider import CustomProvider

    provider_name, error = _strict_match_provider(config, model)
    if error:
        raise ValueError(error)

    p = getattr(config.providers, provider_name, None)

    # OpenAI Codex (OAuth)
    if provider_name == "openai_codex" or model.startswith("openai-codex/"):
        return provider_name, OpenAICodexProvider(default_model=model)

    # Custom: direct OpenAI-compatible endpoint, bypasses LiteLLM
    if provider_name == "custom":
        return provider_name, CustomProvider(
            api_key=p.api_key if p else "no-key",
            api_base=config.get_api_base(model) or "http://localhost:8000/v1",
            default_model=model,
        )

    return provider_name, LiteLLMProvider(
        api_key=p.api_key if p else None,
        api_base=config.get_api_base(model),
        default_model=model,
        extra_headers=p.extra_headers if p else None,
        provider_name=provider_name,
    )


def _make_provider(config: Config):
    """Create the default provider (used at gateway / chat startup)."""
    model = config.agents.defaults.model
    try:
        _, provider = _make_provider_for_model(config, model)
        return provider
    except ValueError as e:
        console.print(f"[red]Error: {e}.[/red]")
        console.print("Set one in ~/.bubbles/config.json under providers section")
        raise typer.Exit(1)
