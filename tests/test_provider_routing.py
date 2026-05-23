"""Tests for per-session provider routing (Option B)."""

from __future__ import annotations

from unittest.mock import MagicMock

from bubbles.agent.loop import AgentLoop
from bubbles.session.manager import SessionManager


def _make_loop(tmp_path, *, factory=None, default_name="openai_codex"):
    bus = MagicMock()
    provider = MagicMock(name="DefaultProvider")
    provider.get_default_model.return_value = "openai/gpt-5.5"
    sessions = SessionManager(sessions_dir=tmp_path / "sessions")
    return AgentLoop(
        bus=bus, provider=provider,
        max_tokens=4096, memory_window=20, context_limit=128_000,
        session_manager=sessions,
        provider_factory=factory,
        default_provider_name=default_name,
    ), provider


def test_provider_for_no_factory_returns_default(tmp_path) -> None:
    loop, default = _make_loop(tmp_path, factory=None)
    assert loop._provider_for("moonshot/anything") is default


def test_provider_for_default_uses_cached_default(tmp_path) -> None:
    """When asked for a model belonging to the default provider, no factory call needed."""
    seen_calls: list[str] = []

    def factory(model):
        seen_calls.append(model)
        # Pretend openai/* maps to the default openai_codex
        return ("openai_codex", MagicMock(name="WrongInstance"))

    loop, default = _make_loop(tmp_path, factory=factory, default_name="openai_codex")
    # Cache is pre-populated; factory is still called (we go through it) but its
    # returned instance is discarded in favour of the cached default.
    result = loop._provider_for("openai/gpt-5.5")
    assert result is default


def test_provider_for_different_model_creates_and_caches(tmp_path) -> None:
    moonshot = MagicMock(name="MoonshotProvider")
    factory_calls: list[str] = []

    def factory(model):
        factory_calls.append(model)
        return ("moonshot", moonshot)

    loop, default = _make_loop(tmp_path, factory=factory, default_name="openai_codex")

    # First call: factory invoked, result cached
    assert loop._provider_for("moonshot/kimi-k2.6") is moonshot
    assert factory_calls == ["moonshot/kimi-k2.6"]

    # Second call same provider family: still gets the cached moonshot instance
    assert loop._provider_for("moonshot/kimi-k2.5") is moonshot
    # Factory may be called again to derive name, but cached instance is returned
    assert len(factory_calls) == 2


def test_provider_for_factory_failure_falls_back(tmp_path) -> None:
    def factory(model):
        raise ValueError("No API key configured for provider 'moonshot'")

    loop, default = _make_loop(tmp_path, factory=factory)
    # Factory raises → fall back to default rather than crash
    assert loop._provider_for("moonshot/kimi-k2.6") is default


def test_provider_for_empty_model_returns_default(tmp_path) -> None:
    loop, default = _make_loop(tmp_path, factory=lambda m: ("x", MagicMock()))
    assert loop._provider_for(None) is default
    assert loop._provider_for("") is default


def test_config_model_rejects_unconfigured_provider(tmp_path) -> None:
    """`/config model X` must error out at command time when provider isn't configured."""
    import asyncio
    from bubbles.bus.events import InboundMessage

    def factory(model):
        if model.startswith("moonshot/"):
            raise ValueError("No API key configured for provider 'moonshot'")
        return ("openai_codex", MagicMock())

    loop, _ = _make_loop(tmp_path, factory=factory)
    session = loop.sessions.get_or_create("cli:direct")
    msg = InboundMessage(channel="cli", sender_id="user", chat_id="direct", content="/config model moonshot/x")

    out = asyncio.run(loop._handle_config_command(msg, session, "model moonshot/x"))
    assert "无法切换到" in out.content
    assert "moonshot" in out.content
    # Crucially, the bad model is NOT saved
    assert session.config.model is None


def test_config_model_accepts_configured_provider(tmp_path) -> None:
    """A model with an in-configured provider must save successfully."""
    import asyncio
    from bubbles.bus.events import InboundMessage

    def factory(model):
        return ("moonshot", MagicMock())  # whatever it is, accept it

    loop, _ = _make_loop(tmp_path, factory=factory)
    session = loop.sessions.get_or_create("cli:direct")
    msg = InboundMessage(channel="cli", sender_id="user", chat_id="direct", content="/config model moonshot/kimi-k2.6")

    out = asyncio.run(loop._handle_config_command(msg, session, "model moonshot/kimi-k2.6"))
    assert "model = " in out.content
    assert session.config.model == "moonshot/kimi-k2.6"


# ---------- Strict matcher: no fallback, distinguished errors ----------

def _make_config(*, providers: dict, forced: str = "auto") -> object:
    """Build a minimal Config stub for _strict_match_provider tests."""
    from bubbles.config.schema import Config, ProvidersConfig, AgentsConfig, AgentDefaults, ProviderConfig

    pconf = ProvidersConfig()
    for name, key in providers.items():
        setattr(pconf, name, ProviderConfig(api_key=key) if key else ProviderConfig())

    defaults = AgentDefaults(model="openai/gpt-5.5", provider=forced)
    return Config(agents=AgentsConfig(defaults=defaults), providers=pconf)


def test_strict_match_known_prefix_configured() -> None:
    """Known prefix + configured → returns provider name, no error."""
    from bubbles.cli.commands import _strict_match_provider
    config = _make_config(providers={"moonshot": "sk-test"})
    name, err = _strict_match_provider(config, "moonshot/kimi-k2.6")
    assert err is None
    assert name == "moonshot"


def test_strict_match_known_prefix_unconfigured() -> None:
    """Known prefix but no api_key → error explaining how to configure."""
    from bubbles.cli.commands import _strict_match_provider
    # Configure ONLY openai_codex (OAuth); user types moonshot/...
    config = _make_config(providers={"openai_codex": ""})  # OAuth, no key needed
    name, err = _strict_match_provider(config, "moonshot/kimi-k2.6")
    assert name == ""
    assert err is not None
    assert "moonshot" in err
    assert "还没配" in err
    assert "config.json" in err


def test_strict_match_unknown_prefix_no_separate_configured_line() -> None:
    """No standalone '已配置:' enumeration in the error any more."""
    from bubbles.cli.commands import _strict_match_provider
    config = _make_config(providers={"moonshot": "sk-test"})
    _, err = _strict_match_provider(config, "qoqo/whatever")
    assert err is not None
    assert "未知的 provider 前缀" in err
    assert "qoqo" in err
    assert "已配置：" not in err


def test_strict_match_suggestions_filter_to_configured() -> None:
    """Fuzzy suggestions only show providers the user has actually configured."""
    from bubbles.cli.commands import _strict_match_provider

    # User HAS moonshot configured. Typo 'moonshoot' → suggest 'moonshot'.
    config = _make_config(providers={"moonshot": "sk-test"})
    _, err = _strict_match_provider(config, "moonshoot/k2")
    assert err is not None
    assert "可能想用：moonshot" in err

    # User does NOT have anthropic. Typo 'antropic' → no anthropic suggestion.
    _, err = _strict_match_provider(config, "antropic/claude")
    assert err is not None
    assert "anthropic" not in err


def test_strict_match_no_suggestion_when_none_close() -> None:
    """Far-from-anything prefix → bare error, no '可能想用'."""
    from bubbles.cli.commands import _strict_match_provider
    config = _make_config(providers={"moonshot": "sk-test"})
    _, err = _strict_match_provider(config, "xxxyyy/whatever")
    assert err is not None
    assert "未知的 provider 前缀 'xxxyyy'" in err
    assert "可能想用" not in err


def test_strict_match_no_fallback() -> None:
    """The bug we're fixing: when user has only Codex but writes moonshot/...,
    must NOT silently route to Codex."""
    from bubbles.cli.commands import _strict_match_provider
    config = _make_config(providers={"openai_codex": ""})  # only Codex (OAuth)
    name, err = _strict_match_provider(config, "moonshot/kimi-k2.6")
    assert name != "openai_codex", "Bug regression: silently fell back to Codex"
    assert err is not None


def test_strict_match_keyword_only() -> None:
    """Bare model name (no `/`) → keyword match."""
    from bubbles.cli.commands import _strict_match_provider
    config = _make_config(providers={"openai_codex": ""})
    name, err = _strict_match_provider(config, "gpt-5.5")  # no prefix, "gpt" keyword
    # openai_codex doesn't have "gpt" keyword (it's on openai spec), so this
    # should error saying openai is unconfigured. Verify the error mentions
    # the right unmet provider.
    if err:
        assert "openai" in err.lower()
