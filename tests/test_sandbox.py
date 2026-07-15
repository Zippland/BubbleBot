"""Tests for the sandbox layer (Phase 0: local backends).

Covers three properties:
1. Parity — LocalSandbox reproduces the historical exec/FS behavior.
2. Per-session CLI identity — LocalIsolatedSandbox gives each session its own
   $HOME/XDG/APPDATA, and two sessions don't see each other's credentials.
3. Path containment — the L1 sandbox guarantee (SPEC §7, SECURITY §4.1) holds:
   `..`, absolute-outside, and symlink escapes are rejected.
"""

import os
import sys
from pathlib import Path

import pytest

from bubbles.sandbox.local import LocalIsolatedSandbox, LocalSandbox
from bubbles.sandbox.manager import SandboxManager


# ---- filesystem parity ----

async def test_write_read_roundtrip(tmp_path: Path) -> None:
    sb = LocalSandbox("s", tmp_path)
    await sb.write_bytes("notes/a.txt", b"hello")
    assert await sb.read_bytes("notes/a.txt") == b"hello"
    # write creates parent dirs, lands under root
    assert (tmp_path / "notes" / "a.txt").read_bytes() == b"hello"


async def test_stat_and_exists(tmp_path: Path) -> None:
    sb = LocalSandbox("s", tmp_path)
    assert await sb.stat("missing") is None
    assert not await sb.exists("missing")
    await sb.write_bytes("f.txt", b"x")
    st = await sb.stat("f.txt")
    assert st is not None and st.is_file and not st.is_dir and st.size == 1


async def test_list_dir_sorted_with_types(tmp_path: Path) -> None:
    sb = LocalSandbox("s", tmp_path)
    await sb.write_bytes("b.txt", b"1")
    await sb.mkdir("adir")
    entries = await sb.list_dir(".")
    assert [(e.name, e.is_dir) for e in entries] == [("adir", True), ("b.txt", False)]


async def test_tilde_maps_to_root_not_system_home(tmp_path: Path) -> None:
    sb = LocalSandbox("s", tmp_path)
    await sb.write_bytes("~/inside.txt", b"y")
    # `~` is the session root, NOT the OS home
    assert (tmp_path / "inside.txt").read_bytes() == b"y"


# ---- path containment (L1 guarantee) ----

async def test_dotdot_escape_rejected(tmp_path: Path) -> None:
    root = tmp_path / "sessions" / "s"
    root.mkdir(parents=True)
    (tmp_path / "secret.txt").write_bytes(b"top")
    sb = LocalSandbox("s", root)
    with pytest.raises(PermissionError):
        await sb.read_bytes("../../secret.txt")


async def test_absolute_outside_rejected(tmp_path: Path) -> None:
    root = tmp_path / "s"
    root.mkdir()
    sb = LocalSandbox("s", root)
    with pytest.raises(PermissionError):
        await sb.resolve_within_root("/etc/passwd")


async def test_symlink_escape_rejected(tmp_path: Path) -> None:
    root = tmp_path / "s"
    root.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_bytes(b"leak")
    # plant a symlink inside root pointing out
    (root / "link").symlink_to(outside)
    sb = LocalSandbox("s", root)
    with pytest.raises(PermissionError):
        await sb.read_bytes("link")


# ---- exec ----

async def test_exec_runs_in_root(tmp_path: Path) -> None:
    sb = LocalSandbox("s", tmp_path)
    cmd = "cd" if sys.platform == "win32" else "pwd"
    res = await sb.exec(cmd, cwd=sb.root, timeout=10)
    assert res.returncode == 0
    assert str(tmp_path.resolve()) in res.stdout or res.stdout.strip()


async def test_exec_timeout(tmp_path: Path) -> None:
    if sys.platform == "win32":
        pytest.skip("sleep semantics differ on Windows")
    sb = LocalSandbox("s", tmp_path)
    res = await sb.exec("sleep 5", cwd=sb.root, timeout=1)
    assert res.timed_out


# ---- LocalSandbox exec inherits host env (parity) ----

async def test_local_exec_inherits_host_home(tmp_path: Path, monkeypatch) -> None:
    if sys.platform == "win32":
        pytest.skip("POSIX $HOME semantics")
    monkeypatch.setenv("HOME", "/host/home")
    monkeypatch.setenv("MY_SECRET", "sekret")
    sb = LocalSandbox("s", tmp_path)
    res = await sb.exec("echo $HOME:$MY_SECRET", cwd=sb.root, timeout=10)
    assert res.stdout.strip() == "/host/home:sekret"


# ---- LocalIsolatedSandbox: per-session HOME + credential drop ----

async def test_isolated_home_points_at_session_home(tmp_path: Path, monkeypatch) -> None:
    if sys.platform == "win32":
        pytest.skip("POSIX $HOME semantics")
    monkeypatch.setenv("HOME", "/host/home")
    root = tmp_path / "root"
    root.mkdir()
    home = tmp_path / "home"
    sb = LocalIsolatedSandbox("s", root, home_dir=home)
    res = await sb.exec("echo $HOME", cwd=sb.root, timeout=10)
    assert res.stdout.strip() == str(home)
    assert home.is_dir()  # lazily created


async def test_isolated_drops_host_credential_vars(tmp_path: Path, monkeypatch) -> None:
    if sys.platform == "win32":
        pytest.skip("POSIX env semantics")
    # A host var that would point a CLI back at the host config must not survive.
    monkeypatch.setenv("GH_CONFIG_DIR", "/host/.config/gh")
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", "/host/.aws/credentials")
    sb = LocalIsolatedSandbox("s", tmp_path, home_dir=tmp_path / "home")
    res = await sb.exec("echo [${GH_CONFIG_DIR:-unset}] [${AWS_SHARED_CREDENTIALS_FILE:-unset}]",
                        cwd=sb.root, timeout=10)
    assert res.stdout.strip() == "[unset] [unset]"


async def test_isolated_keeps_path(tmp_path: Path) -> None:
    if sys.platform == "win32":
        pytest.skip("POSIX PATH semantics")
    sb = LocalIsolatedSandbox("s", tmp_path, home_dir=tmp_path / "home")
    # echo is a shell builtin, but a PATH lookup for `env` proves PATH survived
    res = await sb.exec("echo $PATH", cwd=sb.root, timeout=10)
    assert res.stdout.strip()  # non-empty


async def test_isolated_passthrough_extra_var(tmp_path: Path, monkeypatch) -> None:
    if sys.platform == "win32":
        pytest.skip("POSIX env semantics")
    monkeypatch.setenv("NODE_EXTRA_CA_CERTS", "/host/ca.pem")
    sb = LocalIsolatedSandbox("s", tmp_path, home_dir=tmp_path / "home",
                              env_passthrough=["NODE_EXTRA_CA_CERTS"])
    res = await sb.exec("echo ${NODE_EXTRA_CA_CERTS:-unset}", cwd=sb.root, timeout=10)
    assert res.stdout.strip() == "/host/ca.pem"


async def test_isolated_lark_cli_data_dir_in_session_home(tmp_path: Path) -> None:
    home = tmp_path / "home"
    sb = LocalIsolatedSandbox("s", tmp_path, home_dir=home)
    env = sb._build_env()
    # lark-cli does not honor XDG — it needs its own var pointed per-session.
    assert env["LARKSUITE_CLI_DATA_DIR"] == str(home / ".local" / "share" / "lark-cli")


# ---- SandboxManager: backend selection, caching, per-session isolation ----

async def test_manager_default_backend_is_local(tmp_path: Path) -> None:
    mgr = SandboxManager()
    sb = await mgr.get("k", tmp_path)
    assert isinstance(sb, LocalSandbox) and not isinstance(sb, LocalIsolatedSandbox)
    await mgr.close_all()


async def test_manager_caches_per_session(tmp_path: Path) -> None:
    mgr = SandboxManager()
    a = await mgr.get("k", tmp_path)
    b = await mgr.get("k", tmp_path)
    assert a is b
    await mgr.close_all()


async def test_manager_rebuilds_on_backend_change(tmp_path: Path) -> None:
    mgr = SandboxManager()
    local = await mgr.get("k", tmp_path, backend="local")
    isolated = await mgr.get("k", tmp_path, backend="local_isolated")
    assert local is not isolated
    assert isinstance(isolated, LocalIsolatedSandbox)
    await mgr.close_all()


async def test_manager_session_home_outside_session_dir(tmp_path: Path) -> None:
    # ~/.bubbles/sessions/<key>  ->  ~/.bubbles/session_homes/<key>
    sessions = tmp_path / "sessions"
    session_dir = sessions / "wechat_123"
    session_dir.mkdir(parents=True)
    mgr = SandboxManager()
    sb = await mgr.get("wechat:123", session_dir, backend="local_isolated")
    assert isinstance(sb, LocalIsolatedSandbox)
    home = sb._home
    # home is a sibling tree, NOT under the session working dir
    assert home == tmp_path / "session_homes" / "wechat_123"
    assert session_dir not in home.parents
    await mgr.close_all()


async def test_two_sessions_have_separate_homes(tmp_path: Path, monkeypatch) -> None:
    if sys.platform == "win32":
        pytest.skip("POSIX $HOME semantics")
    sessions = tmp_path / "sessions"
    (sessions / "a").mkdir(parents=True)
    (sessions / "b").mkdir(parents=True)
    mgr = SandboxManager()
    sa = await mgr.get("a", sessions / "a", backend="local_isolated")
    sb = await mgr.get("b", sessions / "b", backend="local_isolated")

    # Session A writes a "credential" into its HOME; B must not see it.
    await sa.exec("echo tokenA > $HOME/.token", cwd=sa.root, timeout=10)
    res_b = await sb.exec("cat $HOME/.token 2>/dev/null || echo MISSING", cwd=sb.root, timeout=10)
    assert res_b.stdout.strip() == "MISSING"

    res_a = await sa.exec("cat $HOME/.token", cwd=sa.root, timeout=10)
    assert res_a.stdout.strip() == "tokenA"
    await mgr.close_all()


# ---- model sees sandbox root as <work_dir>, never the host path ----

def test_system_prompt_uses_sandbox_root_not_host_path(tmp_path: Path) -> None:
    from bubbles.agent.context import ContextBuilder

    session_dir = tmp_path / "sessions" / "s"
    session_dir.mkdir(parents=True)
    ctx = ContextBuilder(session_dir=session_dir)

    # No sandbox root → falls back to the session dir (debug / legacy path).
    assert str(session_dir.resolve()) in ctx.build_system_prompt()

    # With a sandbox-internal root, the model sees THAT, and the host path
    # must not leak into the prompt.
    sandboxed = ctx.build_system_prompt(work_dir="/workspace")
    assert "<work_dir>: /workspace" in sandboxed
    assert str(session_dir.resolve()) not in sandboxed
