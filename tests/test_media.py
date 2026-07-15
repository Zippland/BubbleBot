"""Tests for cross-machine media transfer (Phase 2c core, Redis-free).

This is the correctness-critical path per the Phase 2 plan ("媒体是正确性的
生死线"). Fully exercisable without Redis: the blob store is an injected
in-memory dict, so dehydrate→rehydrate round-trips prove the logic end-to-end.
"""

from pathlib import Path

from bubbles.bus.media import dehydrate_media, rehydrate_media, sha256_hex


def _mem_blob_store():
    store: dict[str, bytes] = {}

    async def put_blob(data: bytes) -> str:
        sha = sha256_hex(data)
        store[sha] = data
        return sha

    async def get_blob(sha: str) -> bytes:
        return store[sha]

    return store, put_blob, get_blob


async def test_small_file_inlined(tmp_path: Path) -> None:
    store, put, get = _mem_blob_store()
    f = tmp_path / "note.txt"
    f.write_bytes(b"hello")
    descs = await dehydrate_media([str(f)], inline_max_bytes=1024, put_blob=put)
    assert len(descs) == 1
    d = descs[0]
    assert d["name"] == "note.txt" and "inline_b64" in d and "sha256" not in d
    assert store == {}  # small file did NOT hit the blob store


async def test_large_file_goes_to_blob(tmp_path: Path) -> None:
    store, put, get = _mem_blob_store()
    f = tmp_path / "big.bin"
    f.write_bytes(b"x" * 5000)
    descs = await dehydrate_media([str(f)], inline_max_bytes=1024, put_blob=put)
    d = descs[0]
    assert "sha256" in d and "inline_b64" not in d
    assert d["size"] == 5000
    assert d["sha256"] in store  # bytes landed in the blob store


async def test_roundtrip_inline(tmp_path: Path) -> None:
    store, put, get = _mem_blob_store()
    src = tmp_path / "img.jpg"
    src.write_bytes(b"\xff\xd8jpegbytes")
    descs = await dehydrate_media([str(src)], inline_max_bytes=1024, put_blob=put)
    dest = tmp_path / "harness_data"
    paths = await rehydrate_media(descs, dest_dir=dest, get_blob=get)
    assert len(paths) == 1
    assert Path(paths[0]).read_bytes() == b"\xff\xd8jpegbytes"
    assert Path(paths[0]).parent == dest


async def test_roundtrip_blob(tmp_path: Path) -> None:
    store, put, get = _mem_blob_store()
    src = tmp_path / "doc.pdf"
    payload = b"%PDF-1.7" + b"z" * 3000
    src.write_bytes(payload)
    descs = await dehydrate_media([str(src)], inline_max_bytes=1024, put_blob=put)
    dest = tmp_path / "out"
    paths = await rehydrate_media(descs, dest_dir=dest, get_blob=get)
    assert Path(paths[0]).read_bytes() == payload
    assert Path(paths[0]).name == "doc.pdf"


async def test_dehydrate_skips_missing_file(tmp_path: Path) -> None:
    store, put, get = _mem_blob_store()
    descs = await dehydrate_media([str(tmp_path / "nope.txt")], inline_max_bytes=1024, put_blob=put)
    assert descs == []


async def test_rehydrate_passes_through_plain_path(tmp_path: Path) -> None:
    # LocalBus / same-disk case: a raw path string must pass through unchanged.
    store, put, get = _mem_blob_store()
    paths = await rehydrate_media(["/already/local/x.png"], dest_dir=tmp_path, get_blob=get)
    assert paths == ["/already/local/x.png"]


async def test_rehydrate_no_name_collision(tmp_path: Path) -> None:
    store, put, get = _mem_blob_store()
    dest = tmp_path / "d"
    dest.mkdir()
    (dest / "a.txt").write_bytes(b"existing")
    src = tmp_path / "a.txt"
    src.write_bytes(b"incoming")
    descs = await dehydrate_media([str(src)], inline_max_bytes=1024, put_blob=put)
    paths = await rehydrate_media(descs, dest_dir=dest, get_blob=get)
    # must not clobber the existing a.txt
    assert Path(paths[0]).name != "a.txt"
    assert Path(paths[0]).read_bytes() == b"incoming"
    assert (dest / "a.txt").read_bytes() == b"existing"


async def test_multiple_media_mixed(tmp_path: Path) -> None:
    store, put, get = _mem_blob_store()
    small = tmp_path / "s.txt"; small.write_bytes(b"hi")
    big = tmp_path / "b.bin"; big.write_bytes(b"y" * 4000)
    descs = await dehydrate_media([str(small), str(big)], inline_max_bytes=1024, put_blob=put)
    assert "inline_b64" in descs[0] and "sha256" in descs[1]
    dest = tmp_path / "out"
    paths = await rehydrate_media(descs, dest_dir=dest, get_blob=get)
    assert Path(paths[0]).read_bytes() == b"hi"
    assert Path(paths[1]).read_bytes() == b"y" * 4000


# ---- media reference contract (harness is the single source of truth) ----
# The harness authors the model-visible "<work_dir>/data/{name}" reference from
# the FINAL on-disk name (post-rehydrate). rehydrate MAY rename on collision;
# the reference must follow the real file, never a channel-frozen name.

def _media_ref(paths: list[str]) -> str:
    # mirrors AgentLoop._media_ref (loop.py)
    return ", ".join(f"<work_dir>/data/{Path(p).name}" for p in paths)


async def test_reference_tracks_rehydrated_name_on_collision(tmp_path: Path) -> None:
    store, put, get = _mem_blob_store()
    dest = tmp_path / "data"
    dest.mkdir()
    (dest / "report.pdf").write_bytes(b"OLD existing")  # pre-existing collision

    src = tmp_path / "report.pdf"
    src.write_bytes(b"NEW incoming bytes")
    descs = await dehydrate_media([str(src)], inline_max_bytes=1, put_blob=put)  # force blob
    assert descs[0]["name"] == "report.pdf"  # descriptor still says the original name

    paths = await rehydrate_media(descs, dest_dir=dest, get_blob=get)
    # rehydrate renamed to avoid clobbering the existing file
    assert Path(paths[0]).name != "report.pdf"
    assert Path(paths[0]).read_bytes() == b"NEW incoming bytes"

    # The harness reference derived from the RETURN VALUE points at the real file;
    # one derived from the descriptor name would dangle. This is the desync the fix kills.
    ref = _media_ref(paths)
    assert Path(paths[0]).name in ref
    assert "report.pdf," not in ref and not ref.endswith("report.pdf")  # not the stale name


async def test_batch_colliding_names_stay_distinct(tmp_path: Path) -> None:
    store, put, get = _mem_blob_store()
    a = tmp_path / "a" / "image.png"; a.parent.mkdir(); a.write_bytes(b"AAAA")
    b = tmp_path / "b" / "image.png"; b.parent.mkdir(); b.write_bytes(b"BBBB")
    descs = await dehydrate_media([str(a), str(b)], inline_max_bytes=1, put_blob=put)
    dest = tmp_path / "data"
    paths = await rehydrate_media(descs, dest_dir=dest, get_blob=get)
    assert len(paths) == 2
    assert paths[0] != paths[1]  # two distinct files, no clobber
    assert Path(paths[0]).read_bytes() == b"AAAA"
    assert Path(paths[1]).read_bytes() == b"BBBB"


def test_media_ref_from_final_paths() -> None:
    ref = _media_ref(["/s/data/a.png", "/s/data/a_1.png"])
    assert ref == "<work_dir>/data/a.png, <work_dir>/data/a_1.png"
