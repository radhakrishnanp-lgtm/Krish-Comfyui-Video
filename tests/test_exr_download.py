"""Server side of 'download a copy of what Krish EXR Save wrote': registry, security, routes, zip."""
import asyncio
import importlib.util
import io
import pathlib
import sys
import zipfile

import pytest

aiohttp = pytest.importorskip("aiohttp")
from aiohttp import web  # noqa: E402
from aiohttp.test_utils import TestClient, TestServer  # noqa: E402

ROOT = pathlib.Path(__file__).parents[1]
spec = importlib.util.spec_from_file_location("exr_download", ROOT / "krish_frame_conform" / "exr_download.py")
dl = importlib.util.module_from_spec(spec)
sys.modules["exr_download"] = dl
spec.loader.exec_module(dl)


def files(tmp_path, n=3, prefix="plate"):
    out = []
    for i in range(n):
        p = tmp_path / f"{prefix}.{1001 + i}.exr"
        p.write_bytes(b"\x76\x2f\x31\x01" + bytes([i]) * 5000)
        out.append(str(p))
    return out


def run(coro):
    return asyncio.run(coro)


async def with_client(fn, tmp_dir=None):
    routes = web.RouteTableDef()
    dl.add_routes(routes, tmp_dir)
    app = web.Application()
    app.add_routes(routes)
    async with TestClient(TestServer(app)) as client:
        return await fn(client)


def test_register_returns_ids_and_batch(tmp_path):
    batch, entries = dl.register(files(tmp_path))
    assert len(entries) == 3 and all(e["batch"] == batch and len(e["id"]) == 24 for e in entries)
    assert [e["name"] for e in entries] == ["plate.1001.exr", "plate.1002.exr", "plate.1003.exr"]
    assert all(e["size"] == 5004 for e in entries)
    assert "path" not in entries[0]                                       # the server path is never sent to the browser
    assert dl.register([str(tmp_path / "missing.exr")])[1] == []          # unknown files are ignored


def test_single_file_download_is_exact_and_an_attachment(tmp_path):
    ps = files(tmp_path)
    _, entries = dl.register(ps)

    async def go(c):
        r = await c.get(f"/krish/exr/file?id={entries[1]['id']}")
        return r.status, r.headers, await r.read()

    status, headers, body = run(with_client(go))
    assert status == 200 and body == pathlib.Path(ps[1]).read_bytes()
    assert headers["Content-Disposition"] == 'attachment; filename="plate.1002.exr"'
    assert headers["Content-Length"] == "5004" and headers["Cache-Control"] == "no-store"


def test_zip_download_contains_every_frame_unchanged(tmp_path):
    ps = files(tmp_path, 4)
    batch, _ = dl.register(ps)

    async def go(c):
        r = await c.get(f"/krish/exr/zip?batch={batch}")
        return r.status, r.headers, await r.read()

    status, headers, body = run(with_client(go, str(tmp_path)))
    assert status == 200 and 'filename="krish_exr_frames.zip"' in headers["Content-Disposition"]
    z = zipfile.ZipFile(io.BytesIO(body))
    assert z.namelist() == [pathlib.Path(p).name for p in ps]
    assert all(z.read(pathlib.Path(p).name) == pathlib.Path(p).read_bytes() for p in ps)
    assert not list(tmp_path.glob("krish_exr_*.zip"))                       # temp zip removed after sending


def test_only_registered_files_can_be_downloaded(tmp_path):
    secret = tmp_path / "secret.txt"
    secret.write_text("do not serve")
    dl.register(files(tmp_path))

    async def go(c):
        out = []
        for q in ("id=../../etc/passwd", f"id={secret}", "id=" + "0" * 24, "id=", "", "id=%2e%2e%2f%2e%2e%2fetc%2fpasswd"):
            r = await c.get(f"/krish/exr/file?{q}")
            out.append((r.status, await r.text()))
        r = await c.get("/krish/exr/zip?batch=nope")
        out.append((r.status, await r.text()))
        return out

    for status, text in run(with_client(go)):
        assert status == 404 and "do not serve" not in text and "root:" not in text


def test_download_stops_working_if_the_file_is_removed(tmp_path):
    ps = files(tmp_path, 1)
    _, entries = dl.register(ps)
    pathlib.Path(ps[0]).unlink()

    async def go(c):
        return (await c.get(f"/krish/exr/file?id={entries[0]['id']}")).status

    assert run(with_client(go)) == 404


def test_old_batches_are_forgotten_and_memory_stays_bounded(tmp_path):
    first_batch, first = dl.register(files(tmp_path, 1, prefix="first"))
    for i in range(dl._MAX_BATCHES + 5):
        dl.register(files(tmp_path, 1, prefix=f"later{i}"))
    assert len(dl._BATCHES) <= dl._MAX_BATCHES
    assert first_batch not in dl._BATCHES                                   # the oldest batch is gone
    assert dl.lookup_file(first[0]["id"]) is None                            # and its file can no longer be downloaded


def test_unsafe_names_are_sanitised_in_headers(tmp_path):
    p = tmp_path / 'we"ird\r\nname é.exr'
    p.write_bytes(b"data")
    _, entries = dl.register([str(p)])

    async def go(c):
        r = await c.get(f"/krish/exr/file?id={entries[0]['id']}")
        return r.headers["Content-Disposition"]

    cd = run(with_client(go))
    assert "\r" not in cd and "\n" not in cd and cd.count('"') == 2


def test_install_routes_outside_comfyui_is_a_safe_noop():
    assert dl.install_routes() is False
