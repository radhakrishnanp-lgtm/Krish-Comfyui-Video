"""Let a browser on another machine download the EXR files that Krish EXR Save wrote on the ComfyUI server.

ComfyUI usually runs on a different machine from the one you work on (remote server, port forward). Saved files live on
the server; this module serves a copy to the browser. Only files registered by the Save node can be downloaded, by an
unguessable id, never an arbitrary path, so the endpoint cannot be used to read other files on the server.

No ComfyUI import at module level: add_routes() takes any aiohttp RouteTableDef (unit-testable).
"""
import hashlib
import os
import re
import secrets
import tempfile
import threading
import zipfile

_LOCK = threading.Lock()
_SECRET = secrets.token_bytes(16)
_FILES = {}      # id -> {"path", "name", "size", "batch"}
_BATCHES = {}    # batch -> [id, ...]  (insertion ordered; oldest evicted first)
_MAX_BATCHES = 50
_INSTALLED = False


def _fid(path):
    return hashlib.sha256(_SECRET + os.path.realpath(path).encode()).hexdigest()[:24]


def register(paths):
    """Register freshly written files. Returns (batch_id, entries); entries go to the node's UI message."""
    batch = secrets.token_hex(8)
    entries = []
    with _LOCK:
        for p in paths:
            if not os.path.isfile(p):
                continue
            fid = _fid(p)
            e = {"id": fid, "name": os.path.basename(p), "size": os.path.getsize(p), "batch": batch}
            _FILES[fid] = {**e, "path": os.path.realpath(p)}
            entries.append(e)
        _BATCHES[batch] = [e["id"] for e in entries]
        while len(_BATCHES) > _MAX_BATCHES:
            old = next(iter(_BATCHES))
            for fid in _BATCHES.pop(old):
                if not any(fid in ids for ids in _BATCHES.values()):
                    _FILES.pop(fid, None)
    return batch, entries


def lookup_file(fid):
    with _LOCK:
        e = _FILES.get(str(fid))
    return e if e and os.path.isfile(e["path"]) else None


def batch_files(batch):
    with _LOCK:
        ids = list(_BATCHES.get(str(batch), []))
    return [e for e in (lookup_file(i) for i in ids) if e]


def safe_name(name):
    return re.sub(r"[^A-Za-z0-9._\-]", "_", os.path.basename(name)) or "file"


def build_zip(entries, tmp_dir=None):
    """Zip the given registry entries (stored, not compressed: EXR is already compressed). Returns the temp zip path."""
    fd, zpath = tempfile.mkstemp(suffix=".zip", prefix="krish_exr_", dir=tmp_dir)
    os.close(fd)
    used = set()
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_STORED, allowZip64=True) as z:
        for e in entries:
            name = safe_name(e["name"])
            while name in used:
                name = "_" + name
            used.add(name)
            z.write(e["path"], name)
    return zpath


async def _send(request, path, download_name, delete_after=False):
    from aiohttp import web
    size = os.path.getsize(path)
    resp = web.StreamResponse(headers={
        "Content-Type": "application/octet-stream",
        "Content-Length": str(size),
        "Content-Disposition": f'attachment; filename="{safe_name(download_name)}"',
        "Cache-Control": "no-store",
    })
    await resp.prepare(request)
    try:
        with open(path, "rb") as f:
            while True:
                chunk = f.read(1 << 20)
                if not chunk:
                    break
                await resp.write(chunk)
        await resp.write_eof()
    finally:
        if delete_after and os.path.exists(path):
            os.remove(path)
    return resp


def add_routes(routes, tmp_dir=None):
    """Attach /krish/exr/file and /krish/exr/zip to an aiohttp RouteTableDef."""
    from aiohttp import web

    async def download_file(request):
        e = lookup_file(request.query.get("id", ""))
        if not e:
            return web.json_response({"error": "unknown or expired file id; run the workflow again"}, status=404)
        return await _send(request, e["path"], e["name"])

    async def download_zip(request):
        entries = batch_files(request.query.get("batch", ""))
        if not entries:
            return web.json_response({"error": "unknown or expired batch; run the workflow again"}, status=404)
        zpath = build_zip(entries, tmp_dir)
        return await _send(request, zpath, "krish_exr_frames.zip", delete_after=True)

    routes.get("/krish/exr/file")(download_file)
    routes.get("/krish/exr/zip")(download_zip)


def install_routes():
    """Register the routes on ComfyUI's server when running inside ComfyUI. Safe to call repeatedly or outside ComfyUI."""
    global _INSTALLED
    if _INSTALLED:
        return True
    try:
        from server import PromptServer
        server = getattr(PromptServer, "instance", None)
        if server is None:
            return False
        try:
            import folder_paths
            tmp = folder_paths.get_temp_directory()
            os.makedirs(tmp, exist_ok=True)
        except Exception:
            tmp = None
        add_routes(server.routes, tmp)
        _INSTALLED = True
        return True
    except Exception as e:  # never let the download feature break the nodes
        print(f"[KrishEXR] download endpoint not installed: {e}")
        return False
