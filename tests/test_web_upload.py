"""Drive the real krish_exr_upload.js in Chromium with stubbed ComfyUI modules (no ComfyUI needed)."""
import glob
import json
import pathlib
import re

import pytest

pw = pytest.importorskip("playwright.sync_api")
CHROME = (glob.glob("/opt/pw-browsers/chromium-*/chrome-linux/chrome") or [None])[0]
if CHROME is None:
    pytest.skip("no Chromium available", allow_module_level=True)

ROOT = pathlib.Path(__file__).parents[1]
BASE = "http://krish.test"
JS = (ROOT / "web" / "krish_exr_upload.js").read_text()
APP = "export const app = { extensions: [], registerExtension(e) { this.extensions.push(e); }, graph: { setDirtyCanvas() {} } };"
API = "export const api = { fetchApi: (url, opts) => fetch(url, opts) };"
HARNESS = """<!doctype html><body><button id="go">go</button><script type="module">
import { app } from "/scripts/app.js";
await import("/extensions/Krish-Comfyui-Video/krish_exr_upload.js");
const mk = (cls) => ({ comfyClass: cls, size: [300, 200], widgets: [
  { name: "path", value: "", callback: null }, { name: "sequence_from_file", value: false },
  { name: "first_frame", value: -1 }], 
  addWidget(type, name, value, callback, options) { const w = { type, name, value, callback, options }; this.widgets.push(w); return w; },
  computeSize() { return [320, 320]; }, setSize(s) { this.size = s; } });
window.node = mk("KrishEXRLoad"); window.other = mk("SomethingElse");
for (const e of app.extensions) { e.nodeCreated(window.node); e.nodeCreated(window.other); }
document.getElementById("go").onclick = () => window.node.widgets.find((w) => w.type === "button").callback();
window.ready = true;
</script></body>"""


@pytest.fixture()
def page():
    with pw.sync_playwright() as p:
        browser = p.chromium.launch(executable_path=CHROME, args=["--no-sandbox"])
        pg = browser.new_page()
        uploads, behaviour = [], {"status": 200}

        def handle(route):
            req = route.request
            url = req.url.replace(BASE, "")
            if url == "/harness.html":
                return route.fulfill(body=HARNESS, content_type="text/html")
            if url == "/scripts/app.js":
                return route.fulfill(body=APP, content_type="text/javascript")
            if url == "/scripts/api.js":
                return route.fulfill(body=API, content_type="text/javascript")
            if url == "/extensions/Krish-Comfyui-Video/krish_exr_upload.js":
                return route.fulfill(body=JS, content_type="text/javascript")
            if url == "/upload/image" and req.method == "POST":
                body = req.post_data_buffer.decode("latin1")
                fields = dict(re.findall(r'name="(\w+)"\r\n\r\n([^\r]*)\r\n', body))
                name = re.search(r'filename="([^"]+)"', body).group(1)
                uploads.append({"name": name, **fields})
                if behaviour["status"] != 200:
                    return route.fulfill(status=behaviour["status"], body="boom")
                return route.fulfill(body=json.dumps({"name": name, "subfolder": fields.get("subfolder", ""), "type": "input"}),
                                     content_type="application/json")
            return route.fulfill(status=404, body="")

        pg.route(BASE + "/**", handle)
        pg.goto(BASE + "/harness.html")
        pg.wait_for_function("window.ready === true")
        yield pg, uploads, behaviour
        browser.close()


def _exr(tmp_path, *names):
    files = []
    for n in names:
        f = tmp_path / n
        f.write_bytes(b"\x76\x2f\x31\x01" + n.encode())
        files.append(str(f))
    return files


def test_button_is_added_last_and_never_serialized(page):
    pg, *_ = page
    names = pg.evaluate("window.node.widgets.map(w => w.name)")
    assert names == ["path", "sequence_from_file", "first_frame", "choose EXR file(s) to upload"]   # existing order untouched
    assert pg.evaluate("window.node.widgets[3].serialize === false && window.node.widgets[3].options.serialize === false")
    assert pg.evaluate("window.other.widgets.length") == 3                                          # other nodes untouched


def test_single_file_upload_fills_the_path(page, tmp_path):
    pg, uploads, _ = page
    with pg.expect_file_chooser() as fc:
        pg.click("#go")
    fc.value.set_files(_exr(tmp_path, "TD_005_290_MP001.1001.exr"))
    pg.wait_for_function("window.node.widgets[0].value !== ''")
    assert pg.evaluate("window.node.widgets[0].value") == "krish_exr/TD_005_290_MP001.1001.exr"
    assert pg.evaluate("window.node.widgets[1].value") is False
    assert uploads == [{"name": "TD_005_290_MP001.1001.exr", "type": "input", "subfolder": "krish_exr", "overwrite": "true"}]
    assert pg.evaluate("window.node.widgets[3].name") == "choose EXR file(s) to upload"             # label restored


def test_multiple_numbered_frames_become_a_sequence(page, tmp_path):
    pg, uploads, _ = page
    with pg.expect_file_chooser() as fc:
        pg.click("#go")
    fc.value.set_files(_exr(tmp_path, "plate.1010.exr", "plate.1002.exr", "plate.1001.exr"))
    pg.wait_for_function("window.node.widgets[0].value !== ''")
    assert [u["name"] for u in uploads] == ["plate.1001.exr", "plate.1002.exr", "plate.1010.exr"]    # natural order
    assert pg.evaluate("window.node.widgets[0].value") == "krish_exr/plate.1001.exr"
    assert pg.evaluate("window.node.widgets[1].value") is True                                      # sequence_from_file switched on


def test_failed_upload_keeps_the_old_path_and_tells_the_user(page, tmp_path):
    pg, uploads, behaviour = page
    pg.evaluate("window.node.widgets[0].value = 'keep_me.exr'")
    behaviour["status"] = 500
    dialogs = []
    pg.on("dialog", lambda d: (dialogs.append(d.message), d.accept()))
    with pg.expect_file_chooser() as fc:
        pg.click("#go")
    fc.value.set_files(_exr(tmp_path, "plate.1001.exr"))
    pg.wait_for_function("window.node.widgets[3].name === 'choose EXR file(s) to upload'")
    pg.wait_for_timeout(200)
    assert pg.evaluate("window.node.widgets[0].value") == "keep_me.exr"
    assert dialogs and "EXR upload failed" in dialogs[0]
