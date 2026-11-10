"""Drive the real krish_exr_download.js in Chromium with stubbed ComfyUI modules."""
import glob
import pathlib

import pytest

pw = pytest.importorskip("playwright.sync_api")
CHROME = (glob.glob("/opt/pw-browsers/chromium-*/chrome-linux/chrome") or [None])[0]
if CHROME is None:
    pytest.skip("no Chromium available", allow_module_level=True)

ROOT = pathlib.Path(__file__).parents[1]
BASE = "http://krish.test"
JS = (ROOT / "web" / "krish_exr_download.js").read_text()
APP = "export const app = { extensions: [], registerExtension(e) { this.extensions.push(e); }, graph: { setDirtyCanvas() {} } };"
API = "export const api = { apiURL: (p) => '/api' + p };"
HARNESS = """<!doctype html><body><script type="module">
import { app } from "/scripts/app.js";
await import("/extensions/Krish-Comfyui-Video/krish_exr_download.js");
window.chained = 0;
const mk = (cls, withAuto) => ({ comfyClass: cls, size: [300, 200],
  widgets: [{ name: "mode", value: "replace_rgb" }, { name: "keep_other_channels", value: true }, { name: "note", value: "" }]
    .concat(withAuto ? [{ name: "auto_download", value: false }] : []),
  onExecuted() { window.chained++; },
  addWidget(type, name, value, callback, options) { const w = { type, name, value, callback, options }; this.widgets.push(w); return w; },
  computeSize() { return [320, 400]; }, setSize(s) { this.size = s; } });
window.node = mk("KrishEXRSave", true); window.other = mk("SomethingElse", false);
for (const e of app.extensions) { e.nodeCreated(window.node); e.nodeCreated(window.other); }
window.btn = () => window.node.widgets.find((w) => w.type === "button");
window.ready = true;
</script></body>"""


@pytest.fixture()
def page():
    with pw.sync_playwright() as p:
        browser = p.chromium.launch(executable_path=CHROME, args=["--no-sandbox"])
        pg = browser.new_page()
        requested = []

        def handle(route):
            url = route.request.url.replace(BASE, "")
            if url == "/harness.html":
                return route.fulfill(body=HARNESS, content_type="text/html")
            if url == "/scripts/app.js":
                return route.fulfill(body=APP, content_type="text/javascript")
            if url == "/scripts/api.js":
                return route.fulfill(body=API, content_type="text/javascript")
            if url == "/extensions/Krish-Comfyui-Video/krish_exr_download.js":
                return route.fulfill(body=JS, content_type="text/javascript")
            if url.startswith("/api/krish/exr/"):
                requested.append(url)
                name = "krish_exr_frames.zip" if "/zip" in url else "plate_qwen.1001.exr"
                return route.fulfill(body=b"EXRBYTES-" + name.encode(), content_type="application/octet-stream",
                                     headers={"Content-Disposition": f'attachment; filename="{name}"'})
            return route.fulfill(status=404, body="")

        pg.route(BASE + "/**", handle)
        # Browser downloads bypass request interception, so record the link the extension creates (URL + filename).
        pg.add_init_script("""window.__clicks = []; const o = HTMLAnchorElement.prototype.click;
          HTMLAnchorElement.prototype.click = function () { window.__clicks.push({ href: this.getAttribute('href'), download: this.download }); return o.apply(this, arguments); };""")
        pg.goto(BASE + "/harness.html")
        pg.wait_for_function("window.ready === true")
        yield pg, requested
        browser.close()


ONE = "[{id: 'abc123', name: 'plate_qwen.1001.exr', size: 100, batch: 'b1'}]"
THREE = "[1001, 1002, 1003].map(n => ({id: 'id' + n, name: 'plate_qwen.' + n + '.exr', size: 100, batch: 'b2'}))"


def test_button_is_last_and_never_serialized(page):
    pg, _ = page
    assert pg.evaluate("window.node.widgets.map(w => w.name)") == [
        "mode", "keep_other_channels", "note", "auto_download", "download EXR copy (run the workflow first)"]
    assert pg.evaluate("window.btn().serialize === false && window.btn().options.serialize === false")
    assert pg.evaluate("window.other.widgets.length") == 3                                       # other nodes untouched


def test_clicking_before_any_run_explains_what_to_do(page):
    pg, requested = page
    dialogs = []
    pg.on("dialog", lambda d: (dialogs.append(d.message), d.accept()))
    pg.evaluate("window.btn().callback()")
    pg.wait_for_timeout(100)
    assert dialogs and "Run the workflow first" in dialogs[0] and pg.evaluate("window.__clicks") == []


def test_single_file_downloads_the_exr_from_the_server(page):
    pg, requested = page
    pg.evaluate(f"window.node.onExecuted({{krish_exr: {ONE}}})")
    assert pg.evaluate("window.btn().name") == "download EXR copy (plate_qwen.1001.exr)"
    assert pg.evaluate("window.chained") == 1                                                   # an existing onExecuted still runs
    with pg.expect_download() as d:
        pg.evaluate("window.btn().callback()")
    assert d.value.suggested_filename == "plate_qwen.1001.exr"
    assert pg.evaluate("window.__clicks") == [{"href": "/api/krish/exr/file?id=abc123", "download": "plate_qwen.1001.exr"}]


def test_several_frames_download_as_one_zip(page):
    pg, requested = page
    pg.evaluate(f"window.node.onExecuted({{krish_exr: {THREE}}})")
    assert pg.evaluate("window.btn().name") == "download EXR copies (3 frames, .zip)"
    with pg.expect_download() as d:
        pg.evaluate("window.btn().callback()")
    assert d.value.suggested_filename == "krish_exr_frames.zip"
    assert pg.evaluate("window.__clicks") == [{"href": "/api/krish/exr/zip?batch=b2", "download": "krish_exr_frames.zip"}]


def test_auto_download_downloads_after_each_run_without_a_click(page):
    pg, requested = page
    pg.evaluate("window.node.widgets.find(w => w.name === 'auto_download').value = true")
    with pg.expect_download() as d:
        pg.evaluate(f"window.node.onExecuted({{krish_exr: {ONE}}})")
    assert d.value.suggested_filename == "plate_qwen.1001.exr"
    with pg.expect_download() as d2:
        pg.evaluate(f"window.node.onExecuted({{krish_exr: {THREE}}})")
    assert d2.value.suggested_filename == "krish_exr_frames.zip"
    assert [c["href"] for c in pg.evaluate("window.__clicks")] == ["/api/krish/exr/file?id=abc123", "/api/krish/exr/zip?batch=b2"]


def test_auto_download_off_does_not_download_and_other_messages_are_ignored(page):
    pg, requested = page
    pg.evaluate(f"window.node.onExecuted({{krish_exr: {ONE}}})")
    pg.evaluate("window.node.onExecuted({text: ['hello']})")                                    # a message without our payload
    pg.wait_for_timeout(300)
    assert pg.evaluate("window.__clicks") == []                                                  # auto_download is off: nothing starts
    assert pg.evaluate("window.btn().name") == "download EXR copy (plate_qwen.1001.exr)"          # last real result kept
