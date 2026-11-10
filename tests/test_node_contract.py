"""Every node must have a definition ComfyUI can register: a malformed INPUT_TYPES is what produces a red node."""
import importlib.util
import pathlib
import re
import sys

import pytest

ROOT = pathlib.Path(__file__).parents[1]


def _pack():
    spec = importlib.util.spec_from_file_location("krish_pack_contract", ROOT / "__init__.py",
                                                  submodule_search_locations=[str(ROOT)])
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)
    return m


PACK = _pack()
NODES = PACK.NODE_CLASS_MAPPINGS


@pytest.mark.parametrize("name", sorted(NODES))
def test_node_definition_is_well_formed(name):
    cls = NODES[name]
    it = cls.INPUT_TYPES()
    assert "required" in it
    seen = set()
    for section in ("required", "optional"):
        for key, spec in it.get(section, {}).items():
            assert key not in seen, f"{name}: duplicate input {key}"
            seen.add(key)
            assert isinstance(spec, (tuple, list)) and spec, f"{name}.{key}"
            kind = spec[0]
            assert isinstance(kind, str) or (isinstance(kind, list) and kind and all(isinstance(o, str) for o in kind)), f"{name}.{key}"
            opts = spec[1] if len(spec) > 1 else {}
            assert isinstance(opts, dict), f"{name}.{key}"
            if kind in ("INT", "FLOAT"):
                d, lo, hi = opts.get("default"), opts.get("min"), opts.get("max")
                if d is not None and lo is not None:
                    assert d >= lo, f"{name}.{key}: default below min"
                if d is not None and hi is not None:
                    assert d <= hi, f"{name}.{key}: default above max"
            if isinstance(kind, list) and "default" in opts:
                assert opts["default"] in kind, f"{name}.{key}: default not in options"
    assert isinstance(cls.RETURN_TYPES, tuple)
    if hasattr(cls, "RETURN_NAMES"):
        assert len(cls.RETURN_NAMES) == len(cls.RETURN_TYPES)
    assert callable(getattr(cls, cls.FUNCTION))
    assert isinstance(cls.CATEGORY, str)
    assert name in PACK.NODE_DISPLAY_NAME_MAPPINGS


def test_pack_exposes_the_web_extension():
    web = ROOT / PACK.WEB_DIRECTORY
    js = web / "krish_exr_upload.js"
    assert js.is_file()
    text = js.read_text()
    assert '"KrishEXRLoad"' in text and "/upload/image" in text and "serialize" in text
    assert re.search(r'import \{ app \} from "\.\./\.\./scripts/app\.js"', text)


def test_pack_exposes_the_download_extension():
    text = (ROOT / PACK.WEB_DIRECTORY / "krish_exr_download.js").read_text()
    assert '"KrishEXRSave"' in text and "/krish/exr/file" in text and "/krish/exr/zip" in text
    assert "auto_download" in text and "serialize" in text
    assert "auto_download" in NODES["KrishEXRSave"].INPUT_TYPES()["optional"]


def test_pack_reports_its_version_and_nodes(capsys):
    _pack()
    out = capsys.readouterr().out
    assert "Krish-Comfyui-Video 0.6.0" in out and "merge node: yes" in out and "download endpoint:" in out
