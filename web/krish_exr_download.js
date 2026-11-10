// Krish EXR Save writes its files on the ComfyUI SERVER. This adds a button that downloads a copy of the files from the
// last run to the machine your browser is on (one file -> the .exr, several frames -> one .zip), and honours the node's
// auto_download switch, which does the same automatically after every run.
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const NODE = "KrishEXRSave";
const LABEL = "download EXR copy (run the workflow first)";

function startDownload(url, name) {
  const a = document.createElement("a");
  a.href = url;
  a.download = name || "";
  a.style.display = "none";
  document.body.appendChild(a);
  a.click();
  a.remove();
}

function downloadEntries(entries) {
  if (!entries || !entries.length) return false;
  if (entries.length === 1) {
    startDownload(api.apiURL(`/krish/exr/file?id=${encodeURIComponent(entries[0].id)}`), entries[0].name);
  } else {
    startDownload(api.apiURL(`/krish/exr/zip?batch=${encodeURIComponent(entries[0].batch)}`), "krish_exr_frames.zip");
  }
  return true;
}

app.registerExtension({
  name: "Krish.EXRSaveDownload",
  nodeCreated(node) {
    if (node.comfyClass !== NODE) return;
    node._krishExr = [];

    const button = node.addWidget(
      "button",
      LABEL,
      null,
      () => {
        if (!downloadEntries(node._krishExr)) {
          alert("Run the workflow first. This button downloads the EXR file(s) that this node saved on the server.");
        }
      },
      { serialize: false }
    );
    button.serialize = false; // never part of widgets_values: saved workflows keep their layout

    const previous = node.onExecuted;
    node.onExecuted = function (message) {
      previous?.apply(this, arguments);
      const entries = message?.krish_exr;
      if (!entries || !entries.length) return;
      node._krishExr = entries;
      button.name =
        entries.length === 1
          ? `download EXR copy (${entries[0].name})`
          : `download EXR copies (${entries.length} frames, .zip)`;
      app.graph?.setDirtyCanvas(true, true);
      const auto = node.widgets?.find((w) => w.name === "auto_download");
      if (auto && auto.value) downloadEntries(entries);
    };

    const size = node.computeSize();
    node.setSize([Math.max(node.size[0], size[0]), Math.max(node.size[1], size[1])]);
  },
});
