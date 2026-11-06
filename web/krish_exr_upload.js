// Adds a "choose EXR file(s) to upload" button to the Krish EXR Load node, like the upload button on Load Image.
// Files go through ComfyUI's standard /upload/image endpoint into input/krish_exr/, then the path is filled in.
// Pick several numbered frames to upload a whole sequence: the path is set to the first frame and
// sequence_from_file is switched on.
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const NODE = "KrishEXRLoad";
const SUBFOLDER = "krish_exr";
const LABEL = "choose EXR file(s) to upload";

function pickFiles() {
  return new Promise((resolve) => {
    const input = document.createElement("input");
    input.type = "file";
    input.accept = ".exr";
    input.multiple = true;
    input.style.display = "none";
    input.addEventListener("change", () => {
      resolve(Array.from(input.files || []));
      input.remove();
    });
    input.addEventListener("cancel", () => {
      resolve([]);
      input.remove();
    });
    document.body.appendChild(input);
    input.click();
  });
}

async function uploadOne(file) {
  const body = new FormData();
  body.append("image", file, file.name);
  body.append("type", "input");
  body.append("subfolder", SUBFOLDER);
  body.append("overwrite", "true"); // keep the exact name so numbered sequences stay a sequence
  const resp = await api.fetchApi("/upload/image", { method: "POST", body });
  if (resp.status !== 200) throw new Error(`${file.name}: ${resp.status} ${resp.statusText}`);
  const data = await resp.json();
  return data.subfolder ? `${data.subfolder}/${data.name}` : data.name;
}

function naturalSort(a, b) {
  return a.name.localeCompare(b.name, undefined, { numeric: true });
}

app.registerExtension({
  name: "Krish.EXRLoadUpload",
  nodeCreated(node) {
    if (node.comfyClass !== NODE) return;
    const pathWidget = node.widgets?.find((w) => w.name === "path");
    const seqWidget = node.widgets?.find((w) => w.name === "sequence_from_file");
    if (!pathWidget) return;

    const button = node.addWidget(
      "button",
      LABEL,
      null,
      async () => {
        const files = (await pickFiles()).sort(naturalSort);
        if (!files.length) return;
        try {
          const names = [];
          for (let i = 0; i < files.length; i++) {
            button.name = `uploading ${i + 1}/${files.length} ...`;
            app.graph?.setDirtyCanvas(true, true);
            names.push(await uploadOne(files[i]));
          }
          pathWidget.value = names[0];
          if (seqWidget) seqWidget.value = names.length > 1;
          pathWidget.callback?.(pathWidget.value);
        } catch (err) {
          alert(`EXR upload failed: ${err.message || err}`);
        } finally {
          button.name = LABEL;
          app.graph?.setDirtyCanvas(true, true);
        }
      },
      { serialize: false }
    );
    button.serialize = false; // never part of widgets_values, so saved workflows keep their layout

    const size = node.computeSize();
    node.setSize([Math.max(node.size[0], size[0]), Math.max(node.size[1], size[1])]);
  },
});
