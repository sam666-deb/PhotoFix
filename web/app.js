const $ = (id) => document.getElementById(id);

const LABELS = {
  underexposure: "Too dark",
  overexposure: "Too bright",
  low_contrast: "Low contrast / haze",
  harsh_shadows: "Harsh shadows",
  color_cast: "Color cast",
  noise: "Noise / grain",
  blur: "Blur / soft focus",
};

let original = null; // ImageBitmap
let enhanced = null; // ImageBitmap
let fileName = "photo";

function severityColor(score) {
  if (score >= 0.5) return "var(--bad)";
  if (score >= 0.2) return "var(--warn)";
  return "var(--ok)";
}

function render() {
  const strength = $("strength").value / 100;
  $("strength-val").textContent = `${$("strength").value}%`;
  const ctx = $("after").getContext("2d");
  ctx.globalAlpha = 1;
  ctx.drawImage(original, 0, 0);
  ctx.globalAlpha = strength;
  ctx.drawImage(enhanced, 0, 0);
  ctx.globalAlpha = 1;
}

function setSplit(pct) {
  $("after").style.clipPath = `inset(0 0 0 ${pct}%)`;
  $("divider").style.left = `${pct}%`;
}

function showReport(data) {
  $("scores").replaceChildren(
    ...Object.entries(data.analysis.scores)
      .sort((a, b) => b[1] - a[1])
      .map(([name, score]) => {
        const li = document.createElement("li");
        li.innerHTML = `<div class="score-row"><span></span><span>${Math.round(score * 100)}%</span></div>
          <div class="bar"><div style="width:${Math.max(score * 100, 2)}%;background:${severityColor(score)}"></div></div>`;
        li.querySelector("span").textContent = LABELS[name] ?? name;
        return li;
      })
  );
  const steps = data.steps.length ? data.steps : ["No edits needed. This photo already looks good."];
  $("steps").replaceChildren(...steps.map((s) => Object.assign(document.createElement("li"), { textContent: s })));
  $("meta").textContent = `${data.width}×${data.height} · processed in ${(data.elapsed_ms / 1000).toFixed(2)} s`;
}

async function handleFile(file) {
  if (!file || !file.type.startsWith("image/")) return;
  fileName = file.name.replace(/\.[^.]+$/, "") || "photo";
  const drop = $("drop");
  drop.classList.add("busy");
  $("drop-text").textContent = "Analyzing and enhancing…";
  $("drop-text").classList.remove("error");

  try {
    const body = new FormData();
    body.append("file", file);
    const [res, orig] = await Promise.all([
      fetch("/api/enhance", { method: "POST", body }),
      createImageBitmap(file, { imageOrientation: "from-image" }),
    ]);
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || `Server error ${res.status}`);

    original = orig;
    enhanced = await createImageBitmap(await (await fetch(data.image)).blob());

    for (const id of ["before", "after"]) {
      $(id).width = original.width;
      $(id).height = original.height;
    }
    $("before").getContext("2d").drawImage(original, 0, 0);
    $("strength").value = 100;
    $("split").value = 50;
    setSplit(50);
    render();
    showReport(data);

    drop.hidden = true;
    $("result").hidden = false;
  } catch (err) {
    $("drop-text").textContent = `Something went wrong: ${err.message}`;
    $("drop-text").classList.add("error");
  } finally {
    drop.classList.remove("busy");
  }
}

$("file").addEventListener("change", (e) => handleFile(e.target.files[0]));

const drop = $("drop");
drop.addEventListener("dragover", (e) => { e.preventDefault(); drop.classList.add("over"); });
drop.addEventListener("dragleave", () => drop.classList.remove("over"));
drop.addEventListener("drop", (e) => {
  e.preventDefault();
  drop.classList.remove("over");
  handleFile(e.dataTransfer.files[0]);
});

$("strength").addEventListener("input", render);
$("split").addEventListener("input", (e) => setSplit(e.target.value));

$("download").addEventListener("click", () => {
  $("after").toBlob((blob) => {
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `${fileName}_photofix.jpg`;
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 1000);
  }, "image/jpeg", 0.95);
});

$("reset").addEventListener("click", () => {
  $("result").hidden = true;
  drop.hidden = false;
  $("file").value = "";
  $("drop-text").textContent = "Drop a photo here, or click to choose";
});
