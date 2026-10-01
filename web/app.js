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

const STYLE_LABELS = { natural: "Natural style", pro: "Pro style (learned from a professional retoucher)" };

let original = null; // ImageBitmap
let enhanced = null; // ImageBitmap
let fileName = "photo";
let currentFile = null;
let currentStyle = "natural";
let results = new Map(); // style -> { data, enhanced } for the current photo, so switching back is instant

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
  const analyzer = data.analysis.source === "dl" ? "neural analyzer" : "classical analyzer";
  $("meta").textContent = `${data.width}×${data.height} · ${analyzer} · ${STYLE_LABELS[data.style]} · ` +
    `processed in ${(data.elapsed_ms / 1000).toFixed(2)} s`;
}

const fromDataUrl = async (url) => createImageBitmap(await (await fetch(url)).blob());

async function fetchStyle(style) {
  if (results.has(style)) return results.get(style);
  const body = new FormData();
  body.append("file", currentFile);
  body.append("style", style);
  const res = await fetch("/api/enhance", { method: "POST", body });
  const data = await res.json();
  if (!res.ok) throw new Error(data.detail || `Server error ${res.status}`);
  // Browsers that can't decode a format (e.g. HEIC in Chrome) get the original from the server instead.
  if (!original && data.original) original = await fromDataUrl(data.original);
  const entry = { data, enhanced: await fromDataUrl(data.image) };
  results.set(style, entry);
  return entry;
}

function show(entry) {
  enhanced = entry.enhanced;
  render();
  showReport(entry.data);
}

function markStyle(style) {
  for (const b of $("styles").querySelectorAll("button")) b.setAttribute("aria-checked", b.dataset.style === style);
}

async function switchStyle(style) {
  if (style === currentStyle) return;
  currentStyle = style;
  markStyle(style);
  if (!currentFile || $("result").hidden) return;
  $("compare").classList.add("loading");
  try {
    const entry = await fetchStyle(style);
    if (style === currentStyle) show(entry); // ignore a slow response if the user already switched again
  } catch (err) {
    $("meta").textContent = `Couldn't apply that style: ${err.message}`;
  } finally {
    if (style === currentStyle) $("compare").classList.remove("loading");
  }
}

async function handleFile(file) {
  if (!file || !(file.type.startsWith("image/") || /\.hei[cf]$/i.test(file.name))) return;
  fileName = file.name.replace(/\.[^.]+$/, "") || "photo";
  currentFile = file;
  results = new Map();
  const drop = $("drop");
  drop.classList.add("busy");
  $("drop-text").textContent = "Analyzing and enhancing…";
  $("drop-text").classList.remove("error");

  try {
    original = await createImageBitmap(file, { imageOrientation: "from-image" }).catch(() => null);
    const entry = await fetchStyle(currentStyle);
    if (!original) throw new Error("This browser can't display that image format.");

    for (const id of ["before", "after"]) {
      $(id).width = original.width;
      $(id).height = original.height;
    }
    $("before").getContext("2d").drawImage(original, 0, 0);
    $("strength").value = 100;
    $("split").value = 50;
    setSplit(50);
    show(entry);

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

$("styles").addEventListener("click", (e) => {
  const button = e.target.closest("button");
  if (button && !button.disabled) switchStyle(button.dataset.style);
});

// Hide styles the server can't offer (e.g. Pro before the learned model is trained).
fetch("/api/health")
  .then((r) => r.json())
  .then(({ styles }) => {
    for (const b of $("styles").querySelectorAll("button")) {
      if (!styles.includes(b.dataset.style)) {
        b.disabled = true;
        b.title = "Not available: the learned model isn't trained on this server.";
      }
    }
  })
  .catch(() => {});

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
