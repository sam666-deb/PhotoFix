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
  const canvas = $("after");
  const ctx = canvas.getContext("2d");
  // Both are drawn at the processed size (smaller than the upload when the server caps resolution).
  ctx.globalAlpha = 1;
  ctx.drawImage(original, 0, 0, canvas.width, canvas.height);
  ctx.globalAlpha = strength;
  ctx.drawImage(enhanced, 0, 0, canvas.width, canvas.height);
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
  const guardNote = data.guardrail?.note;
  $("steps").replaceChildren(...steps.map((s) => Object.assign(document.createElement("li"), {
    textContent: s,
    className: s === guardNote ? "guard" : "",
    title: s === guardNote ? "Safety check: the edit was blended back toward the original to stay natural." : "",
  })));
  const analyzer = data.analysis.source === "dl" ? "neural analyzer" : "classical analyzer";
  $("meta").textContent = `${data.width}×${data.height} · ${analyzer} · ${STYLE_LABELS[data.style]} · ` +
    `processed in ${(data.elapsed_ms / 1000).toFixed(2)} s`;
  const [ow, oh] = data.original_size ?? [data.width, data.height];
  if (ow !== data.width || oh !== data.height) {
    $("meta").textContent += ` · resized from ${ow}×${oh} for the online demo (run locally for full resolution)`;
  }
  $("meta").title = Object.entries(data.timings_ms ?? {}).map(([stage, ms]) => `${stage}: ${ms} ms`).join("\n");
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
      $(id).width = entry.data.width;
      $(id).height = entry.data.height;
    }
    $("before").getContext("2d").drawImage(original, 0, 0, entry.data.width, entry.data.height);
    $("strength").value = 100;
    $("split").value = 50;
    setSplit(50);
    show(entry);

    drop.hidden = true;
    $("drop-text").textContent = "Drop a photo here, or click to choose";
    $("examples").hidden = true;
    $("result").hidden = false;
  } catch (err) {
    $("drop-text").textContent = `Something went wrong: ${err.message}`;
    $("drop-text").classList.add("error");
  } finally {
    drop.classList.remove("busy");
  }
}

$("file").addEventListener("change", (e) => handleFile(e.target.files[0]));

// "Try an example": CC0 photos with a real or clearly labeled simulated problem.
fetch("examples/examples.json")
  .then((r) => (r.ok ? r.json() : []))
  .then((examples) => {
    if (!examples.length) return;
    $("example-list").replaceChildren(...examples.map((ex) => {
      const b = document.createElement("button");
      b.type = "button";
      b.className = "example";
      b.innerHTML = `<img alt="" loading="lazy"><span></span><small></small>`;
      b.querySelector("img").src = `examples/thumbs/${ex.id}.jpg`;
      b.querySelector("span").textContent = ex.label;
      b.querySelector("small").textContent = ex.problem;
      b.title = `${ex.credit.title}, ${ex.credit.author}, ${ex.credit.license}`;
      b.addEventListener("click", async () => {
        const blob = await (await fetch(ex.file)).blob();
        handleFile(new File([blob], `${ex.id}.jpg`, { type: "image/jpeg" }));
        window.scrollTo({ top: 0, behavior: "smooth" });
      });
      return b;
    }));
    $("credits").replaceChildren(...examples.flatMap((ex, i) => {
      const a = Object.assign(document.createElement("a"), { href: ex.credit.source, textContent: ex.credit.author });
      return i ? [", ", a] : [a];
    }));
    $("credits-wrap").hidden = false;
    $("examples").hidden = false;
  })
  .catch(() => {});

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

// Hide styles the server can't offer (e.g. Pro before the learned model is trained), and local-only
// features (rating, feedback) on the public demo.
let isPublic = false;
fetch("/api/health")
  .then((r) => r.json())
  .then(({ styles, public: pub }) => {
    isPublic = pub;
    if (pub) document.querySelector(".rate-link")?.remove();
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

// Implicit feedback: the style and strength someone actually keeps tells us how good the defaults are.
function sendFeedback() {
  const data = results.get(currentStyle)?.data;
  if (!data || isPublic) return;
  fetch("/api/feedback", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      style: currentStyle,
      strength: $("strength").value / 100,
      guard_strength: data.guardrail?.strength ?? 1,
      defects: data.analysis.defects,
      steps: data.steps,
    }),
  }).catch(() => {});
}

$("download").addEventListener("click", () => {
  sendFeedback();
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
  $("examples").hidden = !$("example-list").children.length;
  $("file").value = "";
  $("drop-text").textContent = "Drop a photo here, or click to choose";
});
