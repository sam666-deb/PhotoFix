const $ = (id) => document.getElementById(id);

const NAMES = {
  original: "Unedited original",
  natural: "Natural",
  "natural-noguard": "Natural, no guardrail",
  pro: "Pro",
  "pro-noguard": "Pro, no guardrail",
};

let current = null; // the pair on screen
let shownAt = 0; // when both images finished loading
let busy = false;

const storage = {
  get(key) { try { return localStorage.getItem(key); } catch { return null; } },
  set(key, value) { try { localStorage.setItem(key, value); } catch { /* private mode etc. */ } },
};
$("rater").value = storage.get("photofix-rater") ?? "";
$("rater").addEventListener("change", () => storage.set("photofix-rater", $("rater").value.trim()));
const rater = () => $("rater").value.trim() || "anonymous";

function loadImage(img, src) {
  return new Promise((resolve, reject) => {
    img.onload = resolve;
    img.onerror = () => reject(new Error("image failed to load"));
    img.src = src;
  });
}

function setBusy(on) {
  busy = on;
  $("pair").classList.toggle("loading", on);
  for (const b of $("choices").querySelectorAll("button")) b.disabled = on;
}

async function nextPair() {
  setBusy(true);
  try {
    const res = await fetch(`/api/rate/next?rater=${encodeURIComponent(rater())}`);
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || `Server error ${res.status}`);
    await Promise.all([loadImage($("left"), data.left_url), loadImage($("right"), data.right_url)]);
    current = data;
    shownAt = performance.now();
    const { rated_by_you, total_pairs } = data.progress;
    $("progress").textContent = rated_by_you >= total_pairs
      ? `You've rated all ${total_pairs} comparisons. Extra ratings still help.`
      : `${rated_by_you} of ${total_pairs} comparisons rated`;
    $("status").hidden = true;
    $("pair").hidden = false;
    $("choices").hidden = false;
    setBusy(false);
  } catch (err) {
    $("status").hidden = false;
    $("status").textContent = `Couldn't load a comparison: ${err.message}`;
  }
}

async function submit(choice) {
  if (!current || busy) return;
  setBusy(true);
  try {
    const res = await fetch("/api/rate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        photo: current.photo, left: current.left, right: current.right, choice,
        rater: rater(), ms: Math.round(performance.now() - shownAt),
      }),
    });
    if (!res.ok) throw new Error((await res.json()).detail || `Server error ${res.status}`);
  } catch (err) {
    $("status").hidden = false;
    $("status").textContent = `Couldn't save that rating: ${err.message}`;
    setBusy(false);
    return;
  }
  if ($("board-wrap").open) refreshBoard();
  nextPair();
}

function pct(x) { return `${Math.round(x * 100)}%`; }

async function refreshBoard() {
  const res = await fetch("/api/rate/stats");
  if (!res.ok) return;
  const { ratings, variants } = await res.json();
  $("board").replaceChildren(...variants.map((v) => {
    const tr = document.createElement("tr");
    const [lo, hi] = v.ci90 ?? [NaN, NaN];
    const hasCi = Number.isFinite(lo) && Number.isFinite(hi);
    tr.innerHTML = `<td></td>
      <td><div style="display:flex;align-items:center"><div class="ci" style="flex:1">
        <div class="half"></div>
        ${hasCi ? `<div class="range" style="left:${lo * 100}%;width:${Math.max((hi - lo) * 100, 0.5)}%"></div>` : ""}
        <div class="point" style="left:${v.preferred_over_original * 100}%"></div>
      </div><span class="ci-label">${pct(v.preferred_over_original)}${hasCi ? ` (${pct(lo)}–${pct(hi)})` : ""}</span></div></td>
      <td>${v.wins} / ${v.losses} / ${v.ties}</td>`;
    tr.firstElementChild.textContent = NAMES[v.variant] ?? v.variant;
    return tr;
  }));
  $("board-meta").textContent = ratings < 30
    ? `${ratings} ratings so far. Rankings settle after roughly 30–50.`
    : `${ratings} ratings.`;
}

$("choices").addEventListener("click", (e) => {
  const button = e.target.closest("button");
  if (button) submit(button.dataset.choice);
});

document.addEventListener("keydown", (e) => {
  if (e.target.tagName === "INPUT") return;
  const choice = { ArrowLeft: "left", ArrowRight: "right", ArrowDown: "same", x: "both_bad", X: "both_bad" }[e.key];
  if (choice) {
    e.preventDefault();
    submit(choice);
  }
});

$("board-wrap").addEventListener("toggle", () => { if ($("board-wrap").open) refreshBoard(); });

nextPair();
