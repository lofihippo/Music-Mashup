// Music Mashup frontend.

const state = {
  session: localStorage.getItem("mm_session") || "",
  files: [],           // [{name, size, duration_s, bpm, ...}]
  clips: [],           // editable clip objects
  analyzing: false,
};

const $ = (sel) => document.querySelector(sel);

async function api(path, opts = {}) {
  const headers = { "X-Session-Id": state.session, ...(opts.headers || {}) };
  const res = await fetch(path, { ...opts, headers });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`);
  return data;
}

function sessionId() {
  if (!state.session) {
    state.session = "s" + Math.random().toString(36).slice(2, 12);
    localStorage.setItem("mm_session", state.session);
  }
  return state.session;
}

function setStatus(msg, kind = "") {
  const el = $("#status");
  el.textContent = msg;
  el.className = "status " + kind;
}

// ---- upload ----
$("#uploadBtn").addEventListener("click", async () => {
  const inputs = $("#files").files;
  if (!inputs.length) return;
  setStatus("Uploading...");
  const fd = new FormData();
  for (const f of inputs) fd.append("files", f, f.name);
  try {
    const res = await fetch("/upload", {
      method: "POST",
      headers: { "X-Session-Id": sessionId() },
      body: fd,
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "upload failed");
    state.session = data.session || state.session;
    localStorage.setItem("mm_session", state.session);
    for (const u of data.uploads) {
      state.files.push({ id: u.id, name: u.name, size: u.size });
      state.clips.push({
        source: u.id, name: u.name, start: null, end: null,
        fade_in: 0, fade_out: 0, gain_db: 0, detected_start: null, detected_end: null,
      });
    }
    setStatus("Uploaded " + data.uploads.length + " file(s).");
    renderUi();
    await analyzeAll();
  } catch (e) {
    setStatus("Upload error: " + e.message, "error");
  }
});

async function analyzeAll() {
  state.analyzing = true;
  setStatus("Analyzing...");
  for (const c of state.clips) {
    try {
      const info = await api("/analysis", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ id: c.source }),
      });
      const match = state.files.find((f) => f.id === c.source);
      if (match) { match.duration_s = info.duration_s; match.bpm = info.bpm; }
      c.duration_s = info.duration_s;
      c.bpm = info.bpm;
    } catch (e) { /* leave unknown */ }
  }
  state.analyzing = false;
  renderUi();
  setStatus("Analysis complete.", "ok");
}

// ---- rendering the UI ----
function renderUi() {
  renderUploads();
  renderClips();
}

function renderUploads() {
  const ul = $("#uploadList");
  ul.innerHTML = "";
  for (const f of state.files) {
    const li = document.createElement("li");
    const name = document.createElement("span");
    name.textContent = f.name;
    const meta = document.createElement("span");
    meta.className = "meta";
    meta.textContent = (f.duration_s ? f.duration_s.toFixed(1) + "s" : "?") +
      (f.bpm ? " · " + f.bpm.toFixed(1) + " BPM" : "");
    const rm = document.createElement("button");
    rm.className = "danger";
    rm.textContent = "×";
    rm.addEventListener("click", () => removeFile(f.id));
    li.append(name, meta, rm);
    ul.appendChild(li);
  }
}

function removeFile(id) {
  state.files = state.files.filter((f) => f.id !== id);
  state.clips = state.clips.filter((c) => c.source !== id);
  renderUi();
}

function renderClips() {
  const ul = $("#clipList");
  ul.innerHTML = "";
  $("#clipsEmpty").style.display = state.clips.length ? "none" : "";

  state.clips.forEach((clip, idx) => {
    const li = document.createElement("li");
    li.draggable = true;

    // Track name row (drag handle + filename + meta + remove)
    const head = document.createElement("div");
    head.className = "clip-head";
    const drag = document.createElement("span");
    drag.className = "drag"; drag.textContent = "⠿";
    const file = document.createElement("span");
    file.textContent = clip.name || clip.source;
    file.className = "filename";
    const meta = document.createElement("span");
    meta.className = "meta";
    meta.textContent = clip.bpm ? clip.bpm.toFixed(1) + " BPM · " + (clip.duration_s || 0).toFixed(1) + "s" : (clip.duration_s || 0).toFixed(1) + "s";
    const rm = document.createElement("button");
    rm.className = "danger"; rm.textContent = "×";
    rm.addEventListener("click", () => {
      state.clips.splice(idx, 1);
      renderClips();
    });
    head.append(drag, file, meta, rm);

    // Controls row, on a line below (clean horizontal layout)
    const fields = document.createElement("div");
    fields.className = "clip-fields";
    fields.innerHTML =
      field("Start", "start", idx) + field("End", "end", idx) +
      field("Fade in", "fade_in", idx, 0.1) + field("Fade out", "fade_out", idx, 0.1) +
      field("Gain (dB)", "gain_db", idx, 1);
    fields.querySelectorAll("input").forEach((inp) => {
      inp.addEventListener("input", () => {
        const v = inp.value;
        clip[inp.dataset.key] = v === "" ? null : parseFloat(v);
      });
    });

    li.append(head, fields);
    ul.appendChild(li);
  });

  enableDrag(ul);
}

function field(label, key, idx, step) {
  const v = state.clips[idx][key];
  return `<label>${label}
    <input type="number" step="${step || 0.1}" data-key="${key}"
      value="${v === null || v === undefined ? "" : v}" placeholder="-"></label>`;
}

function enableDrag(ul) {
  let dragged = null;
  ul.querySelectorAll("li[draggable]").forEach((li) => {
    li.addEventListener("dragstart", () => { dragged = li; li.classList.add("dragging"); });
    li.addEventListener("dragend", () => li.classList.remove("dragging"));
    li.addEventListener("dragover", (e) => {
      e.preventDefault();
      const after = getAfter(ul, e.clientY);
      ul.insertBefore(li, after);
    });
  });
}

function getAfter(ul, y) {
  const els = [...ul.querySelectorAll("li:not(.dragging)")];
  return els.find((el) => y < el.getBoundingClientRect().top + el.offsetHeight / 2) || null;
}

// ---- collapsible sections ----
function setupCollapsibles() {
  // Open the workflow-relevant sections by default (add songs + mix options);
  // collapse the (potentially large) track list so the page stays compact.
  const openByDefault = new Set(["sec1", "sec3"]);
  document.querySelectorAll(".collapsible").forEach((card) => {
    const head = card.querySelector(".collapsible-head");
    const body = card.querySelector(".collapsible-body");
    if (!head || !body) return;
    const shouldOpen = openByDefault.has(body.id);
    body.classList.toggle("collapsed", !shouldOpen);
    if (shouldOpen) card.classList.add("open");
    head.addEventListener("click", () => {
      const isOpen = card.classList.toggle("open");
      body.classList.toggle("collapsed", !isOpen);
    });
  });
}
setupCollapsibles();

// ---- detect / render ----
$("#detect").addEventListener("change", async () => {
  await analyzeAll();
});

$("#renderBtn").addEventListener("click", renderMashup);

async function renderMashup() {
  if (!state.clips.length) { setStatus("Add songs first.", "error"); return; }
  setStatus("Rendering...");
  const payload = {
    name: "My Mix",
    crossfade_s: parseFloat($("#crossfade").value) || 0,
    normalize: $("#normalize").checked,
    target_lufs: parseFloat($("#targetLufs").value) || -16,
    // apply current detect to clips with no explicit start/end
    detect: $("#detect").value,
    before_s: parseFloat($("#beforeS").value) || 2.5,
    after_s: parseFloat($("#afterS").value) || 1.0,
    clips: state.clips.map((c) => ({
      source: c.source,
      start: c.start, end: c.end,
      fade_in: c.fade_in || 0, fade_out: c.fade_out || 0, gain_db: c.gain_db || 0,
      detected_start: c.detected_start, detected_end: c.detected_end,
    })),
  };
  // If no explicit cut, let a detect mode fill it in (server may do this too).
  try {
    const meta = await api("/render", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const url = "/audio/" + meta.file;
    $("#player").src = url;
    $("#download").href = url;
    $("#resultCard").hidden = false;
    setStatus("Rendered " + meta.duration_s + "s.", "ok");
  } catch (e) {
    setStatus("Render error: " + e.message, "error");
  }
}
