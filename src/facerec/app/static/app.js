"use strict";

// Interface do Reconhecedor Facial (spec 005). Só conversa com a API local (/api/...).
// Nomes vindos do servidor são sempre inseridos com textContent (nunca innerHTML).

const $ = (id) => document.getElementById(id);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

const el = {
  video: $("video"), overlay: $("overlay"), placeholder: $("cameraPlaceholder"),
  btnCamera: $("btnCamera"), cameraMsg: $("cameraMsg"), banner: $("banner"), serverInfo: $("serverInfo"),
  btnRecognize: $("btnRecognize"), btnRecognizeOnce: $("btnRecognizeOnce"), recognizeFile: $("recognizeFile"),
  recStatus: $("recStatus"), recList: $("recList"),
  personName: $("personName"), btnShot: $("btnShot"), btnUpload: $("btnUpload"), enrollFiles: $("enrollFiles"),
  shots: $("shots"), btnEnroll: $("btnEnroll"), btnClearShots: $("btnClearShots"), enrollMsg: $("enrollMsg"),
  personsEmpty: $("personsEmpty"), personsTable: $("personsTable"), personsBody: $("personsBody"),
  personsMsg: $("personsMsg"),
};

const state = {
  stream: null,
  recognizing: false,
  shots: [], // { blob, url }
  maxImages: 10,
  currentTab: "recognize",
  enrolling: false,
};

class ApiError extends Error {
  constructor(code, message, extra = {}) {
    super(message);
    this.code = code;
    this.extra = extra;
  }
}

async function api(path, options) {
  let res;
  try {
    res = await fetch(path, options);
  } catch {
    throw new ApiError("network", "Não foi possível falar com o servidor. Ele está em execução?");
  }
  let body = null;
  try { body = await res.json(); } catch { /* resposta sem JSON */ }
  if (!res.ok) {
    const e = body && body.error ? body.error : { code: "http_" + res.status, message: "Erro inesperado (HTTP " + res.status + ")." };
    throw new ApiError(e.code, e.message, e);
  }
  return body;
}

function setMsg(node, text, kind = "") {
  node.textContent = text;
  node.className = "msg" + (kind ? " " + kind : "");
}

function showBanner(text) {
  el.banner.textContent = text;
  el.banner.classList.toggle("hidden", !text);
}

// ---------------------------------------------------------------- câmera
function cameraMessage(err) {
  switch (err && err.name) {
    case "NotAllowedError":
    case "SecurityError":
      return "Permissão da câmera negada. Libere o acesso no ícone de cadeado da barra de endereço e tente de novo.";
    case "NotFoundError":
    case "OverconstrainedError":
      return "Nenhuma câmera encontrada neste computador.";
    case "NotReadableError":
    case "AbortError":
      return "A câmera está em uso por outro programa. Feche-o e tente de novo.";
    default:
      return "Não foi possível abrir a câmera" + (err && err.name ? " (" + err.name + ")" : "") + ".";
  }
}

function cameraOn() { return !!state.stream; }

function updateControls() {
  const cam = cameraOn();
  el.btnCamera.textContent = cam ? "Desligar câmera" : "Ligar câmera";
  el.btnRecognize.disabled = !cam;
  el.btnRecognize.textContent = state.recognizing ? "Parar reconhecimento" : "Iniciar reconhecimento";
  el.btnShot.disabled = !cam || state.shots.length >= state.maxImages;
  el.btnEnroll.disabled = state.enrolling || state.shots.length === 0;
  el.btnClearShots.disabled = state.shots.length === 0;
  el.placeholder.classList.toggle("hidden", cam);
}

async function startCamera() {
  setMsg(el.cameraMsg, "");
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    setMsg(el.cameraMsg, "Este navegador não dá acesso à câmera nesta página. Abra http://127.0.0.1:8000 ou http://localhost:8000.", "error");
    return;
  }
  try {
    state.stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: "user" },
      audio: false,
    });
  } catch (err) {
    setMsg(el.cameraMsg, cameraMessage(err), "error");
    return;
  }
  el.video.srcObject = state.stream;
  try { await el.video.play(); } catch { /* o autoplay já inicia o vídeo */ }
  updateControls();
}

function stopCamera() {
  stopRecognition();
  if (state.stream) state.stream.getTracks().forEach((t) => t.stop());
  state.stream = null;
  el.video.srcObject = null;
  clearOverlay();
  updateControls();
}

function captureBlob(maxSide, quality) {
  const w = el.video.videoWidth, h = el.video.videoHeight;
  if (!w || !h) return Promise.reject(new Error("A câmera ainda não está pronta."));
  const scale = Math.min(1, maxSide / Math.max(w, h));
  const c = document.createElement("canvas");
  c.width = Math.round(w * scale);
  c.height = Math.round(h * scale);
  c.getContext("2d").drawImage(el.video, 0, 0, c.width, c.height);
  return new Promise((resolve, reject) =>
    c.toBlob((b) => (b ? resolve(b) : reject(new Error("Falha ao capturar a imagem."))), "image/jpeg", quality));
}

// ---------------------------------------------------------------- desenho das caixas
function clearOverlay() {
  const ctx = el.overlay.getContext("2d");
  ctx.clearRect(0, 0, el.overlay.width, el.overlay.height);
}

function drawFaces(result) {
  const cw = el.video.clientWidth, ch = el.video.clientHeight;
  el.overlay.width = cw;
  el.overlay.height = ch;
  const ctx = el.overlay.getContext("2d");
  ctx.clearRect(0, 0, cw, ch);
  const sx = cw / result.image.width, sy = ch / result.image.height;
  ctx.lineWidth = 3;
  ctx.font = "600 16px system-ui, sans-serif";
  ctx.textBaseline = "top";
  for (const f of result.faces) {
    const [x1, y1, x2, y2] = f.box;
    const x = x1 * sx, y = y1 * sy, w = (x2 - x1) * sx, h = (y2 - y1) * sy;
    const color = f.known ? "#2ecc71" : "#f5a623";
    ctx.strokeStyle = color;
    ctx.strokeRect(x, y, w, h);
    const label = f.known ? f.name + " (" + Math.round(f.score * 100) + "%)" : "Desconhecido";
    const tw = ctx.measureText(label).width + 10;
    const ly = y >= 24 ? y - 24 : y + h;
    ctx.fillStyle = color;
    ctx.fillRect(x, ly, tw, 24);
    ctx.fillStyle = "#0b0d12";
    ctx.fillText(label, x + 5, ly + 4);
  }
}

// ---------------------------------------------------------------- reconhecimento
function renderFaceList(result) {
  el.recList.replaceChildren();
  result.faces.forEach((f, i) => {
    const li = document.createElement("li");
    li.className = f.known ? "known" : "unknown";
    if (f.known) li.textContent = "Face " + (i + 1) + ": " + f.name + " (similaridade " + f.score.toFixed(3) + ")";
    else if (f.score === null) li.textContent = "Face " + (i + 1) + ": desconhecido (nenhuma pessoa cadastrada)";
    else li.textContent = "Face " + (i + 1) + ": desconhecido (melhor similaridade " + f.score.toFixed(3) + ", limiar " + result.threshold.toFixed(3) + ")";
    el.recList.appendChild(li);
  });
}

function reportRecognition(result) {
  const n = result.faces.length;
  const total = result.timing_ms.detect + result.timing_ms.embed;
  setMsg(el.recStatus, n + (n === 1 ? " face" : " faces") + " · " + Math.round(total) + " ms no servidor", "ok");
  renderFaceList(result);
}

function reportRecognitionError(err) {
  clearOverlay();
  el.recList.replaceChildren();
  setMsg(el.recStatus, err.message, err.code === "no_face" ? "warn" : "error");
}

async function recognitionLoop() {
  while (state.recognizing) {
    const t0 = performance.now();
    let failed = false;
    try {
      const blob = await captureBlob(640, 0.85);
      const fd = new FormData();
      fd.append("image", blob, "frame.jpg");
      const result = await api("/api/recognize", { method: "POST", body: fd });
      if (!state.recognizing) break;
      drawFaces(result);
      reportRecognition(result);
    } catch (err) {
      if (!state.recognizing) break;
      reportRecognitionError(err);
      if (err.code === "network") { stopRecognition(); break; }
      failed = err.code !== "no_face";
    }
    await sleep(failed ? 1500 : Math.max(0, 250 - (performance.now() - t0)));
  }
}

function startRecognition() {
  if (!cameraOn() || state.recognizing) return;
  state.recognizing = true;
  el.recList.replaceChildren();  // não mostrar resultado de uma sessão anterior
  setMsg(el.recStatus, "Analisando…");
  updateControls();
  recognitionLoop();
}

function stopRecognition() {
  state.recognizing = false;
  updateControls();
}

async function recognizeUploaded(file) {
  if (!file) return;
  stopRecognition();
  clearOverlay();
  setMsg(el.recStatus, "Analisando…");
  const fd = new FormData();
  fd.append("image", file, file.name);
  try {
    const result = await api("/api/recognize", { method: "POST", body: fd });
    reportRecognition(result);
  } catch (err) {
    reportRecognitionError(err);
  }
}

// ---------------------------------------------------------------- cadastro
function renderShots() {
  el.shots.replaceChildren();
  state.shots.forEach((s, i) => {
    const li = document.createElement("li");
    const img = document.createElement("img");
    img.src = s.url;
    img.alt = "Foto " + (i + 1);
    const rm = document.createElement("button");
    rm.type = "button";
    rm.textContent = "×";
    rm.title = "Remover foto";
    rm.addEventListener("click", () => removeShot(i));
    li.append(img, rm);
    el.shots.appendChild(li);
  });
  updateControls();
}

function addShot(blob) {
  if (state.shots.length >= state.maxImages) {
    setMsg(el.enrollMsg, "Limite de " + state.maxImages + " fotos por cadastro.", "warn");
    return;
  }
  state.shots.push({ blob, url: URL.createObjectURL(blob) });
  setMsg(el.enrollMsg, "");
  renderShots();
}

function removeShot(i) {
  const [s] = state.shots.splice(i, 1);
  if (s) URL.revokeObjectURL(s.url);
  renderShots();
}

function clearShots() {
  state.shots.forEach((s) => URL.revokeObjectURL(s.url));
  state.shots = [];
  renderShots();
}

async function takeShot() {
  try {
    addShot(await captureBlob(1280, 0.92));
  } catch (err) {
    setMsg(el.enrollMsg, err.message, "error");
  }
}

async function submitEnroll(confirmDuplicate = false) {
  const name = el.personName.value.trim();
  if (!name) { setMsg(el.enrollMsg, "Informe o nome da pessoa.", "error"); return; }
  if (state.shots.length === 0) { setMsg(el.enrollMsg, "Capture ou envie ao menos uma foto.", "error"); return; }
  const fd = new FormData();
  fd.append("name", name);
  state.shots.forEach((s, i) => fd.append("images", s.blob, "foto" + (i + 1) + ".jpg"));
  fd.append("confirm_duplicate", confirmDuplicate ? "true" : "false");
  state.enrolling = true;
  updateControls();
  setMsg(el.enrollMsg, "Cadastrando…");
  try {
    const r = await api("/api/persons", { method: "POST", body: fd });
    setMsg(el.enrollMsg, "“" + r.person.name + "” cadastrada com " + r.person.n_images + (r.person.n_images === 1 ? " imagem." : " imagens."), "ok");
    el.personName.value = "";
    clearShots();
    refreshPersons();
  } catch (err) {
    if (err.code === "duplicate_person") {
      state.enrolling = false;
      updateControls();
      if (window.confirm(err.message)) { await submitEnroll(true); return; }
      setMsg(el.enrollMsg, "Cadastro cancelado.", "warn");
    } else {
      setMsg(el.enrollMsg, err.message, "error");
    }
  } finally {
    state.enrolling = false;
    updateControls();
  }
}

// ---------------------------------------------------------------- pessoas
async function refreshPersons() {
  try {
    const { persons } = await api("/api/persons");
    el.personsBody.replaceChildren();
    for (const p of persons) {
      const tr = document.createElement("tr");
      const name = document.createElement("td");
      name.textContent = p.name;
      const n = document.createElement("td");
      n.textContent = String(p.n_images);
      const date = document.createElement("td");
      date.textContent = new Date(p.created_at).toLocaleDateString("pt-BR");
      const act = document.createElement("td");
      const del = document.createElement("button");
      del.type = "button";
      del.className = "danger";
      del.textContent = "Excluir";
      del.addEventListener("click", () => deletePerson(p));
      act.appendChild(del);
      tr.append(name, n, date, act);
      el.personsBody.appendChild(tr);
    }
    el.personsEmpty.classList.toggle("hidden", persons.length > 0);
    el.personsTable.classList.toggle("hidden", persons.length === 0);
    refreshHealth();
  } catch (err) {
    setMsg(el.personsMsg, err.message, "error");
  }
}

async function deletePerson(p) {
  if (!window.confirm("Excluir “" + p.name + "” e todos os dados dessa pessoa? Esta ação não pode ser desfeita.")) return;
  try {
    await api("/api/persons/" + p.id, { method: "DELETE" });
    setMsg(el.personsMsg, "“" + p.name + "” excluída.", "ok");
  } catch (err) {
    setMsg(el.personsMsg, err.message, "error");
  }
  refreshPersons();
}

// ---------------------------------------------------------------- geral
async function refreshHealth() {
  try {
    const h = await api("/api/health");
    state.maxImages = h.enroll_max_images || state.maxImages;
    el.serverInfo.textContent = h.persons + (h.persons === 1 ? " pessoa cadastrada" : " pessoas cadastradas") + " · limiar " + h.threshold.toFixed(3);
    showBanner("");
  } catch (err) {
    el.serverInfo.textContent = "Servidor indisponível";
    showBanner(err.message);
  }
}

function showTab(name) {
  state.currentTab = name;
  for (const btn of document.querySelectorAll("nav.tabs button")) {
    btn.setAttribute("aria-selected", String(btn.dataset.tab === name));
  }
  for (const t of ["recognize", "enroll", "persons"]) {
    $("tab-" + t).classList.toggle("hidden", t !== name);
  }
  $("cameraPanel").classList.toggle("hidden", name === "persons");
  if (name !== "recognize") { stopRecognition(); clearOverlay(); }
  if (name === "persons") { setMsg(el.personsMsg, ""); refreshPersons(); }
}

el.btnCamera.addEventListener("click", () => (cameraOn() ? stopCamera() : startCamera()));
el.btnRecognize.addEventListener("click", () => (state.recognizing ? stopRecognition() : startRecognition()));
el.btnRecognizeOnce.addEventListener("click", () => el.recognizeFile.click());
el.recognizeFile.addEventListener("change", () => { recognizeUploaded(el.recognizeFile.files[0]); el.recognizeFile.value = ""; });
el.btnShot.addEventListener("click", takeShot);
el.btnUpload.addEventListener("click", () => el.enrollFiles.click());
el.enrollFiles.addEventListener("change", () => {
  for (const f of el.enrollFiles.files) addShot(f);
  el.enrollFiles.value = "";
});
el.btnEnroll.addEventListener("click", () => submitEnroll(false));
el.btnClearShots.addEventListener("click", clearShots);
for (const btn of document.querySelectorAll("nav.tabs button")) {
  btn.addEventListener("click", () => showTab(btn.dataset.tab));
}
window.addEventListener("beforeunload", () => { if (state.stream) state.stream.getTracks().forEach((t) => t.stop()); });

updateControls();
refreshHealth();
