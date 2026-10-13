// BT-7274 in the browser: click the eye (or press Space), speak, and BT answers out loud.
//
// The page records the Pilot as 16 kHz WAV, posts it to /api/talk and plays back the
// sentences as they stream in. BT's memory and API key live on the server, on this
// computer; localStorage only holds this browser's settings.

const $ = (id) => document.getElementById(id);

const stage = document.querySelector(".stage");
const eye = $("eye");
const statusLine = $("status");
const heardLine = $("heard");
const caption = $("caption");
const hint = $("hint");
const typeForm = $("type-form");
const typeInput = $("type-input");
const keyboardToggle = $("keyboard-toggle");
const menu = $("menu");
const setupDialog = $("setup");
const setupForm = $("setup-form");
const setupKey = $("setup-key");
const setupError = $("setup-error");

const STATUS = {
  off: "BT-7274 · Offline",
  booting: "BT-7274 · Online",
  idle: "BT-7274 · Standing by",
  listening: "Listening",
  thinking: "Processing",
  speaking: "Transmitting",
  error: "Signal lost",
};

// Listening stops by itself after this much quiet following speech.
const SILENCE_MS = 1300;
const NO_SPEECH_MS = 8000;
const MAX_RECORDING_MS = 60000;
const TARGET_RATE = 16000;

// --- storage (never required: private windows and blocked storage still work) ------

const store = {
  get(key, fallback) {
    try {
      const value = localStorage.getItem(key);
      return value === null ? fallback : JSON.parse(value);
    } catch {
      return fallback;
    }
  },
  set(key, value) {
    try {
      localStorage.setItem(key, JSON.stringify(value));
    } catch {
      /* storage unavailable; keep going in memory */
    }
  },
};

const settings = { radio: false, subtitles: true, ...store.get("bt.settings", {}) };

// --- state ---------------------------------------------------------------------------

let state = "off";

function setState(next, label) {
  state = next;
  stage.dataset.state = next;
  statusLine.textContent = label ?? STATUS[next];
  if (next === "idle") announceDue();
}

function showHeard(text) {
  heardLine.textContent = text;
}

function showCaption(text, { append = false, error = false } = {}) {
  caption.classList.toggle("is-error", error);
  caption.textContent = append && caption.textContent ? `${caption.textContent} ${text}` : text;
}

function clearScreen() {
  showHeard("");
  showCaption("");
}

let failTimer;
function fail(message) {
  setState("error");
  showCaption(message, { error: true, append: Boolean(caption.textContent) && !caption.classList.contains("is-error") });
  clearTimeout(failTimer);
  failTimer = setTimeout(() => state === "error" && setState("idle"), 2400);
}

// --- audio output --------------------------------------------------------------------

let audio = null; // { ctx, gain, analyser }

function audioContext() {
  if (!audio) {
    const ctx = new (window.AudioContext || window.webkitAudioContext)();
    const gain = ctx.createGain();
    const analyser = ctx.createAnalyser();
    analyser.fftSize = 512;
    gain.connect(analyser);
    analyser.connect(ctx.destination);
    audio = { ctx, gain, analyser };
  }
  if (audio.ctx.state === "suspended") audio.ctx.resume();
  return audio.ctx;
}

function setAudioSession(type) {
  // Safari routes sound to the earpiece while the mic is open unless told otherwise.
  try {
    if (navigator.audioSession) navigator.audioSession.type = type;
  } catch {
    /* not supported */
  }
}

const playing = new Set();
const captionTimers = new Set();
let playhead = 0;
let streaming = false;

function decodePcm(base64) {
  const bytes = Uint8Array.from(atob(base64), (char) => char.charCodeAt(0));
  return new Int16Array(bytes.buffer, 0, bytes.byteLength >> 1);
}

function settleIfQuiet() {
  if (!playing.size && !streaming && (state === "speaking" || state === "thinking")) {
    setState("idle");
  }
}

// Queue one sentence right after the previous one and show its text as it starts.
function play(text, base64, rate) {
  const ctx = audioContext();
  const pcm = decodePcm(base64);
  if (!pcm.length) return showCaption(text, { append: true });

  const buffer = ctx.createBuffer(1, pcm.length, rate);
  const channel = buffer.getChannelData(0);
  for (let i = 0; i < pcm.length; i++) channel[i] = pcm[i] / 32768;

  const source = ctx.createBufferSource();
  source.buffer = buffer;
  source.connect(audio.gain);
  const startAt = Math.max(ctx.currentTime + 0.04, playhead);
  source.start(startAt);
  playhead = startAt + buffer.duration;
  playing.add(source);
  source.onended = () => {
    playing.delete(source);
    settleIfQuiet();
  };

  const timer = setTimeout(() => {
    captionTimers.delete(timer);
    if (state === "thinking" || state === "idle" || state === "error") setState("speaking");
    showCaption(text, { append: true });
  }, Math.max(0, (startAt - ctx.currentTime) * 1000));
  captionTimers.add(timer);
}

function hush() {
  for (const source of playing) {
    source.onended = null;
    try {
      source.stop();
    } catch {
      /* already stopped */
    }
  }
  playing.clear();
  for (const timer of captionTimers) clearTimeout(timer);
  captionTimers.clear();
  playhead = 0;
}

// --- talking to the server -----------------------------------------------------------

let request = null; // AbortController for the reply being streamed

function interrupt() {
  request?.abort();
  request = null;
  streaming = false;
  hush();
}

async function* readEvents(response) {
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  const parse = (text) => {
    try {
      return text.trim() ? JSON.parse(text) : null;
    } catch {
      return null; // a garbled line is skipped, the rest of the reply still plays
    }
  };
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let newline;
    while ((newline = buffer.indexOf("\n")) >= 0) {
      const event = parse(buffer.slice(0, newline));
      buffer = buffer.slice(newline + 1);
      if (event) yield event;
    }
  }
  const last = parse(buffer);
  if (last) yield last;
}

function handle(event) {
  switch (event.type) {
    case "heard":
      showHeard(event.text);
      break;
    case "say":
      if (event.audio) play(event.text, event.audio, event.rate);
      else showCaption(event.text, { append: true });
      break;
    case "timer":
      armTimer(event);
      break;
    case "error":
      if (playing.size) showCaption(event.message, { append: true, error: true });
      else fail(event.message);
      break;
  }
}

async function errorMessage(response) {
  try {
    const body = await response.json();
    if (typeof body.detail === "string") return body.detail;
  } catch {
    /* not JSON */
  }
  return `Transmission failed (${response.status}).`;
}

async function transmit({ recording, text }) {
  interrupt();
  clearScreen();
  if (text) showHeard(text);
  setState("thinking");

  const form = new FormData();
  if (recording) form.append("audio", recording, "pilot.wav");
  if (text) form.append("text", text);
  form.append("radio", String(settings.radio));

  const controller = new AbortController();
  request = controller;
  streaming = true;
  try {
    const response = await fetch("api/talk", {
      method: "POST",
      body: form,
      signal: controller.signal,
    });
    if (!response.ok) throw new Error(await errorMessage(response));
    for await (const event of readEvents(response)) {
      if (controller.signal.aborted) return;
      handle(event);
    }
  } catch (error) {
    if (controller.signal.aborted) return;
    // fetch rejects with a TypeError when the server can't be reached at all.
    fail(error instanceof TypeError ? "Can't reach BT. Is bt-web still running, Pilot?" : error.message);
  } finally {
    if (request === controller) {
      request = null;
      streaming = false;
      settleIfQuiet();
    }
  }
}

// --- listening -----------------------------------------------------------------------

// Copies microphone samples off the audio thread.
const TAP_WORKLET = `
class Tap extends AudioWorkletProcessor {
  process(inputs) {
    const channel = inputs[0] && inputs[0][0];
    if (channel) this.port.postMessage(channel.slice(0));
    return true;
  }
}
registerProcessor("tap", Tap);
`;

let workletLoaded = null;
let recording = null;
let starting = false; // waiting on the microphone permission prompt

async function startListening() {
  if (starting || recording) return;
  const ctx = audioContext();
  interrupt();
  clearScreen();
  if (!navigator.mediaDevices?.getUserMedia) {
    return fail("This browser can't reach a microphone. Type instead, Pilot.");
  }

  starting = true;
  try {
    await openMicrophone(ctx);
  } finally {
    starting = false;
  }
}

async function openMicrophone(ctx) {
  setAudioSession("play-and-record");
  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      audio: { echoCancellation: true, noiseSuppression: true, channelCount: 1 },
    });
  } catch {
    setAudioSession("playback");
    return fail("Audio receptors blocked. Allow microphone access, Pilot.");
  }

  try {
    workletLoaded ??= ctx.audioWorklet.addModule(
      URL.createObjectURL(new Blob([TAP_WORKLET], { type: "application/javascript" })),
    );
    await workletLoaded;
  } catch {
    stream.getTracks().forEach((track) => track.stop());
    setAudioSession("playback");
    return fail("This browser can't record audio. Type instead, Pilot.");
  }

  const source = ctx.createMediaStreamSource(stream);
  const tap = new AudioWorkletNode(ctx, "tap");
  const mute = ctx.createGain();
  mute.gain.value = 0;
  source.connect(tap);
  tap.connect(mute); // some browsers only run nodes that reach the output
  mute.connect(ctx.destination);

  const now = performance.now();
  recording = {
    stream, source, tap, mute,
    chunks: [],
    rate: ctx.sampleRate,
    level: 0,
    noise: 0.004,
    loudFrames: 0,
    heardSpeech: false,
    started: now,
    lastVoice: now,
  };
  tap.port.onmessage = (message) => onSamples(message.data);
  clearTimeout(failTimer);
  setState("listening");
}

// Simple voice activity detection: wait for speech, then stop after a pause.
function onSamples(samples) {
  const r = recording;
  if (!r) return;
  r.chunks.push(samples);

  let sum = 0;
  for (let i = 0; i < samples.length; i++) sum += samples[i] * samples[i];
  const rms = Math.sqrt(sum / samples.length);
  r.level = rms;

  const now = performance.now();
  if (!r.heardSpeech) r.noise = r.noise * 0.98 + Math.min(rms, 0.03) * 0.02;
  const threshold = Math.max(0.012, r.noise * 3);
  if (rms > threshold) {
    r.lastVoice = now;
    if (++r.loudFrames > 30) r.heardSpeech = true; // ~80 ms of sound, not a click
  } else {
    r.loudFrames = 0;
  }

  if (r.heardSpeech && now - r.lastVoice > SILENCE_MS) stopListening(true);
  else if (!r.heardSpeech && now - r.started > NO_SPEECH_MS) stopListening(false, "No speech detected, Pilot.");
  else if (now - r.started > MAX_RECORDING_MS) stopListening(true);
}

function stopListening(send, message) {
  const r = recording;
  if (!r) return;
  recording = null;
  r.tap.port.onmessage = null;
  r.source.disconnect();
  r.tap.disconnect();
  r.mute.disconnect();
  r.stream.getTracks().forEach((track) => track.stop());
  setAudioSession("playback");

  // A click before saying anything (under half a second of audio) isn't an order.
  const heardSomething = r.chunks.reduce((sum, chunk) => sum + chunk.length, 0) > r.rate / 2;
  if (send && !heardSomething) {
    send = false;
    message = "No speech detected, Pilot.";
  }

  if (!send) {
    if (message) fail(message);
    else setState("idle");
    return;
  }
  transmit({ recording: encodeWav(r.chunks, r.rate) });
}

// Downsample to 16 kHz mono and wrap it in a WAV header.
function encodeWav(chunks, rate) {
  const total = chunks.reduce((sum, chunk) => sum + chunk.length, 0);
  const input = new Float32Array(total);
  let offset = 0;
  for (const chunk of chunks) {
    input.set(chunk, offset);
    offset += chunk.length;
  }

  const ratio = rate / TARGET_RATE;
  const length = Math.floor(total / ratio);
  const view = new DataView(new ArrayBuffer(44 + length * 2));
  const ascii = (at, text) => [...text].forEach((c, i) => view.setUint8(at + i, c.charCodeAt(0)));
  ascii(0, "RIFF");
  view.setUint32(4, 36 + length * 2, true);
  ascii(8, "WAVE");
  ascii(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true); // PCM
  view.setUint16(22, 1, true); // mono
  view.setUint32(24, TARGET_RATE, true);
  view.setUint32(28, TARGET_RATE * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  ascii(36, "data");
  view.setUint32(40, length * 2, true);

  for (let i = 0; i < length; i++) {
    // Average each window of input samples: cheap low-pass before decimating.
    const from = Math.floor(i * ratio);
    const to = Math.max(from + 1, Math.floor((i + 1) * ratio));
    let sum = 0;
    for (let j = from; j < to; j++) sum += input[j];
    const sample = Math.max(-1, Math.min(1, sum / (to - from)));
    view.setInt16(44 + i * 2, sample * 32767, true);
  }
  return new Blob([view], { type: "audio/wav" });
}

// --- timers --------------------------------------------------------------------------

// Timers that rang while BT was busy wait here, so they never talk over him or the Pilot.
const due = [];

function armTimer(event) {
  setTimeout(() => {
    navigator.vibrate?.([220, 120, 220]);
    if (document.hidden && "Notification" in window && Notification.permission === "granted") {
      new Notification("BT-7274", { body: event.text, icon: "icon-192.png" });
    }
    due.push(event);
    announceDue();
  }, event.seconds * 1000);
  // Ask once, from the moment BT first sets a timer; browsers ignore repeat asks.
  if ("Notification" in window && Notification.permission === "default") {
    Notification.requestPermission().catch(() => {});
  }
}

function announceDue() {
  if (!due.length || (state !== "idle" && state !== "error")) return;
  clearScreen();
  for (const event of due.splice(0)) {
    if (event.audio) play(event.text, event.audio, event.rate);
    else showCaption(event.text, { append: true });
  }
}

// --- the eye's glow follows whoever is talking -----------------------------------------

let level = 0;
let samples = null;

function frame() {
  let target = 0;
  if (recording) {
    target = Math.min(1, recording.level * 9);
  } else if (playing.size && audio) {
    samples ??= new Float32Array(audio.analyser.fftSize);
    audio.analyser.getFloatTimeDomainData(samples);
    let sum = 0;
    for (const sample of samples) sum += sample * sample;
    target = Math.min(1, Math.sqrt(sum / samples.length) * 5);
  }
  level += (target - level) * (target > level ? 0.4 : 0.12);
  eye.style.setProperty("--level", level.toFixed(3));
  requestAnimationFrame(frame);
}

// --- controls ------------------------------------------------------------------------

// What the eye does, by click or by Space: listen, send, or cut BT off.
function pressEye() {
  audioContext(); // browsers only allow sound after a user gesture
  switch (state) {
    case "off":
    case "booting":
      return;
    case "listening":
      return stopListening(true);
    case "thinking":
      return cancelReply();
    default:
      return startListening(); // also cuts BT off if he's mid-sentence
  }
}

function cancelReply() {
  interrupt();
  clearScreen();
  clearTimeout(failTimer);
  setState("idle");
}

eye.addEventListener("click", pressEye);

const dialogOpen = () => Boolean(document.querySelector("dialog[open]"));
const inTextField = (target) =>
  target instanceof Element && Boolean(target.closest("input, textarea, select, [contenteditable]"));

document.addEventListener("keydown", (event) => {
  if (event.defaultPrevented || event.ctrlKey || event.metaKey || event.altKey) return;
  if (dialogOpen()) return; // dialogs have their own Escape and Enter

  if (event.key === "Escape") {
    if (state === "listening") stopListening(false);
    else if (state === "thinking") cancelReply();
    else if (state === "speaking") {
      interrupt();
      setState("idle");
    } else if (document.activeElement === typeInput) typeInput.blur();
    return;
  }

  // Other buttons (and the eye itself) already treat Space as a click.
  const target = event.target;
  const onButton = target instanceof Element && target.closest("button, a, summary");
  if (event.key === " " && !inTextField(target) && !onButton) {
    event.preventDefault(); // no page scroll
    if (!event.repeat) pressEye();
  }
});

function showTypeForm(open) {
  typeForm.hidden = !open;
  keyboardToggle.setAttribute("aria-expanded", String(open));
}

keyboardToggle.addEventListener("click", () => {
  showTypeForm(typeForm.hidden);
  if (!typeForm.hidden) typeInput.focus();
});

typeForm.addEventListener("submit", (event) => {
  event.preventDefault();
  const text = typeInput.value.trim();
  if (!text || state === "off" || state === "booting") return;
  audioContext();
  if (recording) stopListening(false);
  typeInput.value = "";
  transmit({ text });
});

// --- menu ----------------------------------------------------------------------------

let status = null; // the last GET api/status

async function fetchJson(url, options) {
  const response = await fetch(url, options);
  if (!response.ok) throw new Error(await errorMessage(response));
  return response.json();
}

async function refreshStatus() {
  status = await fetchJson("api/status");
  return status;
}

function describeStatus() {
  if (!status) return "No signal.";
  if (!status.brain) return "No API key yet.";
  const ears = status.ears === "whisper" ? "Whisper (local)" : "Gemini";
  return `${status.brain} · Voice ${status.voice ? "online" : "offline"} · Hearing: ${ears}`;
}

function renderMemory(memory) {
  $("facts").replaceChildren(
    ...(memory?.facts ?? []).map((fact) => Object.assign(document.createElement("li"), { textContent: fact })),
  );
  const count = memory?.exchanges;
  $("exchanges").textContent =
    count === undefined ? "Reading memory…" : `${count} past exchange${count === 1 ? "" : "s"} on file.`;
  $("forget").disabled = !memory;
}

function applySettings() {
  document.body.classList.toggle("no-subtitles", !settings.subtitles);
  store.set("bt.settings", settings);
}

async function openMenu() {
  $("opt-radio").checked = settings.radio;
  $("opt-subtitles").checked = settings.subtitles;
  $("menu-brain").textContent = describeStatus();
  renderMemory(null);
  menu.showModal();
  try {
    const [memory] = await Promise.all([fetchJson("api/memory"), refreshStatus().catch(() => {})]);
    $("menu-brain").textContent = describeStatus();
    renderMemory(memory);
  } catch {
    $("exchanges").textContent = "Couldn't read BT's memory.";
  }
}

$("menu-toggle").addEventListener("click", openMenu);

$("opt-radio").addEventListener("change", (event) => {
  settings.radio = event.target.checked;
  applySettings();
});

$("opt-subtitles").addEventListener("change", (event) => {
  settings.subtitles = event.target.checked;
  applySettings();
});

$("forget").addEventListener("click", async () => {
  if (!confirm("Wipe all pilot data? This can't be undone.")) return;
  try {
    await fetchJson("api/memory", { method: "DELETE" });
    renderMemory({ facts: [], exchanges: 0 });
  } catch (error) {
    $("exchanges").textContent = error.message;
  }
});

$("change-key").addEventListener("click", () => {
  menu.close();
  openSetup(false);
});

// Closing a dialog hands focus back to the eye, so Space keeps meaning "talk".
for (const dialog of [menu, setupDialog]) {
  dialog.addEventListener("close", () => eye.focus({ preventScroll: true }));
}

// --- API key setup -------------------------------------------------------------------

let setupRequired = false; // first run: there is nothing to go back to

function openSetup(required) {
  setupRequired = required;
  setupKey.value = "";
  setupError.hidden = true;
  $("setup-cancel").hidden = required;
  $("setup-submit").disabled = false;
  $("setup-submit").textContent = "Link";
  if (!setupDialog.open) setupDialog.showModal();
  setupKey.focus();
}

setupDialog.addEventListener("cancel", (event) => {
  if (setupRequired) event.preventDefault();
});

$("setup-cancel").addEventListener("click", () => setupDialog.close());

setupForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const submit = $("setup-submit");
  submit.disabled = true;
  submit.textContent = "Checking…";
  setupError.hidden = true;
  try {
    const form = new FormData();
    form.append("api_key", setupKey.value);
    await fetchJson("api/setup", { method: "POST", body: form });
    await refreshStatus();
    setupRequired = false;
    setupKey.value = "";
    setupDialog.close();
    if (state === "off") boot();
  } catch (error) {
    setupError.textContent =
      error instanceof TypeError ? "Can't reach BT. Is bt-web still running, Pilot?" : error.message;
    setupError.hidden = false;
    submit.disabled = false;
    submit.textContent = "Link";
    setupKey.select();
  }
});

// --- boot ----------------------------------------------------------------------------

function boot() {
  setState("booting");
  setTimeout(() => state === "booting" && setState("idle"), 1400);
}

async function start() {
  try {
    await refreshStatus();
  } catch {
    setState("off", "BT-7274 · No signal");
    showCaption("Can't reach BT. Is bt-web running? Trying again…", { error: true });
    setTimeout(start, 3000);
    return;
  }
  showCaption("");
  if (status.setup_needed) {
    setState("off", "BT-7274 · Awaiting key");
    openSetup(true);
  } else {
    boot();
  }
}

// Space only means "talk" where there is a keyboard to press it on.
hint.textContent = matchMedia("(pointer: fine)").matches
  ? "Tap the eye or press Space to talk"
  : "Tap the eye to talk";
showTypeForm(matchMedia("(min-width: 720px)").matches); // at a desk, typing is the common case
applySettings();
requestAnimationFrame(frame);
start();
