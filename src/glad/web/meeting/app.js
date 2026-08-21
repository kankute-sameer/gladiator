// Meeting page: unlock audio, show per-participant level meters, and
// stream server-paced PCM through a jitter-buffered AudioWorklet.
//
// Two things learned getting slice 0's audio working in Recall's bot
// Chromium (see README for the full debugging trail):
//   1. Call getUserMedia() before anything else. The bot auto-grants mic
//      permission, and doing this first is what reliably unlocks audio
//      playback with no user gesture.
//   2. AudioContext.destination is captured as page audio output, with no
//      need for a visible <audio>/<video> element.

const statusEl = document.getElementById("status");
const levelsEl = document.getElementById("levels");
const bufferDepthEl = document.getElementById("buffer-depth");
const questionsEl = document.getElementById("questions");

/** @type {AudioWorkletNode | null} */
let playbackNode = null;
/** @type {WebSocket | null} */
let meetingWs = null;

function setStatus(text) {
  statusEl.textContent = text;
}

function sendControl(message) {
  if (meetingWs && meetingWs.readyState === WebSocket.OPEN) {
    meetingWs.send(JSON.stringify(message));
  }
}

async function initAudio() {
  setStatus("Requesting mic (unlocks audio)…");
  await navigator.mediaDevices.getUserMedia({ audio: true });

  setStatus("Creating audio context…");
  // Must match SAMPLE_RATE in worklets/playback.js -- no resampling
  // happens anywhere in this pipeline.
  const audioContext = new AudioContext({ sampleRate: 24000 });

  setStatus("Loading playback worklet…");
  await audioContext.audioWorklet.addModule("/worklets/playback.js");

  if (audioContext.state === "suspended") {
    setStatus("Resuming audio context…");
    await audioContext.resume();
  }

  if (audioContext.state !== "running") {
    setStatus(`Error: AudioContext stuck ${audioContext.state}`);
    return;
  }

  playbackNode = new AudioWorkletNode(audioContext, "playback", {
    outputChannelCount: [2],
  });
  playbackNode.port.onmessage = (event) => {
    const data = event.data;
    if (data.type === "flushed") {
      latestDepthMs = 0;
      latestUnderruns = data.underruns;
      sendControl({
        t: "flush_ack",
        depth_ms: 0,
        underruns: data.underruns,
        client_mono: performance.now(),
      });
      return;
    }
    latestDepthMs = data.depthMs;
    latestUnderruns = data.underruns;
  };
  playbackNode.connect(audioContext.destination);

  setStatus("Listening for streamed audio…");
}

// --- Level meters --------------------------------------------------------

const rows = new Map(); // participant id -> { root, name, fill }

function getOrCreateRow(id, name) {
  let row = rows.get(id);
  if (row) {
    return row;
  }

  const root = document.createElement("div");
  root.className = "level-row";

  const nameEl = document.createElement("div");
  nameEl.className = "level-name";
  nameEl.textContent = name;

  const track = document.createElement("div");
  track.className = "level-track";

  const fill = document.createElement("div");
  fill.className = "level-fill";
  track.appendChild(fill);

  root.appendChild(nameEl);
  root.appendChild(track);
  levelsEl.appendChild(root);

  row = { root, name: nameEl, fill };
  rows.set(id, row);
  return row;
}

function renderLevels(participants) {
  for (const p of participants) {
    const row = getOrCreateRow(p.id, p.name);
    row.name.textContent = p.name;
    row.fill.style.width = `${Math.round(Math.min(p.rms, 1) * 100)}%`;
  }
}

// --- Discovery questions tile ----------------------------------------------
//
// Rows are built once from `/state/questions` (the current question set plus
// anything already recorded server-side -- covers a page reload mid-call).
// After that, each row updates live off the `answer.recorded` control
// message pushed over /ws/meeting; no polling.

const questionRows = new Map(); // question id -> { root, answer }

function buildQuestionRow(id, text) {
  const root = document.createElement("div");
  root.className = "question-row";

  const textEl = document.createElement("div");
  textEl.className = "question-text";
  textEl.textContent = text;

  const answerEl = document.createElement("div");
  answerEl.className = "question-answer";
  answerEl.textContent = "Pending…";

  root.appendChild(textEl);
  root.appendChild(answerEl);
  questionsEl.appendChild(root);

  const row = { root, answer: answerEl };
  questionRows.set(id, row);
  return row;
}

function setQuestionAnswer(id, value) {
  const row = questionRows.get(id);
  if (!row) {
    return; // unknown id (e.g. stale message from a previous question set)
  }
  row.root.classList.add("answered");
  row.answer.textContent = value;
}

async function loadQuestions() {
  try {
    const response = await fetch("/state/questions");
    if (!response.ok) {
      throw new Error(`status ${response.status}`);
    }
    const data = await response.json();
    for (const question of data.questions) {
      buildQuestionRow(question.id, question.text);
    }
    for (const [id, answer] of Object.entries(data.answers || {})) {
      setQuestionAnswer(id, answer.value);
    }
  } catch (error) {
    console.error("Failed to load discovery questions:", error);
  }
}

// --- Buffer depth readout (5Hz) -------------------------------------------

let latestDepthMs = 0;
let latestUnderruns = 0;

setInterval(() => {
  bufferDepthEl.textContent = `Buffer: ${Math.round(latestDepthMs)}ms · underruns: ${latestUnderruns}`;
  sendControl({
    t: "stats",
    depth_ms: latestDepthMs,
    underruns: latestUnderruns,
  });
}, 200);

// --- Meeting websocket -----------------------------------------------------

function connectMeeting() {
  const scheme = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${scheme}://${location.host}/ws/meeting`);
  ws.binaryType = "arraybuffer";
  meetingWs = ws;
  let backoffMs = 500;

  ws.onopen = () => {
    backoffMs = 500;
  };

  ws.onmessage = (event) => {
    if (event.data instanceof ArrayBuffer) {
      if (playbackNode) {
        playbackNode.port.postMessage({ type: "pcm", buffer: event.data }, [event.data]);
      }
      return;
    }

    const message = JSON.parse(event.data);
    if (message.t === "levels") {
      renderLevels(message.data);
    } else if (message.t === "flush") {
      if (playbackNode) {
        playbackNode.port.postMessage({ type: "flush" });
      }
    } else if (message.t === "answer.recorded") {
      setQuestionAnswer(message.question_id, message.value);
    }
  };

  ws.onclose = () => {
    if (meetingWs === ws) {
      meetingWs = null;
    }
    setTimeout(connectMeeting, backoffMs);
    backoffMs = Math.min(backoffMs * 2, 5000);
  };

  ws.onerror = () => {
    ws.close();
  };
}

async function main() {
  loadQuestions();
  try {
    await initAudio();
  } catch (error) {
    setStatus(`Error: ${error.message}`);
  }
  connectMeeting();
}

main();
