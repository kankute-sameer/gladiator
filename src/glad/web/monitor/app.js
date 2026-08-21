// Glad monitor dashboard (slice 5). Read-only consumer of the event bus:
// LIVE mode streams /ws/monitor, REPLAY mode loads a finished run from
// /api/runs/{id}. Same rendering pipeline either way -- every panel is a
// pure function of the in-memory `state.events` array.

const els = {
  modeLive: document.getElementById("mode-live"),
  modeReplay: document.getElementById("mode-replay"),
  runSelect: document.getElementById("run-select"),
  runPicker: document.querySelector(".run-picker"),
  connStatus: document.getElementById("conn-status"),
  runSummary: document.getElementById("run-summary"),
  latencyBody: document.querySelector("#latency-table tbody"),
  timelineSvg: document.getElementById("timeline-svg"),
  timelineLegend: document.getElementById("timeline-legend"),
  timelineEmpty: document.getElementById("timeline-empty"),
  timelineScroll: document.getElementById("timeline-scroll"),
  answersBody: document.querySelector("#answers-table tbody"),
  rosterBody: document.querySelector("#roster-table tbody"),
  statUnderruns: document.getElementById("stat-underruns"),
  statFlush: document.getElementById("stat-flush"),
  statDepth: document.getElementById("stat-depth"),
  depthSvg: document.getElementById("depth-svg"),
  depthEmpty: document.getElementById("depth-empty"),
  eventFilter: document.getElementById("event-filter"),
  eventCount: document.getElementById("event-count"),
  eventLog: document.getElementById("event-log"),
};

const state = {
  mode: "replay", // "live" | "replay"
  events: [],
  runId: null,
  ws: null,
  wsBackoffMs: 500,
  filterText: "",
  renderScheduled: false,
};

const SVG_NS = "http://www.w3.org/2000/svg";

// ---- shared helpers --------------------------------------------------

// Mirrors glad.obs.metrics.percentiles() exactly so numbers here match
// what running that function (or grepping the JSONL by hand) produces.
function percentileAt(sortedValues, pct) {
  if (sortedValues.length === 0) return 0;
  const idx = Math.min(Math.floor(pct * sortedValues.length), sortedValues.length - 1);
  return sortedValues[idx];
}

function fmtMs(v) {
  if (v == null || Number.isNaN(v)) return "--";
  return v >= 1000 ? `${(v / 1000).toFixed(2)}s` : `${v.toFixed(1)}`;
}

function fmtTime(ts) {
  if (!ts) return "--";
  const d = new Date(ts * 1000);
  return `${d.toLocaleTimeString(undefined, { hour12: false })}.${String(d.getMilliseconds()).padStart(3, "0")}`;
}

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}

function setConnStatus(kind, text) {
  els.connStatus.className = `pill pill-${kind}`;
  els.connStatus.textContent = text;
}

function renderRunSummary() {
  if (!state.events.length) {
    els.runSummary.textContent = state.runId ? `${state.runId} · 0 events` : "";
    return;
  }
  const first = state.events[0];
  const last = state.events[state.events.length - 1];
  const span = first.ts && last.ts ? (last.ts - first.ts).toFixed(1) : "?";
  els.runSummary.textContent = `${state.runId || ""} · ${state.events.length} events · ${span}s span`;
}

// ---- panel 1: latency --------------------------------------------------

const METRIC_LABELS = {
  glad_inbound_to_gemini_ms: "meeting audio \u2192 Gemini",
  glad_gemini_first_byte_ms: "Gemini first byte (turn start)",
  glad_gemini_first_byte_ms_grounded: "Gemini first byte \u2014 grounded turns",
  glad_gemini_first_byte_ms_ungrounded: "Gemini first byte \u2014 ungrounded turns",
  glad_gemini_to_outbound_ms: "Gemini reply \u2192 outbound audio",
  glad_tool_roundtrip_ms: "tool call round-trip",
};

function prettifyMetricName(name) {
  return name.replace(/^glad_/, "").replace(/_ms$/, "").replace(/_/g, " ");
}

function renderLatency(events) {
  const byMetric = new Map();
  const push = (key, value) => {
    if (!byMetric.has(key)) byMetric.set(key, []);
    byMetric.get(key).push(value);
  };

  for (const e of events) {
    if (typeof e.t !== "string" || !e.t.startsWith("glad_") || !e.t.endsWith("_ms")) continue;
    if (typeof e.value_ms !== "number") continue;
    push(e.t, e.value_ms);
    // glad_gemini_first_byte_ms carries a `grounded` field rather than
    // being emitted as two separate event types -- split it here so the
    // grounded/ungrounded populations metrics.py tracks in-process are
    // also visible from a replayed JSONL.
    if (e.t === "glad_gemini_first_byte_ms" && typeof e.grounded === "boolean") {
      push(e.grounded ? "glad_gemini_first_byte_ms_grounded" : "glad_gemini_first_byte_ms_ungrounded", e.value_ms);
    }
  }

  const rows = [];
  for (const [metric, values] of byMetric) {
    values.sort((a, b) => a - b);
    rows.push({
      label: METRIC_LABELS[metric] || prettifyMetricName(metric),
      n: values.length,
      p50: percentileAt(values, 0.5),
      p95: percentileAt(values, 0.95),
      p99: percentileAt(values, 0.99),
    });
  }
  rows.sort((a, b) => a.label.localeCompare(b.label));

  els.latencyBody.innerHTML = "";
  if (rows.length === 0) {
    els.latencyBody.innerHTML = '<tr><td colspan="5" style="text-align:center;color:var(--text-dim)">no latency samples yet</td></tr>';
    return;
  }
  const frag = document.createDocumentFragment();
  for (const row of rows) {
    const tr = document.createElement("tr");
    tr.innerHTML = `<td>${escapeHtml(row.label)}</td><td>${row.n}</td><td>${fmtMs(row.p50)}</td><td>${fmtMs(row.p95)}</td><td>${fmtMs(row.p99)}</td>`;
    frag.appendChild(tr);
  }
  els.latencyBody.appendChild(frag);
}

// ---- panel 2: timeline --------------------------------------------------

const TIMELINE_LANES = ["engagement", "turn", "speaker", "tool", "answer", "interrupt", "flush", "wake"];
const CAT_LABELS = {
  engagement: "engagement",
  turn: "turn start",
  speaker: "speaker",
  tool: "tool call",
  answer: "answer",
  interrupt: "interrupt",
  flush: "flush",
  wake: "wake word",
};
const TIMELINE_LEGEND = [
  { color: "#f0c14b", label: "engaged span" },
  { color: "#4fa8ff", label: "turn start" },
  { color: "#9cdcfe", label: "speaker change" },
  { color: "#3fbf6f", label: "tool call" },
  { color: "#c792ea", label: "answer recorded" },
  { color: "#e5566a", label: "interruption" },
  { color: "#e0a83f", label: "flush" },
  { color: "#7ee787", label: "wake word (accepted)" },
  { color: "#4a5058", label: "wake word (suppressed)" },
  { color: "#ff8c42", label: "answer conflicted" },
];

function renderLegend() {
  els.timelineLegend.innerHTML = "";
  for (const item of TIMELINE_LEGEND) {
    const el = document.createElement("span");
    el.className = "legend-item";
    el.innerHTML = `<span class="legend-dot" style="background:${item.color}"></span>${escapeHtml(item.label)}`;
    els.timelineLegend.appendChild(el);
  }
}

function classifyTimelineEvent(e) {
  switch (e.t) {
    case "glad_gemini_first_byte_ms":
      return {
        cat: "turn",
        color: "#4fa8ff",
        title: `turn start \u00b7 first byte ${fmtMs(e.value_ms)}ms${e.grounded ? " \u00b7 grounded" : ""}`,
      };
    case "tool.called":
      return { cat: "tool", color: "#3fbf6f", title: `tool call \u00b7 ${e.name} \u00b7 ${fmtMs(e.latency_ms)}ms` };
    case "answer.recorded":
      return {
        cat: "answer",
        color: "#c792ea",
        title: `answer recorded \u00b7 ${e.question_id} = "${e.value}" (rev ${e.revision})`,
      };
    case "answer.conflicted":
      return {
        cat: "answer",
        color: "#ff8c42",
        title: `answer conflicted \u00b7 ${e.question_id} \u00b7 ${e.previous_participant} \u2192 ${e.new_participant}`,
      };
    case "interrupted":
      return { cat: "interrupt", color: "#e5566a", title: "interruption" };
    case "speaker.changed":
      return {
        cat: "speaker",
        color: "#9cdcfe",
        title: `speaker \u00b7 ${e.name || e.participant_id}${e.rms != null ? ` \u00b7 rms ${Number(e.rms).toFixed(3)}` : ""}`,
      };
    case "flush_sent":
      return { cat: "flush", color: "#e0a83f", title: "flush sent" };
    case "wakeword.stage1_match": {
      const accepted = e.stage2_verdict === "accepted";
      return {
        cat: "wake",
        color: accepted ? "#7ee787" : "#4a5058",
        title: `wake word \u00b7 "${e.phrase}" \u00b7 ${e.stage2_verdict}${e.speaker ? ` \u00b7 ${e.speaker}` : ""} \u00b7 "${e.span}"`,
      };
    }
    default:
      return null;
  }
}

function renderTimeline(events) {
  const items = [];
  for (const e of events) {
    const info = classifyTimelineEvent(e);
    if (info && typeof e.ts === "number") items.push({ ts: e.ts, ...info });
  }
  const hasEngagement = events.some(
    (e) => (e.t === "engagement.opened" || e.t === "engagement.closed") && typeof e.ts === "number"
  );

  const svg = els.timelineSvg;
  const empty = items.length === 0 && !hasEngagement;
  els.timelineEmpty.hidden = !empty;
  if (empty) {
    svg.innerHTML = "";
    return;
  }

  const allTs = events.map((e) => e.ts).filter((t) => typeof t === "number");
  const minTs = Math.min(...allTs);
  const maxTs = Math.max(...allTs);
  const span = Math.max(maxTs - minTs, 1);

  // Fixed user-space size so the SVG never measures the DOM (that feedback
  // loop stretched every panel as the run got longer). CSS maps this to 100%.
  const width = 1000;
  const labelWidth = 92;
  const plotWidth = width - labelWidth - 12;
  const laneHeight = 20;
  const topPad = 22;
  const height = topPad + TIMELINE_LANES.length * laneHeight + 10;

  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  svg.setAttribute("preserveAspectRatio", "none");
  svg.innerHTML = "";

  const tickEvery = span > 900 ? 120 : span > 300 ? 60 : span > 60 ? 15 : 5;
  for (let t = 0; t <= span; t += tickEvery) {
    const x = labelWidth + (t / span) * plotWidth;
    const line = document.createElementNS(SVG_NS, "line");
    line.setAttribute("x1", x);
    line.setAttribute("x2", x);
    line.setAttribute("y1", topPad - 8);
    line.setAttribute("y2", height - 4);
    line.setAttribute("stroke", "#232a33");
    svg.appendChild(line);

    const text = document.createElementNS(SVG_NS, "text");
    text.setAttribute("x", x + 2);
    text.setAttribute("y", topPad - 12);
    text.textContent = span > 120 ? `${Math.round(t / 60)}m` : `${t}s`;
    svg.appendChild(text);
  }

  const laneIndex = Object.fromEntries(TIMELINE_LANES.map((l, i) => [l, i]));
  TIMELINE_LANES.forEach((lane, idx) => {
    const y = topPad + idx * laneHeight + laneHeight / 2;
    const label = document.createElementNS(SVG_NS, "text");
    label.setAttribute("x", 4);
    label.setAttribute("y", y + 3);
    label.textContent = CAT_LABELS[lane];
    svg.appendChild(label);

    const baseline = document.createElementNS(SVG_NS, "line");
    baseline.setAttribute("x1", labelWidth);
    baseline.setAttribute("x2", width - 4);
    baseline.setAttribute("y1", y);
    baseline.setAttribute("y2", y);
    baseline.setAttribute("stroke", "#1c2128");
    svg.appendChild(baseline);
  });

  for (const item of items) {
    const x = labelWidth + ((item.ts - minTs) / span) * plotWidth;
    const y = topPad + laneIndex[item.cat] * laneHeight + laneHeight / 2;
    const circle = document.createElementNS(SVG_NS, "circle");
    circle.setAttribute("cx", x.toFixed(1));
    circle.setAttribute("cy", y);
    circle.setAttribute("r", 4);
    circle.setAttribute("fill", item.color);
    const titleEl = document.createElementNS(SVG_NS, "title");
    titleEl.textContent = `+${(item.ts - minTs).toFixed(1)}s  ${item.title}`;
    circle.appendChild(titleEl);
    svg.appendChild(circle);
  }

  const engY = topPad + laneIndex.engagement * laneHeight + laneHeight / 2;
  let open = null;
  for (const e of events) {
    if (typeof e.ts !== "number") continue;
    if (e.t === "engagement.opened") {
      open = { start: e.ts, trigger: e.trigger || "?" };
    } else if (e.t === "engagement.closed" && open) {
      drawEngagementSpan(svg, open.start, e.ts, minTs, span, labelWidth, plotWidth, engY, open.trigger, e.reason || "?");
      open = null;
    }
  }
  if (open) {
    drawEngagementSpan(svg, open.start, maxTs, minTs, span, labelWidth, plotWidth, engY, open.trigger, "open");
  }
}

function drawEngagementSpan(svg, start, end, minTs, span, labelWidth, plotWidth, y, trigger, reason) {
  const x1 = labelWidth + ((start - minTs) / span) * plotWidth;
  const x2 = labelWidth + ((end - minTs) / span) * plotWidth;
  const rect = document.createElementNS(SVG_NS, "rect");
  rect.setAttribute("x", Math.min(x1, x2).toFixed(1));
  rect.setAttribute("y", y - 6);
  rect.setAttribute("width", Math.max(2, Math.abs(x2 - x1)).toFixed(1));
  rect.setAttribute("height", 12);
  rect.setAttribute("rx", 2);
  rect.setAttribute("fill", "#f0c14b");
  rect.setAttribute("opacity", "0.85");
  const titleEl = document.createElementNS(SVG_NS, "title");
  titleEl.textContent = `ENGAGED \u00b7 opened by ${trigger} \u00b7 closed: ${reason} \u00b7 ${(end - start).toFixed(1)}s`;
  rect.appendChild(titleEl);
  svg.appendChild(rect);
}

// ---- panel 3: answers --------------------------------------------------

function renderAnswers(events) {
  const byQuestion = new Map();
  for (const e of events) {
    if (e.t === "answer.recorded") {
      byQuestion.set(e.question_id, e);
    }
  }
  const rows = [...byQuestion.values()].sort((a, b) => a.question_id.localeCompare(b.question_id));

  els.answersBody.innerHTML = "";
  if (rows.length === 0) {
    els.answersBody.innerHTML = '<tr><td colspan="4" style="text-align:center;color:var(--text-dim)">no answers recorded yet</td></tr>';
    return;
  }
  const frag = document.createDocumentFragment();
  for (const row of rows) {
    const tr = document.createElement("tr");
    if (row.revision > 2) tr.classList.add("flagged");
    tr.innerHTML = `<td>${escapeHtml(row.question_id)}</td><td>${escapeHtml(String(row.value))}</td><td>${row.revision}</td><td>${fmtTime(row.ts)}</td>`;
    frag.appendChild(tr);
  }
  els.answersBody.appendChild(frag);
}

// ---- panel 3b: roster --------------------------------------------------

function renderRoster(events) {
  const byId = new Map();
  for (const e of events) {
    if (e.t === "participant.joined") {
      const existing = byId.get(e.participant_id) || {};
      byId.set(e.participant_id, {
        id: e.participant_id,
        name: e.name || existing.name || "Unknown",
        is_host: e.is_host ?? existing.is_host,
        joined_at: e.joined_at ?? e.ts ?? existing.joined_at,
        left_at: null,
      });
    } else if (e.t === "participant.left") {
      const existing = byId.get(e.participant_id) || {
        id: e.participant_id,
        name: e.name || "Unknown",
        is_host: e.is_host,
        joined_at: e.joined_at,
      };
      existing.name = e.name || existing.name;
      existing.left_at = e.left_at ?? e.ts ?? null;
      byId.set(e.participant_id, existing);
    }
  }
  const rows = [...byId.values()].sort((a, b) => {
    const aPresent = a.left_at == null ? 0 : 1;
    const bPresent = b.left_at == null ? 0 : 1;
    if (aPresent !== bPresent) return aPresent - bPresent;
    return String(a.name).localeCompare(String(b.name));
  });

  els.rosterBody.innerHTML = "";
  if (rows.length === 0) {
    els.rosterBody.innerHTML = '<tr><td colspan="4" style="text-align:center;color:var(--text-dim)">no participants yet</td></tr>';
    return;
  }
  const frag = document.createDocumentFragment();
  for (const row of rows) {
    const tr = document.createElement("tr");
    if (row.left_at != null) tr.classList.add("flagged");
    const host = row.is_host ? "yes" : "";
    tr.innerHTML = `<td>${escapeHtml(String(row.name))}</td><td>${host}</td><td>${fmtTime(row.joined_at)}</td><td>${row.left_at != null ? fmtTime(row.left_at) : "—"}</td>`;
    frag.appendChild(tr);
  }
  els.rosterBody.appendChild(frag);
}

// ---- panel 4: audio health ----------------------------------------------

function computeFlushLatenciesMs(events) {
  const deltas = [];
  let pendingSentMono = null;
  for (const e of events) {
    if (e.t === "flush_sent") {
      pendingSentMono = e.mono;
    } else if (e.t === "flush_ack" && pendingSentMono != null) {
      deltas.push((e.mono - pendingSentMono) * 1000);
      pendingSentMono = null;
    }
  }
  return deltas;
}

function renderAudioHealth(events) {
  const depthSamples = events.filter((e) => e.t === "playback_stats" && typeof e.depth_ms === "number");
  const underrunValues = events.map((e) => e.underruns).filter((v) => typeof v === "number");
  const underruns = underrunValues.length ? Math.max(...underrunValues) : 0;
  const flushDeltas = computeFlushLatenciesMs(events).sort((a, b) => a - b);
  const depthValues = depthSamples.map((e) => e.depth_ms).sort((a, b) => a - b);

  els.statUnderruns.textContent = String(underruns);
  els.statFlush.textContent = flushDeltas.length
    ? `${fmtMs(percentileAt(flushDeltas, 0.5))}ms / ${fmtMs(percentileAt(flushDeltas, 0.95))}ms`
    : "--";
  els.statDepth.textContent = depthValues.length
    ? `${fmtMs(percentileAt(depthValues, 0.5))}ms / ${fmtMs(percentileAt(depthValues, 0.95))}ms`
    : "--";

  const svg = els.depthSvg;
  if (depthSamples.length === 0) {
    svg.innerHTML = "";
    els.depthEmpty.hidden = false;
    return;
  }
  els.depthEmpty.hidden = true;

  const width = 1000;
  const height = 120;
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  svg.setAttribute("preserveAspectRatio", "none");
  svg.innerHTML = "";

  const tsValues = depthSamples.map((e) => e.ts);
  const minTs = Math.min(...tsValues);
  const maxTs = Math.max(...tsValues);
  const span = Math.max(maxTs - minTs, 1);
  const maxDepth = Math.max(...depthSamples.map((e) => e.depth_ms), 100);

  const points = depthSamples
    .map((e) => {
      const x = ((e.ts - minTs) / span) * (width - 8) + 4;
      const y = height - 4 - (Math.min(e.depth_ms, maxDepth) / maxDepth) * (height - 20);
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(" ");

  const polyline = document.createElementNS(SVG_NS, "polyline");
  polyline.setAttribute("points", points);
  polyline.setAttribute("fill", "none");
  polyline.setAttribute("stroke", "#4fa8ff");
  polyline.setAttribute("stroke-width", "1.5");
  svg.appendChild(polyline);

  const label = document.createElementNS(SVG_NS, "text");
  label.setAttribute("x", 4);
  label.setAttribute("y", 12);
  label.setAttribute("fill", "#8892a0");
  label.setAttribute("font-size", "10");
  label.textContent = `max ${Math.round(maxDepth)}ms`;
  svg.appendChild(label);
}

// ---- panel 5: event log --------------------------------------------------

function renderEventLog(events) {
  const filter = state.filterText.trim().toLowerCase();
  const filtered = filter ? events.filter((e) => typeof e.t === "string" && e.t.toLowerCase().includes(filter)) : events;
  els.eventCount.textContent = `${filtered.length} / ${events.length}`;

  const MAX_ROWS = 500;
  const toShow = filtered.slice(-MAX_ROWS).slice().reverse();

  els.eventLog.innerHTML = "";
  const frag = document.createDocumentFragment();
  for (const e of toShow) {
    const row = document.createElement("div");
    row.className = "event-row";

    const ts = document.createElement("span");
    ts.className = "event-ts";
    ts.textContent = fmtTime(e.ts);

    const type = document.createElement("span");
    type.className = "event-type";
    type.textContent = e.t;

    const fields = document.createElement("span");
    fields.className = "event-fields";
    const { ts: _ts, mono: _mono, t: _t, ...rest } = e;
    fields.textContent = Object.keys(rest).length ? JSON.stringify(rest) : "";

    row.appendChild(ts);
    row.appendChild(type);
    row.appendChild(fields);
    frag.appendChild(row);
  }
  els.eventLog.appendChild(frag);
}

// ---- render orchestration ------------------------------------------------

function renderAll() {
  const events = state.events;
  renderLatency(events);
  renderTimeline(events);
  renderAnswers(events);
  renderRoster(events);
  renderAudioHealth(events);
  renderEventLog(events);
  renderRunSummary();
}

function scheduleRender() {
  if (state.renderScheduled) return;
  state.renderScheduled = true;
  requestAnimationFrame(() => {
    state.renderScheduled = false;
    renderAll();
  });
}

// ---- data loading ---------------------------------------------------------

async function fetchRuns() {
  const res = await fetch("/api/runs");
  if (!res.ok) throw new Error(`status ${res.status}`);
  return res.json();
}

async function fetchRun(runId) {
  const res = await fetch(`/api/runs/${encodeURIComponent(runId)}`);
  if (!res.ok) throw new Error(`status ${res.status}`);
  return res.json();
}

function populateRunSelect(runsList) {
  const previous = els.runSelect.value;
  els.runSelect.innerHTML = "";
  for (const run of runsList) {
    const opt = document.createElement("option");
    opt.value = run.id;
    const started = run.started_at ? new Date(run.started_at * 1000).toLocaleString() : run.id;
    const status = run.completed ? "" : " (incomplete)";
    opt.textContent = `${started} \u00b7 ${run.question_set_id || "?"} \u00b7 ${run.event_count} events${status}`;
    els.runSelect.appendChild(opt);
  }
  if (previous && runsList.some((r) => r.id === previous)) {
    els.runSelect.value = previous;
  }
}

async function loadReplay(runId) {
  setConnStatus("replay", "replay");
  try {
    const data = await fetchRun(runId);
    state.runId = runId;
    state.events = data.events;
    renderAll();
  } catch (error) {
    console.error("Failed to load run", runId, error);
    setConnStatus("down", "load failed");
  }
}

function mergeHistorical(historicalEvents) {
  const seen = new Set(state.events.map((e) => e.mono));
  const merged = [...historicalEvents.filter((e) => !seen.has(e.mono)), ...state.events];
  merged.sort((a, b) => (a.mono ?? 0) - (b.mono ?? 0));
  state.events = merged;
  scheduleRender();
}

function connectLive() {
  if (state.ws) {
    state.ws.onclose = null;
    state.ws.close();
  }
  setConnStatus("connecting", "connecting\u2026");
  const scheme = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${scheme}://${location.host}/ws/monitor`);
  state.ws = ws;

  ws.onopen = () => {
    state.wsBackoffMs = 500;
    setConnStatus("live", "live");
  };

  ws.onmessage = (event) => {
    let parsed;
    try {
      parsed = JSON.parse(event.data);
    } catch (error) {
      return;
    }
    if (parsed.t === "run_start") {
      // A new process/run began -- stop splicing a new run's `mono` clock
      // onto the previous run's timeline.
      state.events = [];
      state.runId = null;
    }
    if (parsed.t === "run_meta" && parsed.session_id) {
      state.runId = state.runId || parsed.session_id;
    }
    state.events.push(parsed);
    scheduleRender();
  };

  ws.onclose = () => {
    if (state.ws === ws) state.ws = null;
    if (state.mode !== "live") return;
    setConnStatus("connecting", "reconnecting\u2026");
    setTimeout(() => {
      if (state.mode === "live") startLive(false);
    }, state.wsBackoffMs);
    state.wsBackoffMs = Math.min(state.wsBackoffMs * 2, 5000);
  };

  ws.onerror = () => ws.close();
}

async function startLive(resetEvents = true) {
  setMode("live");
  if (resetEvents) state.events = [];
  connectLive(); // buffering starts immediately; historical events merge in below
  try {
    const runsList = await fetchRuns();
    populateRunSelect(runsList);
    const current = runsList.find((r) => !r.completed) || runsList[0];
    if (current) {
      state.runId = state.runId || current.id;
      const data = await fetchRun(current.id);
      mergeHistorical(data.events);
    }
  } catch (error) {
    console.error("Failed to bootstrap live run", error);
  }
}

function setMode(mode) {
  state.mode = mode;
  els.modeLive.classList.toggle("active", mode === "live");
  els.modeReplay.classList.toggle("active", mode === "replay");
  els.runPicker.style.display = mode === "replay" ? "" : "none";
}

// ---- wiring ---------------------------------------------------------------

els.modeLive.addEventListener("click", () => startLive(true));
els.modeReplay.addEventListener("click", () => {
  if (state.ws) {
    state.ws.onclose = null;
    state.ws.close();
    state.ws = null;
  }
  setMode("replay");
  setConnStatus("replay", "replay");
  if (els.runSelect.value) loadReplay(els.runSelect.value);
});
els.runSelect.addEventListener("change", () => {
  if (state.mode === "replay") loadReplay(els.runSelect.value);
});
els.eventFilter.addEventListener("input", () => {
  state.filterText = els.eventFilter.value;
  renderEventLog(state.events);
});
window.addEventListener("resize", () => scheduleRender());

async function init() {
  renderLegend();
  try {
    const runsList = await fetchRuns();
    populateRunSelect(runsList);
    const current = runsList.find((r) => !r.completed);
    if (current) {
      // No run_stop yet on the most recent run -- treat it as still live.
      await startLive(true);
    } else if (runsList.length > 0) {
      setMode("replay");
      els.runSelect.value = runsList[0].id;
      await loadReplay(runsList[0].id);
    } else {
      setMode("replay");
      setConnStatus("unknown", "no runs yet");
    }
  } catch (error) {
    console.error("Failed to load run list", error);
    setMode("replay");
    setConnStatus("down", "server unreachable");
  }
}

init();
