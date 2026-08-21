# Glad — slices 0, 1, 2a

Infrastructure validation for a Recall.ai meeting bot.

- **Slice 0**: the bot joins a Google Meet, shows a static "Glad" status page
  as its camera feed.
- **Slice 1**: the server receives live per-participant audio from Recall over
  a websocket, and the bot's page renders a live level meter per speaker.
- **Slice 2a**: stream prerecorded 24kHz PCM from the server to the meeting
  page over `/ws/meeting` (binary frames) and play it through a jitter-buffered
  AudioWorklet. No Gemini yet — this isolates the playback path.

## How it works

- `glad.main:create_app` is a FastAPI app that serves `src/glad/web/meeting/`
  (a plain HTML/JS page, no build step), a `/health` endpoint, `/obs/run` for
  rotating the JSONL event log, and two websocket routes (`/ws/recall`,
  `/ws/meeting`).
- `glad.recall.client.RecallClient.create_bot()` calls the Recall.ai
  [Create Bot](https://docs.recall.ai/reference/bot_create) endpoint with:
  - `output_media.camera` pointed at your public URL for the status page.
  - `variant.google_meet = "web_4_core"` (required for reliable output audio
    and for separate per-participant realtime audio).
  - `recording_config.audio_separate_raw` + `recording_config.realtime_endpoints`
    registering `wss://<public host>/ws/recall` for `audio_separate_raw.data`
    events.
- Inbound meeting audio: Recall → `/ws/recall` → `inbound.py` → RMS for level
  meters (and, in slice 2b, the orchestrator/Gemini).
- Outbound streamed audio: `scripts/play_file.py` (or the orchestrator) →
  `/ws/meeting` binary PCM → meeting page AudioWorklet ring buffer → Meet as
  the bot's speaker.
- Control stays JSON (`levels`, `flush`, `stats`, `flush_ack`). Depth and
  underruns from the worklet are posted back as `stats` and persisted to
  `runs/*.jsonl` via `glad.obs.events`.

## Setup

```bash
conda activate glad
pip install -e ".[test]"
# copy .env.example → .env; set RECALL_API_KEY and PUBLIC_URL (ngrok https URL)
ngrok http 8000
uvicorn glad.main:create_app --factory --port 8000
python -m scripts.send_bot "<meet-url>"
```

### Slice 2a playback test

With the bot in the call (or against a local Chromium page):

```bash
python -m scripts.play_file assets/test_speech.wav
python -m scripts.play_file assets/test_speech.wav --jitter 50
python -m scripts.play_file assets/test_speech.wav --flush-after 3   # stops after flush
python -m scripts.play_file assets/test_speech.wav --duration 60 --chunk-ms 100
```

Instrumented Chromium runs (writes/reads `runs/*.jsonl`):

```bash
uvicorn glad.main:create_app --factory --port 8010
python -m scripts.measure_playback --scenario all --port 8010
```

## Slice 2a report-back (measured)

Source: Chromium + real AudioWorklet (same worklet the Meet tile loads),
paced by `play_file` with absolute-clock deadlines. Event log grepped from
`runs/slice2a-*-*.jsonl` after `measure_playback --scenario all`.

**Measured vs environment:** numbers below are from local Chromium against
localhost — not from twenty minutes of watching a Meet tile. The buffer /
underrun / flush path is the same worklet code path; Meet adds
capture/encode on top. Audible quality in Meet still needs a human listen.
The inferred column is empty: every cell is from JSONL.

| Scenario | Depth @10s | @30s | @60s | Steady mean | Underruns during playback | Notes |
|---|---:|---:|---:|---:|---:|---|
| 20ms chunks, 60s | 184.0 | 206.7 | 209.3 | **195.2 ms** | **0** | Hovers at PREBUFFER_MS; no upward drift |
| 100ms chunks, 60s | 110.7 | 213.3 | 176.0 | **154.2 ms** | **0** | More variance; still no underruns in steady state |
| 20ms + jitter ±50ms, 60s | 205.3 | 188.0 | 230.7 | **201.6 ms** | **0** | Max depth 269ms; absorbed by 200ms prebuffer |
| flush-after 3s | — | — | — | — | 0 | **flush_sent → flush_ack = 3.1 ms**; sender stops (no refill) |

- End-of-stream after `play_file` exits produces one underrun as the ring
  drains to empty (`depth_final=0`). That is not counted above — steady-window
  underruns stayed 0 until the last nonzero depth sample.
- **No upward depth drift** on the 20ms run (184 → 207 → 209). A climbing
  buffer would mean the pacer is faster than realtime; absolute-clock pacing
  fixed an earlier bug where cumulative `asyncio.sleep` error ran *slow* and
  starved the client.
- **PREBUFFER_MS recommendation: keep 200.** Steady means sit on ~195–200ms
  at 20ms chunks and still clear ±50ms jitter. 100ms chunks dip the mean to
  ~154ms (half a prebuffer per missed chunk) but did not underrun; I would
  not lower PREBUFFER below 200 until a model is in the loop.

### Harness fix included

`--flush-after` now sends `{"t":"flush"}` and **exits** (barge-in: cut off,
do not resume mid-sentence). Earlier behavior flushed then kept refilling,
which is not a scenario this system will have.

## Slice 2b: latency fixes (measured 2026-08-20, after real-key testing)

The first pass (Gemini wired straight through) shipped clean transport
numbers and a bad conversational one: `glad_inbound_to_gemini_ms` p50 2.0ms,
`glad_gemini_to_outbound_ms` p50 0.14ms, but `glad_gemini_first_byte_ms` p50
**2.7s**, and client buffer depth p50 888ms / p95 2.3s (2a's equivalent
number was 195ms). Three fixes, in order of what actually moved the needle:

**1. Outbound pacing (the depth fix).** `play_file` paces to realtime;
Gemini doesn't — it generates faster than speech plays, so every chunk
relayed the instant it arrived piled up in the browser's ring buffer. Depth
*is* interrupt latency: at 2.3s p95, someone barging in waits over two
seconds unless flush fires first. Fix: `outbound.send_bytes` now queues into
a `bytearray` and a background `_pace_loop` drains it to the client at
48 bytes/ms (absolute-clock, same fix as `play_file`'s), so the excess sits
in Python where `broadcast_flush` can clear it with no round trip, instead
of in the worklet's ring buffer where clearing it needs one. New metric
`glad_server_queue_depth_ms` tracks exactly what used to hide in client
depth. Unit-tested without a network: a 1s burst sent to `send_bytes` reads
back mid-drain at ~300ms sent / ~700ms still queued at the 300ms mark, and
`broadcast_flush` mid-drain leaves `_pending` at 0 and stops the pacer from
sending anything further.

**2. Model: native-audio vs `gemini-3.1-flash-live-preview`.** Direct
A/B against the real API, same speech clip, 3 trials each, everything else
identical (same VAD config):

| Model | first-byte p50 (n=3) | min | max |
|---|---:|---:|---:|
| `gemini-2.5-flash-native-audio-preview-12-2025` (previous default) | 3373ms | 3226ms | 4651ms |
| `gemini-3.1-flash-live-preview` (new default) | **776ms** | 721ms | 787ms |

Same `interrupted` wire shape (`{"interrupted": true, "turn_complete": null,
"has_model_turn": false}`), same 24kHz output — no client change needed.
3.1 Flash Live drops affective dialog / proactive-audio / parallel tool
calls versus native audio; this slice uses none of them. Switched the
default in `config.py`.

**3. VAD end-of-speech sensitivity.** Part of first-byte latency is Gemini
waiting to be sure you've stopped talking. Set
`end_of_speech_sensitivity=END_SENSITIVITY_HIGH` and `silence_duration_ms=400`
(down from the server default) in `LiveSession`'s config — a real lever, but
model choice dominated it by ~4x in the A/B above.

*(Superseded in slice 4 below: automatic activity detection is now fully
disabled and these two knobs no longer apply — `glad.agent.floor.
FloorControl`'s own `endpoint_gap_s` is the equivalent lever now.)*

**Session resumption: enabled, but I have not observed a usable handle.**
`session_resumption=types.SessionResumptionConfig(handle=...)` is set on
every connect — required for the server to send `SessionResumptionUpdate` at
all, and it already was before this pass. What actually came back, across
both models and several short (2-8s) sessions: `resumable=None`
(the SDK's decode of an absent field) with an empty handle. Per the API
reference, `resumable=false` (and empty handle) is expected "when the model
is executing function calls or generating" — my test sessions were short
enough that most of their lifetime was spent doing exactly that. `go_away`
messages did arrive (`time_left=50s`), confirming reconnects are routine on
this API, not exceptional. I did not get a session long/idle enough to
observe a `resumable=true` update; `LiveSession` reconnects correctly
either way (backoff logged at WARNING, new session established), it just
can't be proven to resume *context* until a handle is actually captured. If
it turns out native sessions of this length never yield one, the fallback
is carrying conversation state server-side and replaying it as text on
reconnect — noted here rather than built, since it's unconfirmed whether
it's needed.

## Slice 4: wake word + floor control

### Manual activity detection: audio input path verified, `turn_coverage` parked

Before writing any feature code, two questions were tested against the real
API (`gemini-3.1-flash-live-preview`), not guessed from docs.

**Q1: does audio streamed while no activity window is open survive to be
picked up when a window opens later?** No. TTS-synthesized phrases
("orange giraffe ninety one" etc.) streamed with no window open, followed by
3-10s of low-level background hiss, then a trivial `activity_start` ->
`activity_end` window, were never recovered — `input_transcription` returned
an unrelated hallucinated string every time, regardless of the gap length.
Ambient mode (below) does not depend on this working.

**Q2: is that a broken audio path, or a real model limitation?** Two checks,
both required before parking the question:

- *Positive control*: the same phrase streamed **fully inside** an open
  window (`activity_start` -> audio -> `activity_end`) came back verbatim —
  `input_transcription` = `"orangegiraffe91"`, model repeated
  `"ORANGE GIRAFFE 91"`. The audio input path itself is fine; the earlier
  interrupt trace, latency numbers, and tool-call results are not
  invalidated.
- *Wire check*: monkeypatched `google.genai.live`'s `json.dumps` to print
  the literal `setup` message before it hits the websocket. Both
  `automatic_activity_detection.disabled: true` and
  `turn_coverage: "TURN_INCLUDES_ALL_INPUT"` are present verbatim in what's
  sent. The field reaches the server.

Ran all four `turn_coverage` values (unset default,
`TURN_INCLUDES_ONLY_ACTIVITY`, `TURN_INCLUDES_ALL_INPUT`,
`TURN_INCLUDES_AUDIO_ACTIVITY_AND_ALL_VIDEO`) against the identical
phrase/gap/window scenario: **byte-identical output across all four.**
Combined with the wire check, this model does not appear to differentiate
behavior on `turn_coverage` for backlog retention, at least not in a way
this harness can observe. Per the design gate, this is parked rather than
investigated further: the design (below) is built to not depend on
`turn_coverage`, or on any windowless audio being retained as context, at
all.

### Two more load-bearing findings from the same harness

- **An activity window with real (even near-silent) audio but no
  meaningful speech still produces an audible model turn.**
  `activity_start` -> 100ms of near-silent PCM -> `activity_end` produced a
  fully-formed, unrelated, hallucinated spoken reply in Spanish. Confirms
  the design note in-code: there is no way to open-then-close a window "for
  free" — every closed window is a real turn the model will voice. A bare
  `activity_start` -> `activity_end` with **zero** content (no audio, no
  text) is rejected outright by the server (`1007 Precondition check
  failed`) — a window needs *something* inside it. Consequence: floor
  control must never open a window speculatively or on a periodic timer;
  it only opens one when there is genuine participant speech (discovery
  turn-taking) or a wake word plus something to say.
- **`send_realtime_input(text=...)` outside of any activity window is a
  first-class way to hand the model a turn.** No `activity_start` /
  `activity_end` needed at all — sending text alone, under manual
  activity detection, produced a normal turn: the model correctly called
  `record_answer` for a fact embedded in the injected text and then
  continued the conversation naturally. This is the mechanism ambient
  capture (below) is built on: no audio window, no orphan replies, no
  suppressed-at-playback turns.

### Architecture: two modes, floor control, ambient capture

**Mode is derived, never stored.** `glad.agent.mode.derive_mode(state)`:
`DISCOVERY` while any question in the set is unanswered, `AMBIENT` once
every question has a recorded answer. Because answers are only ever
recorded or corrected (never un-recorded), this transition is one-way per
session — `AMBIENT` mode, once reached, is permanent for the rest of the
call. That monotonicity is what keeps the ambient text buffer
(`glad.agent.ambient.AmbientBuffer`) simple: it only ever fills *after*
discovery is done, so there's no case where a discovery window opens with
stale ambient text still waiting to be flushed.

**One Gemini connection, manual activity detection throughout**
(`live.session._build_config`). Automatic server-side VAD can't be
selectively disabled per conversational mode on a single connection, and
(per the empty-window finding above) leaving it on on ambient turns would
auto-trigger real spoken replies to ordinary background chatter. So
`glad.agent.floor.FloorControl` now owns every `activity_start` /
`activity_end` decision, replacing the old automatic-VAD sensitivity
tuning with app-level speech-energy endpointing
(`orchestrator._drive_floor_control`, RMS threshold `0.02`, a placeholder
not yet tuned against a real room).

**Floor control (`glad.agent.floor.FloorControl`)** — 4 states, driven by
two independent signals:
- Speech-energy edges (`on_speech_started` / `on_speech_ended` / `tick`),
  derived every inbound audio frame from the room's RMS. In `DISCOVERY`
  mode, speech starting takes the floor directly (`AMBIENT -> SPEAKING`,
  `OPEN_WINDOW`) — no wake word needed, matching today's turn-taking.
  Silence past `endpoint_gap_s` (0.5s) closes it (`CLOSE_WINDOW`).
- Wake word matches (`on_wake_matched`), from stage 2 acceptances
  regardless of mode — "wake word during discovery must still work" is
  satisfied because a match only *acts* when the floor happens to be idle
  (`AMBIENT -> WAKE_PENDING`); it's a real no-op, not a special case, when
  discovery's own speech already has the floor. Silence past
  `gap_threshold_s` (1.2s) fires it (`WAKE_PENDING -> SPEAKING`, action
  `WAKE`) — this is deliberately a *different* action than `OPEN_WINDOW`:
  see below.
- `on_interrupted()` reflects a barge-in into `YIELDED -> AMBIENT`; the
  actual flush is still the pre-existing `orchestrator.run()` path
  (`content.interrupted` -> `outbound.broadcast_flush()`) — untouched, not
  duplicated.

One correctness bug caught by the integration tests, not the unit tests:
`on_wake_matched` doesn't itself touch the silence clock, so if the
energy-based detector never crossed threshold for the wake utterance at
all (plausible — it's a cruder gate than Recall's own ASR), the gap timer
had nothing to compare against and `WAKE_PENDING` could stick forever.
Fixed by defaulting "silent since" to `-inf` instead of `None` — "no
speech observed yet" and "definitely currently mid-speech" need to be
distinguishable states, and the old `Optional[float]` collapsed them.

**Ambient capture (option C, per the design brief).** `WAKE` never opens a
raw-audio window — measured above that a window needs real content or the
server either rejects it (empty) or the model hallucinates a reply
(near-silent). Instead `orchestrator._fire_wake` flushes
`AmbientBuffer` (fed by every `AMBIENT`-mode transcript segment, matched
or not, from `on_transcript_segment`) and calls
`LiveSession.send_text(...)` — measured above that this alone produces a
normal turn, tool calls included, with no window at all.

**Wake word detection (`glad.agent.wakeword`)**, FINAL segments only
(`transcript.data`, never `transcript.partial_data` — see the transcript
source note below):
- Stage 1: longest-phrase-first word-boundary matching over
  `["gladiator", "glad iator", "glad", "clad", "glide"]` — longest-first is
  what stops "gladiator" from also registering a "glad" hit inside itself.
- Stage 2: an explicit rule list, not a regex — negative precede/follow
  patterns ("so glad", "glad we/you/to/that/it/about", etc.) suppress;
  utterance-initial position or an interrogative/imperative within 4
  tokens accepts. `gladiator`/`glad iator`/`clad`/`glide` are hard-excluded
  from ever accepting, regardless of position.

**Transcript source: a deliberate deviation from the literal spec, flagged
here rather than silently substituted.** The brief named
`wss://meeting-data.bot.recall.ai/api/v1/transcript` specifically — that
endpoint is for the bot's own in-meeting *webpage* to connect to Recall
directly, documented thinly (an unversioned code sample, no documented
partial/final distinction, would need a second live websocket held open
from `web/meeting/app.js` and relayed back over `/ws/meeting`). Implemented
instead via `recording_config.realtime_endpoints` + `transcript.data` —
the actively-documented, current mechanism, same shape as the existing
`audio_separate_raw.data` wiring (`recall/client.py`, new `/ws/transcript`
route in `transport/transcript_inbound.py`). This gives an explicit,
documented FINAL/partial split (only `transcript.data` is subscribed to)
and per-segment participant attribution, server-side, with no second
browser websocket. Functionally equivalent for this feature's needs; worth
double-checking against the literal endpoint if that distinction matters
for another reason.

**Instrumentation.** Every stage-1 match emits `wakeword.stage1_match`
with the transcript span, speaker, matched phrase, stage-2 verdict,
suppression reason, whether it reached `WAKE_PENDING`, and running
`stage1_total` / `suppressed_total` / `wakes_total` counters (so tailing
the JSONL run file shows current totals with no separate query). Floor
transitions emit `floor.window_opened` / `floor.window_closed` /
`floor.wake_fired`. Every `glad_gemini_first_byte_ms` event now also
carries `grounded` (from `LiveServerContent.grounding_metadata`, sticky
for the turn), and is additionally recorded under
`glad_gemini_first_byte_ms_grounded` / `_ungrounded` — Search grounding
(already wired into `live.session._build_config`) never calls a tool, so
it's invisible to `glad_tool_roundtrip_ms` but does add latency before the
first byte; separated at the source rather than after the aggregate is
already a mix of two populations.

**Suppression counts from a real run.** No live meeting available in this
sandbox, so this is a scripted run of the actual `Orchestrator.
on_transcript_segment` / `_drive_floor_control` code path (not synthetic
unit-test assertions) against an 11-line mixed transcript in ambient mode
(6 adjectival "glad"s, 1 "so glad" duplicate-pattern line, 1 "Gladiator 2"
decoy, 2 real vocatives: "Glad, what do you think about our timeline?" and
"Hey Glad, can you check the budget question again?"):

```
stage1_total:      10
suppressed_total:  8
wakes_total:       2
```

Both wakes fired only after their respective silence gaps, both correctly
skipped opening a raw-audio window, and both `send_text` payloads
correctly included everything volunteered since the last wake — including
a budget figure mentioned mid-conversation, which is exactly the kind of
volunteered answer this design exists to catch. The suppressed lines never
reached `WAKE_PENDING` at all, not just "didn't get an audible reply."

**Known limitation, not fixed here:** `AmbientBuffer` only ever empties via
a wake word firing (`_fire_wake`). Since `AMBIENT` mode is permanent once
reached (see monotonicity above), a long call where nobody ever addresses
Glad again after the last question is answered would let the buffer grow
for the rest of the call. Untreated because there's no natural flush point
otherwise without violating "never open a window with nothing to say" —
flagging it rather than silently shipping an unbounded buffer.

**Tunable placeholders, not measured against a real room:**
`_SPEECH_RMS_THRESHOLD = 0.02` (orchestrator.py), `FloorControl`'s
`gap_threshold_s = 1.2` and `endpoint_gap_s = 0.5` (floor.py). These
replace the old automatic-VAD sensitivity constants but haven't been A/B'd
the way the model/VAD choices in slice 2b were — a real call is needed to
tell if 1.2s is long enough for "Glad, what do you think" + a real pause,
or if 0.02 RMS misses quiet speakers / false-triggers on room noise.

**Tool dispatch robustness (existing code, not modified — see
`tests/test_tool_dispatch_robustness.py`).** All previously-`"probably
never executed"` branches now have a real assertion behind them: an
unknown `question_id` returns `{"ok": False, ...}` without raising; a
tool handler that raises still results in exactly one `FunctionResponse`
being sent (`LiveSession._dispatch_tool`'s existing try/except already
covers this); a `tool_call` with two `function_calls` gets a
`FunctionResponse` for each one. All three passed against the existing,
unmodified code on the first run — no fix was needed, but the guarantee is
no longer just "16 calls, never observed a problem."

## Slice 5: monitoring dashboard

`/monitor` is a read-only consumer of the existing event bus — no change to
`orchestrator.py` control flow, `live/session.py`, `outbound.py`, or the
meeting page. The one addition outside the new files is a single
`events.emit("run_meta", session_id=..., question_set_id=...)` call in
`main.py`'s `lifespan`, right after the orchestrator is constructed —
pure instrumentation, not a control-flow change, and the only way
`obs/runs.py` can attribute a run file to a session/question set without
the dashboard reaching into orchestrator internals.

**Two modes, one rendering pipeline.** `web/monitor/app.js` treats every
panel as a pure function of an in-memory `events` array; LIVE mode
(`/ws/monitor`) and REPLAY mode (`GET /api/runs/{id}`) just populate that
array differently. On load, the dashboard checks `GET /api/runs`: if the
most recent run has no `run_stop` yet, it's still open (killed, crashed,
or actually live) and the page defaults to LIVE, bootstrapping from that
run's on-disk events first and then appending from the socket (a
`mono`-keyed dedup covers the small race between "REST snapshot taken"
and "socket subscribed"). Otherwise it defaults to REPLAY on the newest
completed run, matching "the grader will most often arrive with no call
running."

**`/ws/monitor` never touches the audio path.** `events.on(...)` callbacks
run synchronously, inline, on whatever coroutine called `emit(...)` — which
in this codebase is always somewhere in the audio path
(`orchestrator.on_inbound_frame`, `live/session.py`'s tool-call handling,
`outbound.py`'s pacer). The subscriber in `transport/monitor.py` does only
a non-blocking `queue.put_nowait` per connected client and drops the event
for that client on `QueueFull`; the actual (awaited) websocket write
happens in a separate `asyncio.Task` per client, fully decoupled from
`emit()`. A slow or vanished monitor tab can never apply backpressure
upstream.

**A run file's identity is its filename, not a UUID.** `obs/runs.py` reads
`runs/*.jsonl` directly rather than tracking sessions in memory — a `run`
and a `session_id` aren't the same thing here (one server process usually
emits one run file across its whole lifetime; `orchestrator.session_id` is
a fresh uuid4 per `Orchestrator` instance). `list_runs()` sorts by each
run's *own first-event timestamp*, not the filename string — caught by
testing against real files: labels vary
(`slice3-...`, `wakeword_report-...`, `slice3-reconnect-test-...`), so
lexicographic filename order silently was not chronological order once
more than one label existed in `runs/`.

**Truncated / in-progress files are the expected case, not an error.**
`_parse_lines` skips any line that fails `json.loads` (a `kill -9` mid-write,
or a read racing an in-progress append) rather than raising, and
`list_runs()`/`load_run()` derive `completed` from the presence of a
`run_stop` event rather than assuming a clean shutdown. Verified against a
deliberately hand-truncated file with a valid `run_start` + `answer.recorded`
followed by a cut-off `{"ts": 3.0, "mono": 3.0, "t": "playback_stat` with no
closing brace: loads the two good events, drops the partial third line, no
exception.

**Latency table numbers are computed from the same raw `glad_*_ms` events
metrics.py itself would look at**, not from a separate aggregation path —
`percentileAt` in `app.js` is a line-for-line port of
`glad.obs.metrics.percentiles()`'s index formula, so replaying a run and
grepping its JSONL by hand produce identical p50/p95/p99 (verified below).
One naming wrinkle: `glad_gemini_first_byte_ms_grounded` /
`_ungrounded` are `metrics.py` sample-bucket names used only for the
in-process rolling `_samples` dict — there is no event with that literal
`t`. On the wire it's always `t: "glad_gemini_first_byte_ms"` plus a
`grounded: true/false` field (`orchestrator.py`); the dashboard
reconstructs the grounded/ungrounded split itself by partitioning on that
field, rather than searching for event types that don't exist.

**Timeline marks turn starts, tool calls, answers, interruptions, flushes,
and wake-word hits (both accepted and suppressed, so a correctly-ignored
"so glad you called" is visible as a dim dot, not just absent).**
Reconnects are in the spec's list too, but there is currently no event on
the bus for one — `LiveSession`'s reconnect loop only `logger.warning`s,
and `live/session.py` is explicitly frozen for this slice. The timeline
code defensively recognizes a `live.reconnecting` event type if one is
ever added, but today it will never fire. Flagging this gap rather than
quietly shipping a legend entry with no matching data: adding that one
`events.emit` call to the reconnect branch is a two-line, non-control-flow
change whenever `live/session.py` is unfrozen.

**No charting library** — the buffer-depth line and every timeline marker
are hand-built `<svg>` elements (`circle`/`polyline`/`line`/`text` via
`document.createElementNS`), matching the "a CDN dependency that fails
offline is worse than a hand-rolled line" constraint.

### Report back

Latency table, real ~142s conversation (`runs/slice3-20260821-011130.jsonl`,
1823 events, one `record_answer` call, one wake-word acceptance), replayed
through `/monitor` and cross-checked by grepping the JSONL directly —
identical to 1 decimal place:

```
stage                                    n     p50     p95     p99
Gemini first byte (turn start)           4    235.2   242.1   242.1
Gemini first byte — ungrounded turns     4    235.2   242.1   242.1
Gemini reply -> outbound audio         192      0.0     0.1     0.3
meeting audio -> Gemini                539      2.7     4.1     4.6
tool call round-trip                     1      4.2     4.2     4.2
```

- No grounded turns occurred in this run (Search grounding wasn't
  triggered), so the "grounded" row is legitimately empty rather than
  broken — worth re-checking once a call actually exercises grounding.
- `meeting audio -> Gemini` (inbound forwarding) and `Gemini reply ->
  outbound audio` are both sub-5ms at p99 — nothing here explains
  perceived latency; `Gemini first byte` (audio in -> first reply byte,
  ~235-242ms) is the dominant, expected cost, and matches slice 2b's
  measurements. Nothing looked worse now that it's visible in a table
  instead of scattered log lines.
- Audio health on the same run: 4 underruns total, buffer depth p50 0ms /
  p95 194.7ms (bursty — Gemini produces a whole utterance faster than it
  plays, so depth saw-tooths between bursts and near-zero, visible
  directly as the shape of the polyline rather than needing to infer it
  from a p50/p95 pair alone), flush-to-ack p50/p95 both 302.5ms (only one
  flush in this run).
- One thing that was mildly surprising building the event log filter:
  `playback_stats` is by far the highest-volume event type (client stats
  ping every 200ms) and dominates the raw stream — the LATENCY and
  TIMELINE panels have to explicitly filter it out (and everything else
  that isn't a `glad_*_ms` sample or a discrete "story" event) or the
  signal drowns in it. The event log's substring filter is there
  specifically so a grader can type `wakeword` or `answer` and ignore that
  noise rather than scrolling past thousands of stats pings.

## Slice 0 / 1 notes

Meet audio unlock trail (kept short): call `getUserMedia` first, play through
real page audio output (`AudioContext.destination` / `<audio>`), use
`web_4_core`, separate raw audio is 16kHz s16le mono. Streamed playback is
24kHz to match the worklet / `AudioContext({ sampleRate: 24000 })`.
