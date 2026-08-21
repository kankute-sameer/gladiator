# Glad

A Google Meet bot that stays quiet until someone addresses it by name, then runs a discovery conversation over Gemini Live.

Recall.ai puts Glad in the call. Glad mixes per-person audio, watches finalized transcripts for a wake word, and speaks back through the bot’s meeting page.

## What it does

- Joins a Meet as **Glad** (camera is a status page; speaker is Gemini’s voice).
- Starts **DORMANT** — no audio sent to Gemini, no talking.
- Wakes when someone actually addresses Glad (`Hey Glad`, `Glad, you there?`).
- Stays in the conversation until Gemini calls `go_dormant` (sidebar, dismissal, or all questions answered).
- Records answers with the `record_answer` tool against a YAML question set.

Default script (`question_sets/discovery_v1.yaml`): full legal name, date of birth, employer, employment start date, current address.

## Quick start

Python 3.11+, a Recall.ai key, a Gemini API key, and a public HTTPS URL (ngrok is fine).

```bash
conda activate glad
pip install -e ".[test]"
cp .env.example .env   # fill RECALL_API_KEY, GEMINI_API_KEY, PUBLIC_URL

ngrok http 8000
uvicorn glad.main:create_app --factory --port 8000
python -m scripts.send_bot "<meet-url>"
```

`PUBLIC_URL` must be the ngrok HTTPS origin. Restart uvicorn after changing `.env` or the question set.

Open `/monitor` while a run is live to watch latency, the event timeline, answers, and the log.

## Architecture

```
Recall.ai
  ├─ camera  → PUBLIC_URL          (meeting page)
  ├─ audio   → /ws/recall          (16 kHz s16le per speaker)
  ├─ text    → /ws/transcript      (final utterances only)
  └─ roster  → /ws/participants    (join / leave / update)

Glad server
  inbound mix → orchestrator
    DORMANT: wake-word on transcripts only
    ENGAGED: PCM → Gemini Live → paced 24 kHz PCM → /ws/meeting
  meeting page AudioWorklet → Meet hears Glad
```

Gemini Live uses **manual** activity detection. Glad opens and closes listen windows; the model does not auto-reply to background chatter.

Outbound audio is generated faster than realtime. `outbound.send_bytes` queues it and `_pace_loop` drips ~20 ms chunks at 24 kHz so the browser buffer does not balloon (that queue is also what a barge-in flush can clear instantly).

## Layout

| Path | Role |
|---|---|
| `src/glad/main.py` | FastAPI app, lifespan, static pages |
| `src/glad/orchestrator.py` | DORMANT / ENGAGED, floor, tools, Gemini I/O |
| `src/glad/live/session.py` | Gemini Live websocket |
| `src/glad/conversation/` | prompt, wake word, tools, session state |
| `src/glad/audio/` | mix / gate / RMS, inbound WAV recorder |
| `src/glad/transport/` | Recall and meeting websockets |
| `src/glad/recall/` | Create-bot API client |
| `src/glad/web/meeting/` | Bot camera + playback worklet |
| `src/glad/web/monitor/` | `/monitor` dashboard |
| `question_sets/` | Discovery questions (YAML) |
| `runs/` | JSONL event logs + inbound WAVs |

## HTTP and websockets

| Route | Purpose |
|---|---|
| `GET /` | Meeting page (bot camera / speaker) |
| `GET /monitor` | Operator dashboard (live or replay) |
| `GET /health` | Liveness |
| `GET /state/questions` | Question set + recorded answers |
| `GET /api/runs`, `GET /api/runs/{id}` | List / load JSONL runs |
| `/ws/recall` | Per-participant raw audio from Recall |
| `/ws/transcript` | Final transcript segments from Recall |
| `/ws/participants` | Roster events from Recall |
| `/ws/meeting` | Levels + outbound PCM to the meeting page |
| `/ws/monitor` | Event stream for the dashboard |

Recall is created with `web_4_core` (needed for output audio and separate per-participant realtime audio). Inbound PCM is 16 kHz mono s16le; playback is 24 kHz.

## Conversation

Glad stays silent until stage-2 wake-word detection accepts an address.

- **Stage 1** matches `glad` and ASR near-misses on word boundaries (`gladiator` before `glad`, so the movie title does not also count as `glad`).
- **Stage 2** requires a vocative cue (greeting, question, imperative). `"We watched Gladiator last night"` and `"so glad you called"` do not wake.

While ENGAGED, audio is mixed (noise-gated, summed, never averaged) and forwarded to Gemini. Gemini records answers with `record_answer` and leaves with `go_dormant`.

Change the script with `QUESTION_SET` in `.env` (filename under `question_sets/`, no `.yaml`). Optional `notes` on a question are model-only hints, not spoken text.

## Config

See `.env.example`. The important ones:

| Variable | Meaning |
|---|---|
| `RECALL_API_KEY` / `RECALL_BASE_URL` | Recall region + key |
| `PUBLIC_URL` | HTTPS origin Recall loads as the camera |
| `GEMINI_API_KEY` | Gemini Live |
| `GEMINI_VOICE` | Prebuilt voice (default `Kore`) |
| `QUESTION_SET` | YAML under `question_sets/` (default `discovery_v1`) |
| `RECORD_INBOUND_AUDIO` | Mixed 16 kHz WAV next to the run JSONL |

Model default: `gemini-3.1-flash-live-preview`.

## Scripts

```bash
python -m scripts.send_bot "<meet-url>"
python -m scripts.play_file assets/test_speech.wav          # outbound path only
python -m scripts.fetch_recall_audio <bot-id>               # post-call artifacts
python -m pytest
```

## Monitor

`http://localhost:8000/monitor`

- **Live** follows the current run over `/ws/monitor`.
- **Replay** loads a finished `runs/*.jsonl`.

Panels: latency percentiles, event timeline, answers, roster, audio health, raw event log.
