// Jitter-buffered playback processor. Runs on the audio rendering thread,
// called every 128 samples (one "render quantum"). Nothing in process()
// may allocate, construct objects, or log -- all of that happens in the
// port message handler instead, which runs off the render-quantum path.
//
// AudioContext on the main thread MUST be constructed with
// { sampleRate: 24000 } to match this constant -- no resampling happens
// anywhere in this pipeline.
const SAMPLE_RATE = 24000;

// Do not start draining until this much audio has arrived. Tune here.
const PREBUFFER_MS = 200;

// Ring buffer capacity: generous headroom above PREBUFFER_MS so normal
// jitter doesn't overflow it, while still bounded (no unbounded growth).
const RING_CAPACITY_MS = 3000;

// Post depth/underrun stats back to the main thread this often, not every
// block -- process() runs far too often (every ~5.3ms at 24kHz) for a
// per-call postMessage to be free.
const REPORT_EVERY_N_BLOCKS = 128;

const RING_CAPACITY_SAMPLES = Math.round((RING_CAPACITY_MS / 1000) * SAMPLE_RATE);
const PREBUFFER_SAMPLES = Math.round((PREBUFFER_MS / 1000) * SAMPLE_RATE);
const FULL_SCALE = 32768.0;

class PlaybackProcessor extends AudioWorkletProcessor {
  constructor() {
    super();

    // Preallocated once. process() only ever reads/writes into this.
    this._ring = new Float32Array(RING_CAPACITY_SAMPLES);
    this._writeIndex = 0;
    this._readIndex = 0;
    this._filledSamples = 0;

    this._prebuffering = true;
    this._underrunCount = 0;
    this._blocksSinceReport = 0;

    this.port.onmessage = (event) => this._handleMessage(event.data);
  }

  _handleMessage(message) {
    if (message.type === "pcm") {
      this._enqueue(new Int16Array(message.buffer));
    } else if (message.type === "flush") {
      this._flush();
      // Ack on the message path (not process()): main thread uses this to
      // timestamp flush -> silence for the latency table.
      this.port.postMessage({
        type: "flushed",
        depthMs: 0,
        underruns: this._underrunCount,
      });
    }
  }

  // Converts s16le -> float32 and copies into the ring buffer. Runs in the
  // message handler, off the process() hot path, so allocating a view here
  // is fine.
  _enqueue(int16Samples) {
    for (let i = 0; i < int16Samples.length; i++) {
      if (this._filledSamples >= RING_CAPACITY_SAMPLES) {
        // Buffer is full (sender is pushing faster than we can drain).
        // Drop the newest sample rather than overwrite unplayed audio.
        break;
      }
      this._ring[this._writeIndex] = int16Samples[i] / FULL_SCALE;
      this._writeIndex = (this._writeIndex + 1) % RING_CAPACITY_SAMPLES;
      this._filledSamples++;
    }

    if (!this._prebuffering && this._filledSamples === 0) {
      this._prebuffering = true;
    }
  }

  _flush() {
    this._writeIndex = 0;
    this._readIndex = 0;
    this._filledSamples = 0;
    this._prebuffering = true;
  }

  process(_inputs, outputs) {
    const output = outputs[0];
    const blockSize = output[0] ? output[0].length : 128;

    if (this._prebuffering && this._filledSamples >= PREBUFFER_SAMPLES) {
      this._prebuffering = false;
    }

    let underran = false;
    for (let i = 0; i < blockSize; i++) {
      let sample = 0.0;

      if (!this._prebuffering && this._filledSamples > 0) {
        sample = this._ring[this._readIndex];
        this._readIndex = (this._readIndex + 1) % RING_CAPACITY_SAMPLES;
        this._filledSamples--;
      } else if (!this._prebuffering) {
        underran = true;
      }

      for (let channel = 0; channel < output.length; channel++) {
        output[channel][i] = sample;
      }
    }

    if (underran) {
      // One count per render quantum that ran dry, then re-enter prebuffer
      // so a single gap doesn't spin the counter every sample forever.
      this._underrunCount++;
      this._prebuffering = true;
    }

    this._blocksSinceReport++;
    if (this._blocksSinceReport >= REPORT_EVERY_N_BLOCKS) {
      this._blocksSinceReport = 0;
      this.port.postMessage({
        depthMs: (this._filledSamples / SAMPLE_RATE) * 1000,
        underruns: this._underrunCount,
      });
    }

    return true;
  }
}

registerProcessor("playback", PlaybackProcessor);
