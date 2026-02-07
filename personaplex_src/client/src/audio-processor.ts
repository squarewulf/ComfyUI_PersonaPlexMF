// @ts-nocheck
// Low-latency audio processor with hard latency cap

class MoshiProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    
    // Default: 100ms target, 200ms max before dropping
    this.targetMs = 100;
    this.maxMs = 200;
    this.targetSamples = Math.round(this.targetMs * sampleRate / 1000);
    this.maxSamples = Math.round(this.maxMs * sampleRate / 1000);
    
    this.frames = [];
    this.offset = 0;
    this.playing = false;
    this.pktCount = 0;
    
    // Stats tracking
    this.totalPlayed = 0;
    this.actualPlayed = 0;
    this.timeInStream = 0;
    this.minDelay = 9999;
    this.maxDelay = 0;

    this.port.onmessage = (event) => {
      if (event.data.type === "reset") {
        this.frames = [];
        this.offset = 0;
        this.playing = false;
        this.pktCount = 0;
        this.totalPlayed = 0;
        this.actualPlayed = 0;
        this.timeInStream = 0;
        this.minDelay = 9999;
        this.maxDelay = 0;
        console.log("[AUDIO] Reset");
        return;
      }

      // Apply buffer configuration from main thread
      if (event.data.type === "config") {
        this.targetMs = event.data.initBufferMs || 100;
        this.maxMs = Math.max(this.targetMs * 2, event.data.partialBufferMs || 200);
        this.targetSamples = Math.round(this.targetMs * sampleRate / 1000);
        this.maxSamples = Math.round(this.maxMs * sampleRate / 1000);
        console.log("[AUDIO] Config: target=" + this.targetMs + "ms, max=" + this.maxMs + "ms");
        return;
      }

      const frame = event.data.frame;
      const micDuration = event.data.micDuration || 0;
      this.frames.push(frame);
      
      if (this.pktCount < 5) {
        console.log("[AUDIO] PKT", this.pktCount++, "buf:", this.bufMs() + "ms");
      }

      // Start when we have minimum buffer (half of target)
      if (!this.playing && this.bufSize() >= this.targetSamples / 2) {
        this.playing = true;
        console.log("[AUDIO] PLAY started, buf:", this.bufMs() + "ms");
      }

      // Drop if over max - this prevents latency from climbing
      if (this.bufSize() > this.maxSamples) {
        const excess = this.bufSize() - this.targetSamples;
        const dropped = this.drop(excess);
        console.log("[AUDIO] DROP", Math.round(dropped * 1000 / sampleRate) + "ms, buf:", this.bufMs() + "ms");
      }

      // Calculate delay
      const delay = micDuration - this.timeInStream;
      if (delay > 0 && delay < 100) {
        this.minDelay = Math.min(this.minDelay, delay);
        this.maxDelay = Math.max(this.maxDelay, delay);
      }

      // Report stats
      this.port.postMessage({
        totalAudioPlayed: this.totalPlayed,
        actualAudioPlayed: this.actualPlayed,
        delay: delay,
        minDelay: this.minDelay === 9999 ? 0 : this.minDelay,
        maxDelay: this.maxDelay,
      });
    };
  }

  bufSize() {
    let size = 0;
    for (const f of this.frames) size += f.length;
    return size - this.offset;
  }
  
  bufMs() {
    return Math.round(this.bufSize() * 1000 / sampleRate);
  }

  drop(count) {
    let dropped = 0;
    while (dropped < count && this.frames.length > 0) {
      const first = this.frames[0];
      const avail = first.length - this.offset;
      const toDrop = Math.min(avail, count - dropped);
      this.offset += toDrop;
      this.timeInStream += toDrop / sampleRate;
      dropped += toDrop;
      if (this.offset >= first.length) {
        this.frames.shift();
        this.offset = 0;
      }
    }
    return dropped;
  }

  process(inputs, outputs, parameters) {
    const output = outputs[0][0];
    const needed = output.length;

    // Check and drop if over max every frame - keeps latency bounded
    if (this.bufSize() > this.maxSamples) {
      const excess = this.bufSize() - this.targetSamples;
      this.drop(excess);
    }

    if (!this.playing || this.frames.length === 0) {
      output.fill(0);
      return true;
    }

    let written = 0;
    while (written < needed && this.frames.length > 0) {
      const first = this.frames[0];
      const avail = first.length - this.offset;
      const toWrite = Math.min(avail, needed - written);
      
      output.set(first.subarray(this.offset, this.offset + toWrite), written);
      this.offset += toWrite;
      written += toWrite;

      if (this.offset >= first.length) {
        this.frames.shift();
        this.offset = 0;
      }
    }

    // Buffer underrun - wait for refill
    if (written < needed) {
      output.fill(0, written);
      this.playing = false;
      console.log("[AUDIO] UNDERRUN, waiting for buffer");
    }

    this.totalPlayed += needed / sampleRate;
    this.actualPlayed += written / sampleRate;
    this.timeInStream += written / sampleRate;

    return true;
  }
}

registerProcessor("moshi-processor", MoshiProcessor);
