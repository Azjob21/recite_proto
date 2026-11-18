"""One live recitation session: audio in -> VAD/segmentation -> ASR -> tracker -> JSON out."""
import asyncio, os, time, wave
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
from engine import RevealTracker

FRAME = 320  # 20 ms at 16 kHz


@dataclass
class Config:
    end_silence_ms: int = 600
    interim_ms: int = 900
    min_seg_ms: int = 900
    max_seg_s: int = 20
    min_rms: float = 0.012
    debug_dir: str = field(default_factory=lambda: os.getenv("RECITE_DEBUG_DIR", ""))


class Session:
    def __init__(self, send, recognizer, tracker: RevealTracker, cfg: Config = None):
        self.send, self.rec, self.tracker = send, recognizer, tracker
        self.cfg = cfg or Config()
        self.lock = asyncio.Lock()
        self.tasks = set()
        self.pending = np.zeros(0, dtype=np.float32)
        self.noise, self.onset, self.speaking = 0.005, 0, False
        self.preroll = deque(maxlen=15)
        self.frames, self.rmss = [], []
        self.silence = self.since_interim = 0
        self.seg_id = 0
        self.discarding = False
        self.interim_busy = False
        self.n_frames = 0

    # ---------- audio in ----------
    def on_audio(self, pcm: bytes):
        x = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        self.pending = np.concatenate([self.pending, x])
        while len(self.pending) >= FRAME:
            fr, self.pending = self.pending[:FRAME], self.pending[FRAME:]
            self._frame(fr)

    def _frame(self, fr):
        c = self.cfg
        rms = float(np.sqrt(np.mean(fr * fr)))
        speech = rms > max(c.min_rms, self.noise * 2.5)
        self.n_frames += 1
        if self.n_frames % 5 == 0:
            self._spawn(self.send({"type": "vad", "speech": bool(self.speaking or speech), "rms": round(rms, 4)}))
        if not self.speaking:
            self.noise = 0.97 * self.noise + 0.03 * rms
            self.preroll.append((fr, rms))
            self.onset = self.onset + 1 if speech else 0
            if self.onset >= 3:
                self.speaking, self.silence, self.since_interim = True, 0, 0
                self.seg_id += 1
                self.frames = [f for f, _ in self.preroll]
                self.rmss = [r for _, r in self.preroll]
                self.preroll.clear()
            return
        self.frames.append(fr); self.rmss.append(rms)
        self.silence = 0 if speech else self.silence + 1
        self.since_interim += 1
        if self.silence * 20 >= c.end_silence_ms:
            self._end_segment()
        elif len(self.frames) * 20 >= c.max_seg_s * 1000:
            self._cut_segment()
        elif (self.since_interim * 20 >= c.interim_ms and len(self.frames) * 20 >= c.min_seg_ms
              and not self.discarding and not self.interim_busy):
            self.since_interim = 0
            self.interim_busy = True
            audio, voiced = self._prep(self.frames, self.rmss)
            self._spawn(self._recognize(audio, False, self.seg_id, voiced))

    def _end_segment(self):
        audio, voiced = self._prep(self.frames, self.rmss)
        if not self.discarding and voiced >= 0.3:       # ignore < 300 ms blips
            self._spawn(self._recognize(audio, True, self.seg_id, voiced))
        self.discarding = False
        self.speaking, self.frames, self.rmss, self.onset = False, [], [], 0

    def _cut_segment(self):
        """Segment too long: cut at the quietest frame of the last 3 s."""
        lo = max(0, len(self.frames) - 150)
        idx = lo + int(np.argmin(self.rmss[lo:]))
        head, tail = self.frames[: idx + 1], self.frames[idx + 1:]
        if not self.discarding:
            audio, voiced = self._prep(head, self.rmss[: idx + 1])
            self._spawn(self._recognize(audio, True, self.seg_id, voiced))
        self.seg_id += 1
        self.frames, self.rmss = tail, self.rmss[idx + 1:]
        self.since_interim = 0

    def _thr(self):
        return max(self.cfg.min_rms, self.noise * 2.5)

    def _prep(self, frames, rmss):
        """Audio for ASR: drop long trailing silence (a classic Whisper hallucination trigger)
        and report how many seconds are actually voiced."""
        thr = self._thr()
        last = max((i for i, r in enumerate(rmss) if r > thr), default=len(rmss) - 1)
        keep = frames[: min(len(frames), last + 11)]          # keep ~200 ms tail
        voiced = sum(1 for r in rmss if r > thr) * 0.02
        return np.concatenate(keep), voiced

    # ---------- recognition ----------
    def _spawn(self, coro):
        t = asyncio.ensure_future(coro)
        self.tasks.add(t)
        t.add_done_callback(self.tasks.discard)

    async def _recognize(self, audio, final, seg_id, voiced=None):
        try:
            async with self.lock:
                t0 = time.time()
                text = await asyncio.get_running_loop().run_in_executor(None, self.rec.transcribe, audio)
                ms = int((time.time() - t0) * 1000)
        except Exception as e:                      # never fail silently / never wedge the scheduler
            if not final:
                self.interim_busy = False
            print("Recognizer error:", repr(e))
            await self.send({"type": "error", "message": f"Recognizer failed: {e}"[:300]})
            return
        if not final:
            self.interim_busy = False
            if seg_id != self.seg_id or self.discarding:
                return
        print(f"[{'final' if final else 'interim'}] seg={seg_id} audio={len(audio)/16000:.1f}s asr={ms}ms words={len(text.split())}")
        if final and self.cfg.debug_dir:
            self._dump(audio, text, seg_id, ms)
        await self.apply_hypothesis(text, final, seg_id, ms, voiced)

    def _dump(self, audio, text, seg_id, ms):
        """Save the exact audio the model heard + its output, to tell mic problems from model hallucination."""
        d = Path(self.cfg.debug_dir); d.mkdir(parents=True, exist_ok=True)
        base = str(d / f"{time.strftime('%H%M%S')}_seg{seg_id}")
        with wave.open(base + ".wav", "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000)
            w.writeframes((np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes())
        Path(base + ".txt").write_text(f"{text}\n({ms} ms)\n", encoding="utf-8")

    async def apply_hypothesis(self, text, final, seg_id, ms=0, voiced=None):
        err = self.tracker.update(text.split(), final, seg_id, voiced)
        if err and not final:
            self.discarding = True      # stop until the reader pauses
        await self.send({"type": "state", **self.tracker.state(), "hyp": text,
                         "final": final, "asr_ms": ms, "seg": seg_id})

    async def close(self):
        if self.tasks:
            await asyncio.gather(*list(self.tasks), return_exceptions=True)
