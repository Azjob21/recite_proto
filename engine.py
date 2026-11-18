"""Reveal engine: compares recognised words with the reference verse and decides
what to reveal and when to stop on an error.  Pure Python, no ML dependencies."""
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Optional

_STRIP = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED\u0640\u200c-\u200f]")
_NON_AR = re.compile(r"[^\u0621-\u064A]")


def normalize(word: str) -> str:
    """Strip diacritics / marks and unify letter variants so ASR spelling noise
    does not count as an error.  Applied identically to reference and hypothesis."""
    w = _STRIP.sub("", word)
    for a, b in (("أ", "ا"), ("إ", "ا"), ("آ", "ا"), ("ٱ", "ا"), ("ى", "ي"), ("ی", "ي"),
                 ("ک", "ك"), ("ة", "ه"), ("ؤ", "و"), ("ئ", "ي"), ("ء", "")):
        w = w.replace(a, b)
    return _NON_AR.sub("", w)


def tokenize(text: str):
    """Return [(display_word, normalised_word)]. Pure-mark tokens (waqf signs,
    ayah-end symbols) are glued to the previous word so indices stay aligned."""
    out = []
    for tok in text.split():
        n = normalize(tok)
        if not n:
            if out:
                out[-1] = (out[-1][0] + tok, out[-1][1])
            continue
        out.append((tok, n))
    return out


def sim(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    return SequenceMatcher(None, a, b).ratio()


@dataclass
class Result:
    cursor: int
    error: Optional[dict] = None


def simulate(ref, start, hyp, thr, anchor=True) -> Result:
    """Walk the hypothesis words along the reference starting at `start`."""
    n, p, i = len(ref), start, 0
    # Re-read support: if the first words match an earlier spot, jump back there.
    if anchor and len(hyp) >= 2 and p < n and sim(hyp[0], ref[p]) < thr:
        for q in range(p, -1, -1):
            if q + 1 < n and sim(hyp[0], ref[q]) >= thr and sim(hyp[1], ref[q + 1]) >= thr:
                p = q
                break
    while i < len(hyp) and p < n:
        h = hyp[i]
        if sim(h, ref[p]) >= thr:                                   # normal match
            p += 1; i += 1; continue
        if p + 1 < n and sim(h, ref[p] + ref[p + 1]) >= thr:       # ASR merged two words
            p += 2; i += 1; continue
        if i + 1 < len(hyp) and sim(h + hyp[i + 1], ref[p]) >= thr:  # ASR split a word
            p += 1; i += 2; continue
        if p > 0 and sim(h, ref[p - 1]) >= thr:                     # stutter / repeat
            i += 1; continue
        kind = "skip" if (p + 1 < n and sim(h, ref[p + 1]) >= thr) else "wrong"
        return Result(p, {"index": p, "expected": ref[p], "heard": h, "kind": kind})
    return Result(p, None)


class RevealTracker:
    """Stateful tracker across speech segments.

    Streaming ASR re-transcribes a growing window, so the last word is unstable.
    Interim results therefore (a) ignore the last `lag` word(s) and (b) only
    commit words / errors seen in two consecutive runs.  A final result (after a
    pause) is committed directly."""

    def __init__(self, ref_norm, thr=0.85, lag=1, max_wps=5.0):
        self.ref, self.thr, self.lag, self.max_wps = ref_norm, thr, lag, max_wps
        self.cursor = 0
        self.revealed = 0
        self.error = None
        self._seg = None
        self._seg_start = 0
        self._prev: Optional[Result] = None

    def update(self, hyp_words, final: bool, seg_id: int, speech_sec=None):
        if seg_id != self._seg:                       # new segment starts at the cursor
            self._seg, self._seg_start, self._prev = seg_id, self.cursor, None
        hyp = [normalize(w) for w in hyp_words]
        hyp = [w for w in hyp if w]
        if speech_sec is not None and self.max_wps:
            # Plausibility cap: nobody recites faster than ~max_wps words per second of voiced audio.
            # Blocks the model "completing" a verse it only half heard.
            hyp = hyp[: int(speech_sec * self.max_wps) + 1]
        use = hyp if (final or not self.lag) else hyp[: -self.lag]
        res = simulate(self.ref, self._seg_start, use, self.thr)
        prev, self._prev = self._prev, res
        confirmed_error = None
        if final:
            new_cursor, confirmed_error = res.cursor, res.error
        else:
            new_cursor = min(res.cursor, prev.cursor) if prev else self.cursor
            if res.error and prev and prev.error and \
               prev.error["index"] == res.error["index"] and prev.error["heard"] == res.error["heard"]:
                new_cursor, confirmed_error = res.cursor, res.error
        if confirmed_error:
            self.cursor = confirmed_error["index"]
            self.error = confirmed_error
        else:
            if new_cursor > self._seg_start or (new_cursor != self.cursor and new_cursor > 0):
                self.error = None
            self.cursor = new_cursor
        self.revealed = max(self.revealed, self.cursor)
        return confirmed_error is not None

    def state(self):
        return {"revealed": self.revealed, "cursor": self.cursor, "error": self.error,
                "done": self.revealed >= len(self.ref)}
