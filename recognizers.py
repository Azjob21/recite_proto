"""ASR backends.  A recognizer turns a float32 16 kHz mono array into text."""
import os
import numpy as np


class Recognizer:
    name = "base"
    def transcribe(self, audio: np.ndarray) -> str:
        raise NotImplementedError


class NullRecognizer(Recognizer):
    """No model: use the 'simulate' box in the UI to test the reveal logic."""
    name = "none"
    def transcribe(self, audio):
        return ""


def _guard(text):
    """Drop Whisper's repetition loops (typical on noise/silence)."""
    words = text.split()
    if len(words) > 6 and len(set(words[-6:])) == 1:
        return ""
    return " ".join(words)


class WhisperRecognizer(Recognizer):
    def __init__(self, model_id=None):
        import torch
        from transformers import pipeline
        self.model_id = model_id or os.getenv("RECITE_MODEL", "tarteel-ai/whisper-base-ar-quran")
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.name = f"whisper:{self.model_id}@{device}"
        self.pipe = pipeline("automatic-speech-recognition", model=self.model_id, device=device)
        # Older fine-tunes (e.g. the Tarteel one) ship an outdated generation config
        # without a language table: passing language/task would raise ValueError.
        if os.getenv("RECITE_QUANT") == "1":      # experimental: int8 dynamic quantisation on CPU
            try:
                self.pipe.model = torch.quantization.quantize_dynamic(
                    self.pipe.model, {torch.nn.Linear}, dtype=torch.qint8)
                self.name += "+int8"
            except Exception as e:
                print("Quantisation skipped:", repr(e))
        self._kw = {"num_beams": 1}
        if getattr(self.pipe.model.generation_config, "lang_to_id", None):
            self._kw.update(language="arabic", task="transcribe")

    def _run(self, audio, kw):
        return self.pipe({"raw": audio.astype("float32"), "sampling_rate": 16000}, generate_kwargs=kw)

    def transcribe(self, audio):
        try:
            out = self._run(audio, self._kw)
        except ValueError:
            self._kw = {}                      # last resort: model defaults
            out = self._run(audio, self._kw)
        return _guard(out["text"])


class FasterWhisperRecognizer(Recognizer):
    """CTranslate2 build of the same model (int8): typically several times faster on CPU.
    Convert once:  ct2-transformers-converter --model tarteel-ai/whisper-base-ar-quran
                   --output_dir whisper-ar-quran-ct2 --quantization int8 --copy_files tokenizer.json preprocessor_config.json"""
    def __init__(self):
        from faster_whisper import WhisperModel
        path = os.getenv("RECITE_CT2_DIR", "whisper-ar-quran-ct2")
        self.model = WhisperModel(path, device="cpu", compute_type="int8")
        self.name = f"faster-whisper:{path}@cpu-int8"

    def transcribe(self, audio):
        segs, _ = self.model.transcribe(audio.astype("float32"), language="ar", beam_size=1,
                                        without_timestamps=True, condition_on_previous_text=False,
                                        vad_filter=False, temperature=0.0)
        return _guard(" ".join(s.text.strip() for s in segs))


def make_recognizer():
    kind = os.getenv("RECITE_BACKEND", "whisper")
    if kind == "none":
        return NullRecognizer()
    if kind == "faster":
        return FasterWhisperRecognizer()
    return WhisperRecognizer()
