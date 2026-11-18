import sys, pathlib, json
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
import numpy as np
from fastapi.testclient import TestClient
from server import create_app

SR = 16000
def tone(sec):  # loud "speech"
    t = np.arange(int(SR * sec)) / SR
    return (0.3 * np.sin(2 * np.pi * 220 * t) * 32767).astype(np.int16)
def silence(sec): return np.zeros(int(SR * sec), dtype=np.int16)

class FakeRec:
    """Perfect ASR: one word per 0.5 s of audio."""
    name = "fake"
    def __init__(self, words): self.words = words
    def transcribe(self, audio): return " ".join(self.words[: int(len(audio) / SR / 0.5)])

def run(words_heard, custom="قل هو الله أحد"):
    client = TestClient(create_app(FakeRec(words_heard)))
    with client.websocket_connect("/ws") as ws:
        ws.send_text(json.dumps({"type": "init", "text": custom}))
        ready = ws.receive_json()
        assert ready["type"] == "ready" and len(ready["words"]) == 4
        audio = np.concatenate([silence(0.5), tone(2.4), silence(1.0)])
        for i in range(0, len(audio), 1600):
            ws.send_bytes(audio[i:i + 1600].tobytes())
        last = None
        for _ in range(400):
            m = ws.receive_json()
            if m["type"] == "state":
                last = m
                if m["final"] or m["error"]:
                    break
        return last

def test_correct_recitation_reveals_everything():
    s = run("قل هو الله أحد".split())
    assert s["revealed"] == 4 and s["error"] is None and s["done"]

def test_wrong_word_stops_with_error():
    s = run("قل هو الرحمن أحد".split())
    assert s["error"]["index"] == 2 and s["revealed"] == 2
