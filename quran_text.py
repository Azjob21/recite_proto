"""Fetch reference text from the Quran.com API (cached on disk)."""
import json, urllib.request
from pathlib import Path
from engine import tokenize

API = "https://api.quran.com/api/v4"
CACHE = Path(__file__).parent / "cache"


def fetch_ayah(surah: int, ayah: int) -> str:
    f = CACHE / f"{surah}_{ayah}.json"
    if f.exists():
        return json.loads(f.read_text("utf-8"))["text"]
    url = f"{API}/verses/by_key/{surah}:{ayah}?language=ar&fields=text_imlaei"
    req = urllib.request.Request(url, headers={"User-Agent": "recite-proto/0.1"})
    with urllib.request.urlopen(req, timeout=15) as r:
        data = json.load(r)
    text = data["verse"]["text_imlaei"]
    CACHE.mkdir(exist_ok=True)
    f.write_text(json.dumps({"text": text}, ensure_ascii=False), "utf-8")
    return text


def get_range(surah: int, first: int, last: int) -> str:
    return " ".join(fetch_ayah(surah, a) for a in range(first, last + 1))


def words_for(text: str):
    return tokenize(text)
