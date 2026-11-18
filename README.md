# Quran Recitation Reveal

A local prototype for following Quran recitation word by word. The reader starts with a blank page. As the system recognizes words in the expected passage, it reveals them; when recognition no longer matches, it pauses and marks the point to review.

The project explores a practical first step toward recitation assistance: checking whether the words being recited match a known Quran passage. It is an experimental prototype, not a tajweed grader or a replacement for a qualified Quran teacher.

## How it works

1. Select a surah and ayah range, or enter custom reference text.
2. The browser captures microphone audio and sends it to the local Python server over a WebSocket.
3. Voice activity detection groups the audio around pauses. The recognizer transcribes each active segment with a Quran-tuned Whisper model.
4. The reveal engine aligns recognized words with the expected passage and updates the page.

The current pipeline is:

```text
Browser microphone -> WebSocket -> voice activity detection -> Whisper -> word alignment -> reveal interface
```

Quran text is loaded from the Quran.com API and cached locally. If the API is unavailable, the custom text field can be used instead. Audio recognition runs in the local Python process.

## Current capabilities

- Reveals recognized words from a selected passage.
- Stops and highlights the likely error position when the recognized text diverges from the reference.
- Supports restarting from the error word or replaying the passage.
- Includes a strictness control for adjusting how closely recognized words must match.
- Includes a simulation input for trying the reveal logic without a speech model.
- Can use the default Quran-tuned Whisper model or another compatible Hugging Face model.

## Run locally

Python 3.10 or newer is recommended.

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m uvicorn server:app --port 8000
```

### macOS or Linux

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m uvicorn server:app --port 8000
```

Open `http://localhost:8000` in a browser. Microphone access requires localhost or a secure connection. Select a passage, choose **Load**, then choose **Start mic**. The first launch downloads the default model, `tarteel-ai/whisper-base-ar-quran` (approximately 300 MB), and may take longer while the model loads.

## Configuration

Use a different compatible Hugging Face model by setting `RECITE_MODEL` before starting the server.

Windows PowerShell:

```powershell
$env:RECITE_MODEL = "your-model-id"
python -m uvicorn server:app --port 8000
```

macOS or Linux:

```bash
RECITE_MODEL="your-model-id" python -m uvicorn server:app --port 8000
```

To run without a speech model and exercise the reveal logic through the simulation input, set `RECITE_BACKEND=none`.

## Try the prototype

- Recite a passage with natural pauses and see whether the words appear in order.
- Skip, substitute, or repeat a word and observe where the reveal stops.
- Use the simulation input to check how the reveal engine handles recognized text.
- Adjust the strictness control to see how tolerance changes the matching behavior.

The debug bar reports recognition time. End-to-end reveal delay also includes the audio segmentation window and the reveal engine's buffering.

## Project structure

- `server.py`: FastAPI application and WebSocket endpoint.
- `session.py`: audio buffering, voice activity detection, segmentation, and recognition scheduling.
- `recognizers.py`: speech recognition backends.
- `engine.py`: text normalization, word alignment, and reveal/error tracking.
- `quran_text.py`: Quran.com text retrieval, local caching, and tokenization.
- `static/index.html`: browser interface and microphone capture.
- `tests/`: unit tests for the reveal engine and session behavior.

Run the test suite with:

```bash
python -m pytest tests
```

## Scope and limitations

- Matching is based on recognized text. The prototype does not verify individual phonemes, vowel marks, letter articulation, or tajweed rules.
- A language model can infer or correct words from context, so Whisper may conceal a recitation error or introduce a false mismatch.
- The interface simulates streaming by re-transcribing the current speech segment as it grows; it is not a true streaming recognizer.
- Voice activity detection is energy-based, and browser audio is resampled with a simple linear method. Performance can vary with room noise, microphone, device, and recitation style.
- Quran text matching uses Quran.com imlaei spelling. The current assumptions are Hafs and a single expected passage.
- Recognition speed depends on the selected model and hardware. The default model may be slow on a CPU.

## Next direction

The next experiment is to compare the text-based recognizer with phoneme-level verification, such as Quran Muaalem. That approach needs a phonetic reference for the selected passage and would require a verification interface that returns per-unit scores. The current `transcribe(audio) -> text` recognizer interface and word-level reveal tracker are not sufficient for that evaluation without an additional backend and comparison logic.
