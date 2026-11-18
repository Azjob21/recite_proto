"""Run:  python -m uvicorn server:app --port 8000   then open http://localhost:8000"""
import asyncio, json
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from engine import RevealTracker
from session import Session
import quran_text

STATIC = Path(__file__).parent / "static"


def create_app(recognizer=None):
    holder = {"rec": recognizer}

    @asynccontextmanager
    async def lifespan(_):
        if holder["rec"] is None:
            from recognizers import make_recognizer
            print("Loading recognizer (first run downloads the model)...")
            holder["rec"] = make_recognizer()
        print("Recognizer ready:", holder["rec"].name)
        yield

    app = FastAPI(lifespan=lifespan)

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    @app.get("/api/info")
    def info():
        return {"backend": holder["rec"].name if holder["rec"] else "loading"}

    @app.websocket("/ws")
    async def ws(sock: WebSocket):
        await sock.accept()
        session = None

        async def send(obj):
            try:
                await sock.send_text(json.dumps(obj, ensure_ascii=False))
            except Exception:
                pass

        try:
            while True:
                msg = await sock.receive()
                if msg.get("type") == "websocket.disconnect":
                    break
                if msg.get("bytes") is not None:
                    if session:
                        session.on_audio(msg["bytes"])
                    continue
                data = json.loads(msg["text"])
                kind = data.get("type")
                if kind == "init":
                    try:
                        text = data.get("text") or await asyncio.get_running_loop().run_in_executor(
                            None, quran_text.get_range, int(data["surah"]), int(data["from"]), int(data["to"]))
                    except Exception as e:
                        await send({"type": "error", "message": f"Could not load text: {e}"})
                        continue
                    words = quran_text.words_for(text)
                    if session:
                        await session.close()
                    tracker = RevealTracker([n for _, n in words], thr=float(data.get("thr", 0.85)))
                    session = Session(send, holder["rec"], tracker)
                    await send({"type": "ready", "words": [d for d, _ in words], "backend": holder["rec"].name})
                elif kind == "config" and session:
                    session.tracker.thr = float(data.get("thr", session.tracker.thr))
                elif kind == "hyp" and session:      # simulated recognition (no mic / no model)
                    session.seg_id += 1
                    await session.apply_hypothesis(data.get("text", ""), True, session.seg_id)
        except WebSocketDisconnect:
            pass
        finally:
            if session:
                await session.close()

    return app


app = create_app()
