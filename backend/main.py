"""Local office server.

View-only (default): GET, HEAD, and OPTIONS only. The microphone stays closed.
Interactive (`OFFICE_READ_ONLY=false`): localhost clients may talk to a desk,
assign tasks, pin notes, edit card text, and archive cards.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
BACKEND = Path(__file__).resolve().parent
sys.path.insert(0, str(BACKEND))

load_dotenv(ROOT / ".env")
load_dotenv(BACKEND / ".env", override=True)

from runtime.providers import (  # noqa: E402
    allowed_providers,
    api_key,
    apply_privacy_defaults,
    gemini_extras_available,
    resolve,
)

# Anonymous mode switches CrewAI telemetry off before anything imports crewai.
apply_privacy_defaults()

from runtime.browser import (  # noqa: E402
    _fix_playwright_browser_path,
    _latest_frames,
    add_frame_listener,
    remove_frame_listener,
)

_fix_playwright_browser_path()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("office.app")

CONFIG_PATH = Path(os.getenv("OFFICE_CONFIG", str(ROOT / "config" / "office.yml")))
STORAGE = os.getenv("OFFICE_STORAGE_DIR", str(ROOT / "data" / "blackboard"))
PORT = int(os.getenv("OFFICE_PORT", os.getenv("PORT", "8780")))
HOST = os.getenv("OFFICE_HOST", "127.0.0.1")
READ_ONLY = os.getenv("OFFICE_READ_ONLY", "true").lower() != "false"
DATA_DIR = Path(STORAGE).parent if Path(STORAGE).name == "blackboard" else Path(STORAGE)

_LOCAL_ORIGINS = [
    "http://127.0.0.1:8780",
    "http://localhost:8780",
    "http://127.0.0.1:3000",
    "http://localhost:3000",
]


def _prepare_blogger_token() -> None:
    """Copy host credentials into a writable path so OAuth refresh can persist.

    Must run before runtime.blogger is imported (the token path is read at client init).
    """
    source = Path(
        os.getenv("BLOGGER_CREDENTIALS_SOURCE", str(ROOT / "secrets" / "blogger_credentials.json"))
    )
    raw_dest = os.getenv("BLOGGER_TOKEN_FILE", str(DATA_DIR / "blogger_credentials.json"))
    dest = Path(raw_dest)
    if "secrets" in dest.parts and dest.name == "blogger_credentials.json":
        dest = DATA_DIR / "blogger_credentials.json"
    if not dest.is_absolute() or str(dest).startswith("./") or str(dest).startswith("../"):
        dest = DATA_DIR / "blogger_credentials.json"
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        if source.is_file():
            dest.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
            os.environ["BLOGGER_TOKEN_FILE"] = str(dest)
            logger.info("Blogger token file ready at %s", dest)
        elif dest.is_file():
            os.environ["BLOGGER_TOKEN_FILE"] = str(dest)
            logger.info("Blogger token file using existing %s", dest)
        else:
            logger.info("No Blogger credentials yet. Publishing stays disabled until you add them.")
    except OSError as exc:
        logger.error("Could not prepare Blogger token file: %s", exc)


_prepare_blogger_token()

from office import Office, load_office_config  # noqa: E402
from runtime.voice import OfficeVoiceSocket  # noqa: E402
from scheduler import DailyPublisher  # noqa: E402

office_config = load_office_config(CONFIG_PATH)
office = Office(STORAGE, config=office_config, config_path=CONFIG_PATH)

# Fail closed: an unknown provider, or one anonymous mode rules out, stops the start.
_provider_problems = office.provider_problems()
if _provider_problems:
    for _problem in _provider_problems:
        logger.error("Model settings: %s", _problem)
    raise SystemExit("The Office did not start: " + "; ".join(_provider_problems))
for _desk_id, _model in office.desk_models():
    _resolved = resolve(_model)
    if not api_key(_resolved.provider):
        logger.warning("Desk %s runs on %s but %s is not set", _desk_id, _resolved.label, _resolved.key_env)

publisher = DailyPublisher(office, office_config.get("office") or {})

_monitor_queues: list[asyncio.Queue] = []
_frame_log_at = 0.0


async def _on_monitor_frame(frame: dict) -> None:
    global _frame_log_at
    jpeg = frame.get("jpeg") or b""
    if not jpeg:
        return
    message = {
        "agentId": frame.get("context"),
        "url": frame.get("url") or "",
        "jpeg": base64.b64encode(jpeg).decode("ascii"),
    }
    import time as _time

    now = _time.monotonic()
    if now - _frame_log_at >= 2.0:
        _frame_log_at = now
        logger.info(
            "Desk monitor frame for %s (%s bytes, %s clients, %s)",
            message["agentId"],
            len(jpeg),
            len(_monitor_queues),
            (message["url"] or "")[:80],
        )
    for queue in list(_monitor_queues):
        if queue.full():
            try:
                queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
        try:
            queue.put_nowait(message)
        except asyncio.QueueFull:
            pass


def _latest_monitor_messages() -> list[dict]:
    out = []
    for context, frame in list(_latest_frames.items()):
        jpeg = frame.get("jpeg") or b""
        if not jpeg:
            continue
        out.append(
            {
                "agentId": context,
                "url": frame.get("url") or "",
                "jpeg": base64.b64encode(jpeg).decode("ascii"),
            }
        )
    return out


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    add_frame_listener(_on_monitor_frame)
    await office.start()
    await publisher.start()
    if os.getenv("OFFICE_KICKOFF_ON_START", "").lower() in {"1", "true", "yes"}:
        try:
            result = await publisher.kickoff(force=True)
            logger.info("Startup kickoff: %s", result)
        except Exception:
            logger.exception("Startup kickoff failed")
    yield
    await publisher.stop()
    remove_frame_listener(_on_monitor_frame)
    await office.stop()


app = FastAPI(
    title="The Office",
    lifespan=_lifespan,
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=_LOCAL_ORIGINS,
    allow_methods=["GET", "HEAD", "OPTIONS", "POST", "PATCH"],
    allow_headers=["Content-Type"],
)

_SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})
_INTERACTIVE_PATHS = frozenset({
    "/api/tasks",
    "/api/notes",
    "/api/nodes",
    "/api/nodes/archive",
})


def _forbidden(detail: str = "This office is view-only. Set OFFICE_READ_ONLY=false to assign work."):
    return JSONResponse(status_code=403, content={"detail": detail})


@app.middleware("http")
async def guard_writes(request, call_next):
    method = request.method.upper()
    if method in _SAFE_METHODS:
        return await call_next(request)
    if READ_ONLY:
        return _forbidden()
    path = request.url.path.rstrip("/") or "/"
    if method in {"POST", "PATCH"} and path in _INTERACTIVE_PATHS:
        return await call_next(request)
    return _forbidden("That write is not part of interactive mode.")


class TaskIn(BaseModel):
    agent_id: str = Field(min_length=1, max_length=80)
    prompt: str = Field(min_length=1, max_length=8000)
    title: str = Field(default="", max_length=180)


class NoteIn(BaseModel):
    title: str = Field(default="", max_length=180)
    content: str = Field(min_length=1, max_length=8000)
    assignee_id: str | None = Field(default=None, max_length=80)


class NodePatch(BaseModel):
    id: str = Field(min_length=1, max_length=200)
    title: str | None = Field(default=None, max_length=180)
    content: str | None = Field(default=None, max_length=8000)


class ArchiveIn(BaseModel):
    id: str = Field(min_length=1, max_length=200)


def _require_interactive() -> None:
    if READ_ONLY:
        raise HTTPException(status_code=403, detail="View-only mode.")


def _voice_available() -> bool:
    """Live voice is Gemini Live, so it needs Gemini as well as interactive mode."""
    return not READ_ONLY and gemini_extras_available()


@app.get("/api/health")
async def health():
    cfg = office_config.get("office") or {}
    return {
        "status": "ok",
        "read_only": READ_ONLY,
        "mode": "view-only" if READ_ONLY else "interactive",
        "anonymous": allowed_providers() is not None,
        "voice_available": _voice_available(),
        "host": HOST,
        "office": cfg.get("name"),
        "articles_per_day": cfg.get("articles_per_day", 3),
        "agents": len(office.agents),
        "schedule": publisher.schedule_snapshot(),
    }


@app.get("/api/office")
async def get_office():
    snap = office.snapshot()
    snap["read_only"] = READ_ONLY
    snap["mode"] = "view-only" if READ_ONLY else "interactive"
    snap["voice_available"] = _voice_available()
    cfg = office_config.get("office") or {}
    blog_url = os.getenv("BLOGGER_BLOG_URL", "").strip() or (cfg.get("blog") or {}).get("url") or ""
    snap["config"] = {
        "name": cfg.get("name"),
        "articles_per_day": cfg.get("articles_per_day", 3),
        "timezone": cfg.get("timezone"),
        "blog_url": blog_url or None,
        "publish_slots": publisher.schedule_snapshot().get("publish_slots"),
        "scheduler_enabled": publisher.enabled,
    }
    return snap


@app.post("/api/tasks")
async def create_task(body: TaskIn):
    _require_interactive()
    if body.agent_id not in office.agents:
        raise HTTPException(status_code=404, detail="Unknown agent")
    posted = await office.dispatch_task(
        body.agent_id,
        body.prompt.strip(),
        author="You",
        title=body.title.strip(),
        allow_duplicate=True,
    )
    return posted


@app.post("/api/notes")
async def create_note(body: NoteIn):
    _require_interactive()
    node_id = await office.post_note(
        body.title.strip() or "Nota",
        body.content.strip(),
        author="You",
        assignee_id=body.assignee_id,
    )
    return {"id": node_id}


@app.patch("/api/nodes")
async def patch_node(body: NodePatch):
    _require_interactive()
    if body.id not in office.board._node_registry:
        raise HTTPException(status_code=404, detail="Unknown card")
    await office.update_node_text(body.id, body.title, body.content)
    return {"ok": True, "id": body.id}


@app.post("/api/nodes/archive")
async def archive_node(body: ArchiveIn):
    _require_interactive()
    if body.id not in office.board._node_registry:
        raise HTTPException(status_code=404, detail="Unknown card")
    await office.archive_node(body.id, agent_name="You")
    return {"ok": True, "id": body.id}


@app.get("/api/jobs")
async def list_jobs(include_archived: bool = False):
    return {"jobs": office.list_jobs(include_archived=include_archived)}


@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str):
    return office.job_snapshot(job_id, include_archived=True)


@app.get("/api/events")
async def events():
    from fastapi.responses import StreamingResponse

    queue = office.subscribe()

    async def stream():
        try:
            while True:
                item = await queue.get()
                yield f"data: {json.dumps(item, ensure_ascii=False)}\n\n"
        finally:
            office.unsubscribe(queue)

    return StreamingResponse(stream(), media_type="text/event-stream")


@app.websocket("/ws/monitors")
async def monitors(websocket: WebSocket):
    await websocket.accept()
    queue: asyncio.Queue = asyncio.Queue(maxsize=8)
    _monitor_queues.append(queue)
    for snapshot in _latest_monitor_messages():
        try:
            await websocket.send_json(snapshot)
        except Exception:
            break
    try:
        while True:
            message = await queue.get()
            await websocket.send_json(message)
    except WebSocketDisconnect:
        pass
    finally:
        if queue in _monitor_queues:
            _monitor_queues.remove(queue)


@app.websocket("/ws/live-voice")
async def live_voice(websocket: WebSocket, agent_id: str):
    if READ_ONLY:
        await websocket.accept()
        await websocket.send_json({
            "event": "error",
            "message": "View-only mode. Set OFFICE_READ_ONLY=false to talk to a desk.",
        })
        await websocket.close()
        return
    if not gemini_extras_available():
        await websocket.accept()
        await websocket.send_json({
            "event": "error",
            "message": "Voice needs Gemini. Set GEMINI_API_KEY, and allow gemini if OFFICE_ALLOWED_PROVIDERS is set.",
        })
        await websocket.close()
        return
    try:
        await OfficeVoiceSocket(websocket, office, agent_id).run()
    except (WebSocketDisconnect, Exception):
        pass


def mount_frontend() -> None:
    dist = ROOT / "frontend" / "dist"
    index = dist / "index.html"
    if not index.is_file():
        logger.warning("Frontend index missing at %s — API only", index)
        return
    app.mount("/", StaticFiles(directory=dist, html=True), name="ui")


if __name__ == "__main__":
    import uvicorn

    mount_frontend()
    mode = "view-only" if READ_ONLY else "interactive"
    logger.info("The Office (%s) at http://%s:%s", mode, HOST, PORT)
    uvicorn.run(app, host=HOST, port=PORT)
