"""Event-driven office: EpochDB blackboard, router, and CrewAI desk agents."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from runtime.blackboard import EpochBlackboard
from runtime.events import EventType, LineageContext
from runtime.providers import (
    blocked,
    crew_llm,
    default_text_model,
    gemini_extras_available,
    require,
    resolve,
)
from runtime.router import EventRouter

logger = logging.getLogger("office.runtime")


def load_office_config(path: str | Path) -> Dict[str, Any]:
    """Load the YAML roster / schedule used by this autonomous newsroom."""
    import yaml

    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Office config not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Office config must be a mapping: {config_path}")
    return data

# Separate visual-brief handoffs stall the pipeline. Bete generates the cover herself.
_IMAGE_TASK_RE = re.compile(
    r"("
    r"(gerar|criar|criacao|criação|produzir|producao|produção|fazer|desenhar|renderizar|ilustrar)"
    r".{0,60}"
    r"(imagem|imagens|ilustracao|ilustração|ilustracoes|ilustrações|arte de capa|artes|"
    r"banner|thumbnail|capa visual|peca visual|peça visual|pecas visuais|peças visuais|"
    r"asset visual|assets visuais)"
    r"|"
    r"(midjourney|dall-?e|stable diffusion|imagen)"
    r"|"
    r"(generate|create|produce|draw|render).{0,40}(image|images|illustration|cover art|hero banner)"
    r")",
    re.IGNORECASE | re.DOTALL,
)


def is_image_generation_task(text: str) -> bool:
    """True when the brief asks to create binary images (unsupported in this office)."""
    return bool(_IMAGE_TASK_RE.search(text or ""))


_STATUS_HANDOFF_RE = re.compile(
    r"\b(conclu[ií]d[oa]|completo|completa|salvo em|saved (?:to|in)|"
    r"entregue|done|finished|completed|ready|pronto)\b",
    re.IGNORECASE,
)

_VISUAL_BRIEF_RE = re.compile(
    r"\b(brief\s+visual|prompts?(?:\s+de)?\s+(?:imagem|arte|capa)|"
    r"direcionamento\s+visual|publicacoes/prompts)\b",
    re.IGNORECASE,
)


def is_status_handoff(text: str) -> bool:
    """True when the handoff is a completion/status note, not a work transfer."""
    return bool(_STATUS_HANDOFF_RE.search(text or ""))


def is_visual_brief_task(text: str) -> bool:
    """True when the ask is a Markdown visual brief (Junior's real deliverable)."""
    return bool(_VISUAL_BRIEF_RE.search(text or ""))


def is_visual_designer(agent: Dict[str, Any] | None) -> bool:
    if not agent:
        return False
    blob = " ".join(
        str(agent.get(k) or "")
        for k in ("id", "name", "role", "systemPrompt")
    ).lower()
    return any(
        token in blob
        for token in ("junior", "designer", "artes", "visual", "ilustr")
    )


_TRANSIENT_LLM_RE = re.compile(
    r"(503|429|UNAVAILABLE|RESOURCE_EXHAUSTED|high demand|try again later|"
    r"temporarily unavailable|overloaded|deadline exceeded|timed?\s*out|"
    r"connection reset|connection refused|broken pipe|ssl|"
    r"InternalServerError|ServiceUnavailable|TooManyRequests)",
    re.IGNORECASE,
)


def is_transient_llm_error(exc: BaseException | str) -> bool:
    """True for capacity / network blips that should be retried, not permanent fails."""
    if isinstance(exc, (TimeoutError, ConnectionError, asyncio.TimeoutError, OSError)):
        return True
    name = type(exc).__name__ if not isinstance(exc, str) else ""
    text = f"{name} {exc}"
    code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    if code in (408, 429, 500, 502, 503, 504):
        return True
    return bool(_TRANSIENT_LLM_RE.search(text))


def _task_retry_settings() -> tuple[int, float, float]:
    attempts = max(1, int(os.getenv("OFFICE_TASK_RETRIES", "8")))
    base = max(1.0, float(os.getenv("OFFICE_RETRY_BASE_SEC", "5")))
    cap = max(base, float(os.getenv("OFFICE_RETRY_MAX_SEC", "90")))
    return attempts, base, cap


_PUBLISH_TASK_RE = re.compile(
    r"publicar|publique|postar|blogger|publish|post_to_blogger",
    re.IGNORECASE,
)


def is_publish_task(text: str) -> bool:
    return bool(_PUBLISH_TASK_RE.search(text or ""))


def article_paths_in(text: str) -> list[str]:
    return [p.lower() for p in re.findall(r"(?:artigos|publicacao)/[\w./-]+\.md", text or "", re.I)]


def _office_section(config: Dict[str, Any]) -> Dict[str, Any]:
    return dict(config.get("office") or {}) if isinstance(config, dict) else {}


def shared_instructions_from(config: Dict[str, Any]) -> str:
    """Optional policy text from office.yml — empty for a generic office."""
    return str(_office_section(config).get("shared_instructions") or "").strip()


def task_nudge_from(config: Dict[str, Any], key: str) -> str:
    nudges = _office_section(config).get("task_nudges") or {}
    if not isinstance(nudges, dict):
        return ""
    return str(nudges.get(key) or "").strip()


def skip_visual_handoffs(config: Dict[str, Any]) -> bool:
    meta = _office_section(config)
    if "skip_visual_handoffs" in meta:
        return bool(meta.get("skip_visual_handoffs"))
    return False


def blocked_visual_handoff_reply(config: Dict[str, Any]) -> str:
    return str(
        _office_section(config).get("blocked_visual_handoff_reply")
        or "SKIPPED: visual / image handoffs are disabled for this office."
    ).strip()


def _safe_markdown_path(root: Path, relative_path: str) -> Path:
    raw = (relative_path or "").strip().replace("\\", "/").lstrip("/")
    if not raw:
        raise ValueError("relative_path is required")
    if not raw.lower().endswith(".md"):
        raw += ".md"
    target = (root / raw).resolve()
    root = root.resolve()
    if root != target and root not in target.parents:
        raise ValueError("Path must stay inside the company workspace")
    return target


def markdown_tools(workspace: Path) -> list:
    """Safe Markdown read/write tools shared by the company."""
    from runtime.text_similarity import best_similar_match, text_similarity
    from runtime.tools import Tool

    async def save_markdown(relative_path: str, content: str, force: bool = False) -> str:
        """Save UTF-8 Markdown inside the company workspace.

        Refuses to overwrite or create a near-duplicate article unless force=True.
        """
        target = _safe_markdown_path(workspace, relative_path)
        content = content or ""
        force = bool(force)

        # Exact path already has same/similar body
        if target.exists() and not force:
            existing = target.read_text(encoding="utf-8")
            if text_similarity(existing[:4000], content[:4000]) >= 0.78:
                return (
                    f"DUPLICATE: '{target.relative_to(workspace)}' already has very similar content "
                    f"({len(existing)} chars). Do not rewrite. Reuse this file. "
                    f"Pass force=true only if you must intentionally replace it."
                )

        # Similar filename or body elsewhere in the workspace
        if not force and content.strip():
            candidates = []
            for path in workspace.rglob("*.md"):
                if not path.is_file():
                    continue
                rel = str(path.relative_to(workspace))
                candidates.append((rel, path.stem.replace("-", " ").replace("_", " ")))
                try:
                    snippet = path.read_text(encoding="utf-8")[:2500]
                except Exception:
                    continue
                if snippet.strip():
                    candidates.append((rel, snippet[:500]))
            stem_query = target.stem.replace("-", " ").replace("_", " ")
            match = best_similar_match(stem_query, ((c[0], c[1]) for c in candidates), threshold=0.78)
            if match and match[0] != str(target.relative_to(workspace)):
                return (
                    f"DUPLICATE: similar Markdown already exists at '{match[0]}' "
                    f"(similarity {match[2]:.2f}). Reuse that file instead of creating another. "
                    f"Pass force=true only for an intentional rewrite."
                )
            body_match = best_similar_match(
                content[:800],
                ((c[0], c[1]) for c in candidates if len(c[1]) > 80),
                threshold=0.8,
            )
            if body_match and body_match[0] != str(target.relative_to(workspace)):
                return (
                    f"DUPLICATE: body is very similar to '{body_match[0]}' "
                    f"(similarity {body_match[2]:.2f}). Do not save a second copy."
                )

        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return f"Saved {target.relative_to(workspace)} ({len(content)} characters)"

    async def read_markdown(relative_path: str) -> str:
        """Read a Markdown file from the company workspace."""
        target = _safe_markdown_path(workspace, relative_path)
        if not target.exists():
            return f"Markdown file not found: {target.relative_to(workspace)}"
        text = target.read_text(encoding="utf-8")
        return text[:120_000]

    async def list_markdown(directory: str = "") -> str:
        """List Markdown files available to the team."""
        base = (workspace / (directory or "")).resolve()
        if workspace.resolve() != base and workspace.resolve() not in base.parents:
            raise ValueError("Directory must stay inside the company workspace")
        if not base.exists():
            return "(no Markdown files)"
        paths = sorted(
            str(path.relative_to(workspace))
            for path in base.rglob("*.md")
            if path.is_file()
        )
        return "\n".join(paths) if paths else "(no Markdown files)"

    return [
        Tool(
            save_markdown,
            name="save_markdown",
            description=(
                "Write a Markdown file into the shared company workspace. "
                "Use a relative .md path such as artigos/meu-artigo.md. "
                "Blocks near-duplicate files unless force=true."
            ),
        ),
        Tool(
            read_markdown,
            name="read_markdown",
            description="Read a Markdown file from the shared company workspace.",
        ),
        Tool(
            list_markdown,
            name="list_markdown",
            description="List Markdown files in the shared company workspace. Check this before writing a new article.",
        ),
    ]


def browser_tools(agent_id: str) -> list:
    """One Chromium page per agent. Frames are mirrored onto that desk's monitor."""
    from runtime.browser import _get_browser_state, browser_tool_list

    state = _get_browser_state(agent_id)
    state["headless"] = True
    state["mirror_monitor"] = True
    return browser_tool_list(agent_id)


def desk_model(spec: Dict[str, Any]) -> str:
    """The text model a desk runs on: its own ``model`` or the office default."""
    model = str(spec.get("model") or "").strip()
    # The live voice model is not a text model, so that desk writes on the default.
    if not model or model == "gemini-3.8-live":
        return default_text_model()
    return model

HIDDEN_LABELS = {"Agent", "AgentStatus"}
BOARD_LABELS = {"Task", "Result", "Note", "Handoff", "Job"}
TEAM_ID = os.getenv("OFFICE_TEAM_ID", "the-office-v1")

TOOL_FACTORIES = {
    "navigate_browser",
    "click_element",
    "input_text",
    "extract_text",
    "scroll_page",
    "press_key",
    "current_webpage",
    "save_markdown",
    "read_markdown",
    "list_markdown",
    "post_to_blogger",
    "list_blogger_posts",
    "get_blog_info",
    "generate_blog_image",
    "hand_off",
    "list_jobs",
    "close_job",
}

_TERMINAL_TASK_STATUSES = frozenset(
    {"completed", "failed", "done", "reassigned", "cancelled"}
)


ALL_TOOLS = [
    "navigate_browser",
    "click_element",
    "input_text",
    "extract_text",
    "scroll_page",
    "press_key",
    "current_webpage",
    "save_markdown",
    "read_markdown",
    "list_markdown",
    "post_to_blogger",
    "list_blogger_posts",
    "get_blog_info",
    "generate_blog_image",
    "hand_off",
    "list_jobs",
    "close_job",
]


def office_language_code() -> str:
    code = os.getenv("OFFICE_LANGUAGE", "pt-BR").strip().lower()
    if code.startswith("en"):
        return "en-US"
    return "pt-BR"


def office_language() -> str:
    if office_language_code().startswith("en"):
        return "English"
    return "Portuguese (Brazil)"


def language_rule() -> str:
    if office_language_code().startswith("en"):
        return (
            "The office language is English. "
            "Speak, write, and save every file only in English."
        )
    return (
        "O idioma do escritório é português do Brasil (pt-BR). "
        "Fale, escreva e salve todos os arquivos somente em português brasileiro. "
        "Nunca responda em inglês. Se a pergunta vier em português, a resposta "
        "falada e escrita deve ser em português."
    )


# Empty on purpose: the user hires and configures the company roster.
SEED_AGENTS: List[Dict[str, Any]] = []


def format_team_roster(
    agents: Dict[str, Dict[str, Any]],
    exclude_id: str = "",
    *,
    max_prompt_chars: int = 500,
) -> str:
    """Human-readable roster with each teammate's role and full instructions."""
    blocks: List[str] = []
    for item in agents.values():
        if exclude_id and item.get("id") == exclude_id:
            continue
        name = item.get("name") or item.get("id") or "?"
        role = (item.get("role") or "no role set").strip()
        prompt = (item.get("systemPrompt") or "").strip()
        if len(prompt) > max_prompt_chars:
            prompt = prompt[: max_prompt_chars - 3] + "..."
        if prompt:
            blocks.append(f"- {name}\n  Role: {role}\n  Description: {prompt}")
        else:
            blocks.append(f"- {name}\n  Role: {role}\n  Description: (none)")
    return "\n".join(blocks) if blocks else "(no teammates)"


def roster_colleague_hint(agents: Dict[str, Dict[str, Any]], exclude_id: str = "") -> str:
    """Compact name→role list for tool parameter descriptions."""
    parts = []
    for item in agents.values():
        if exclude_id and item.get("id") == exclude_id:
            continue
        name = item.get("name") or "?"
        role = (item.get("role") or "?").strip()
        parts.append(f"'{name}' ({role})")
    return "; ".join(parts) if parts else "(alone)"



def _lineage(agent_name: str, rationale: str, event_id: str = "") -> LineageContext:
    return LineageContext(
        triggering_event_id=event_id or f"EV-{uuid.uuid4().hex[:8]}",
        agent_name=agent_name,
        rationale=rationale,
    )


def _public_node(node_id: str, label: str, properties: Dict[str, Any]) -> Dict[str, Any]:
    author = (
        properties.get("author")
        or properties.get("assignee_name")
        or ""
    )
    title = properties.get("title") or label
    content = properties.get("text") or properties.get("content") or ""
    if not isinstance(content, str):
        content = json.dumps(content, ensure_ascii=False)
    safe_props = {}
    for key, value in properties.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            safe_props[key] = value
    return {
        "id": node_id,
        "label": label,
        "title": str(title)[:180],
        "content": content[:1200],
        "author": str(author)[:80],
        "archived": bool(properties.get("archived")),
        "properties": safe_props,
    }


class DeskAgent:
    """Wakes on task inserts addressed to this desk and runs them with CrewAI."""

    def __init__(self, spec: Dict[str, Any], router: EventRouter, board: EpochBlackboard, office: "Office"):
        self.spec = spec
        self.agent_id = spec["id"]
        self.office = office
        self._router = router
        self._running = False
        self._in_flight: set[str] = set()
        self._waiting: Dict[str, Dict[str, Any]] = {}
        self._deferred: set[str] = set()

    async def start(self) -> None:
        queue = await self._router.subscribe("global:*")
        self._running = True
        logger.info("%s listening for blackboard tasks", self.spec["name"])
        while self._running:
            try:
                event = await queue.get()
            except asyncio.CancelledError:
                break
            try:
                await self.handle(event)
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("%s failed handling a blackboard event", self.spec["name"])

    def _task_done(self, task_key: str) -> bool:
        return self.office.is_task_complete(task_key)

    async def _try_start_waiting(self) -> None:
        ready = [
            key for key, payload in list(self._waiting.items())
            if not payload.get("depends_on") or self._task_done(str(payload.get("depends_on")))
        ]
        for key in ready:
            payload = self._waiting.pop(key, None)
            if payload:
                await self._run_task(key, payload)

    async def _defer_retry(self, task_key: str, payload: Dict[str, Any], delay: float) -> None:
        """Re-open a transient failure after a cooldown so the desk can recover."""
        if task_key in self._deferred:
            return
        self._deferred.add(task_key)
        try:
            logger.info(
                "%s will retry %s in %.0fs after transient API error",
                self.spec["name"],
                task_key,
                delay,
            )
            await asyncio.sleep(delay)
            entry = self.office.board._node_registry.get(task_key) or {}
            props = dict(entry.get("properties") or {})
            status = str(props.get("status") or "")
            if status in ("completed", "reassigned", "cancelled"):
                return
            if task_key in self._in_flight:
                return
            # Fresh payload from the board (may have updated text/deps)
            merged = {**payload, **props}
            merged["status"] = "open"
            await self.office.update_task_status(
                task_key,
                merged,
                "open",
                self.spec["name"],
                "",
                result=str(props.get("result") or "")[:400],
            )
            await self._run_task(task_key, merged)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("%s deferred retry failed for %s", self.spec["name"], task_key)
        finally:
            self._deferred.discard(task_key)

    async def handle(self, event) -> None:
        if event.event_type not in (EventType.NODE_INSERTED, EventType.NODE_UPDATED):
            return
        payload = event.payload if isinstance(event.payload, dict) else {}
        label = payload.get("label")

        # Any completed task/result may unblock a dependent waiting on this desk
        if label in ("Result", "Task"):
            status = str(payload.get("status") or "")
            if label == "Result" or status in ("completed", "failed", "done"):
                await self._try_start_waiting()

        if label != "Task":
            return
        if payload.get("assignee") != self.agent_id:
            return

        task_key = event.key_path
        status = str(payload.get("status") or "open")
        if status in ("completed", "failed", "done", "reassigned", "cancelled"):
            self._waiting.pop(task_key, None)
            return
        if status == "in_progress" and task_key in self._in_flight:
            return
        if status == "in_progress":
            return

        if task_key in self._in_flight:
            return

        depends_on = str(payload.get("depends_on") or "").strip()
        if status == "blocked" or (depends_on and not self._task_done(depends_on)):
            self._waiting[task_key] = dict(payload)
            logger.info(
                "%s waiting on %s before starting %s",
                self.spec["name"],
                depends_on or "(blocked)",
                task_key,
            )
            return

        await self._run_task(task_key, payload)

    async def _run_task(self, task_key: str, payload: Dict[str, Any]) -> None:
        if task_key in self._in_flight:
            return
        prompt = str(payload.get("text") or "").strip()
        if not prompt:
            return
        depth = int(payload.get("depth") or 0)
        depends_on = str(payload.get("depends_on") or "").strip()
        if depends_on and not self._task_done(depends_on):
            self._waiting[task_key] = dict(payload)
            return

        # Do not auto-complete as "already done". Topic overlap across roles used to
        # mark misrouted tasks complete and then block the correct hand_off.
        # Dedup of the same assignee's open twin happens at dispatch time only.

        self._in_flight.add(task_key)
        self._waiting.pop(task_key, None)
        await self.office.update_task_status(
            task_key,
            payload,
            "in_progress",
            self.spec["name"],
            "",
        )
        await self.office.set_status(self.agent_id, "thinking", "using_computer")

        max_attempts, base_delay, max_delay = _task_retry_settings()
        failed = False
        transient = False
        reply = ""
        last_exc: BaseException | None = None

        for attempt in range(1, max_attempts + 1):
            try:
                reply = await self.office.complete_task(
                    self, prompt, depth, parent_task_key=task_key
                )
                failed = False
                transient = False
                last_exc = None
                break
            except Exception as exc:
                last_exc = exc
                transient = is_transient_llm_error(exc)
                if transient and attempt < max_attempts:
                    delay = min(base_delay * (2 ** (attempt - 1)), max_delay)
                    logger.warning(
                        "%s hit transient API error on %s (%s/%s); retry in %.0fs: %s",
                        self.spec["name"],
                        task_key,
                        attempt,
                        max_attempts,
                        delay,
                        exc,
                    )
                    await self.office.update_task_status(
                        task_key,
                        payload,
                        "in_progress",
                        self.spec["name"],
                        "",
                        result=(
                            f"API temporariamente indisponível "
                            f"(tentativa {attempt}/{max_attempts}). "
                            f"Nova tentativa em {delay:.0f}s…"
                        ),
                    )
                    await self.office.set_status(
                        self.agent_id, "thinking", "using_computer"
                    )
                    await asyncio.sleep(delay)
                    continue
                logger.exception("Desk agent failed")
                reply = f"Não consegui concluir a tarefa: {exc}"
                failed = True
                break

        try:
            current = self.office.board._node_registry.get(task_key) or {}
            cur_status = str((current.get("properties") or {}).get("status") or "")
            # hand_off may have already closed this as reassigned — do not overwrite
            if cur_status in ("reassigned", "cancelled"):
                logger.info(
                    "%s leaving task %s as %s (handoff already transferred it)",
                    self.spec["name"],
                    task_key,
                    cur_status,
                )
            elif failed and transient:
                # Keep the task alive: park as open and schedule another pass.
                note = (
                    f"Falha transitória da API — reabertura agendada. "
                    f"Último erro: {last_exc}"
                )
                await self.office.update_task_status(
                    task_key,
                    payload,
                    "open",
                    self.spec["name"],
                    "",
                    result=note[:1000],
                )
                cooldown = min(max_delay * 1.5, 180.0)
                asyncio.create_task(
                    self._defer_retry(task_key, dict(payload), cooldown),
                    name=f"retry-{task_key[-12:]}",
                )
            else:
                result_props = {
                    "title": prompt[:80],
                    "text": reply,
                    "author": self.spec["name"],
                    "assignee": self.agent_id,
                    "assignee_name": self.spec["name"],
                    "task_key": task_key,
                    "job_id": str(payload.get("job_id") or task_key),
                    "status": "delivered",
                }
                await self.office.write_node(
                    f"global:result:{uuid.uuid4().hex[:12]}",
                    "Result",
                    result_props,
                    agent_name=self.spec["name"],
                    rationale="Task result posted to the shared blackboard",
                )
                await self.office.update_task_status(
                    task_key,
                    payload,
                    "failed" if failed else "completed",
                    self.spec["name"],
                    "",
                    result=reply,
                )
                await self.office.release_dependents(task_key)
                job_id = str(payload.get("job_id") or task_key)
                await self.office.refresh_job_status(
                    job_id, agent_name=self.spec["name"]
                )
        finally:
            self._in_flight.discard(task_key)
            await self.office.set_status(self.agent_id, "idle", "using_computer")
            await self._try_start_waiting()


class Office:
    def __init__(
        self,
        storage_dir: str,
        config: Optional[Dict[str, Any]] = None,
        config_path: Optional[str | Path] = None,
    ):
        self.storage_dir = storage_dir
        os.makedirs(storage_dir, exist_ok=True)
        self.config_path = Path(config_path) if config_path else None
        self.config = config or {}
        office_meta = self.config.get("office") or {}
        self.team_id = str(office_meta.get("team_id") or TEAM_ID)
        self.router = EventRouter()
        self.board = EpochBlackboard(storage_dir=os.path.join(storage_dir, "blackboard"))
        self.agents: Dict[str, Dict[str, Any]] = {}
        self._workers: Dict[str, DeskAgent] = {}
        self._worker_tasks: Dict[str, asyncio.Task] = {}
        self._listeners: List[asyncio.Queue] = []
        self._fanout_task: Optional[asyncio.Task] = None
        self._recover_task: Optional[asyncio.Task] = None
        self._client = None
        self._agents_path = os.path.join(storage_dir, "agents.json")
        default_workspace = Path(storage_dir).resolve().parent.parent / "workspace"
        self.workspace = Path(
            os.getenv(
                "COMPANY_WORKSPACE",
                os.getenv("OFFICE_WORKSPACE", str(default_workspace)),
            )
        ).resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)
        blog = office_meta.get("blog") or {}
        if blog.get("url") and not os.getenv("BLOGGER_BLOG_URL"):
            os.environ["BLOGGER_BLOG_URL"] = str(blog["url"])
        if office_meta.get("language"):
            os.environ.setdefault("OFFICE_LANGUAGE", str(office_meta["language"]))

    def _gemini(self):
        if self._client is None:
            from google import genai
            api_key = os.getenv("GEMINI_API_KEY", "")
            if not api_key or api_key == "MY_GEMINI_API_KEY":
                raise RuntimeError("GEMINI_API_KEY is required")
            self._client = genai.Client(api_key=api_key)
        return self._client

    def desk_models(self) -> List[tuple[str, str]]:
        """(desk id, model) for every desk in the config, before the office starts."""
        return [(str(spec.get("id")), desk_model(spec)) for spec in self._load_specs()]

    def provider_problems(self) -> List[str]:
        """Model settings that must stop the start: unknown providers, or ones anonymous mode blocks."""
        problems: List[str] = []
        entries: List[tuple[str, Optional[str]]] = []
        for desk_id, model in self.desk_models():
            try:
                entries.append((f"desk '{desk_id}' ({model})", resolve(model).provider))
            except ValueError as exc:
                problems.append(f"desk '{desk_id}': {exc}")
        entries.append(("board embeddings", self.board.embedding_provider))
        return problems + blocked(entries)

    async def start(self) -> None:
        queue = await self.router.subscribe("*")
        self._fanout_task = asyncio.create_task(self._fanout(queue))
        # Restore jobs/tasks/handoffs from durable Epoch WAL before desks wake.
        try:
            hydrated = await self.board.hydrate_registry()
            logger.info("Restored %s blackboard nodes from epoch storage", hydrated)
        except Exception:
            logger.exception("Failed to hydrate blackboard registry from disk")
        specs = self._load_specs()
        for spec in specs:
            await self._register(spec, persist=False)
        reopened = await self.recover_transient_failures()
        if reopened:
            logger.info("Reopened %s task(s) after prior transient API failures", reopened)
        cancelled = await self.cancel_orphaned_visual_tasks()
        if cancelled:
            logger.info("Cancelled %s orphaned visual / missing-assignee task(s)", cancelled)
        self._recover_task = asyncio.create_task(self._recover_loop())
        resumed = await self.resume_open_tasks()
        if resumed:
            logger.info("Resumed %s open task(s) from prior sessions", resumed)
        logger.info("Office started with %s agents", len(self.agents))

    async def stop(self) -> None:
        if self._recover_task:
            self._recover_task.cancel()
            try:
                await self._recover_task
            except asyncio.CancelledError:
                pass
            self._recover_task = None
        for task in list(self._worker_tasks.values()):
            task.cancel()
        if self._worker_tasks:
            await asyncio.gather(*self._worker_tasks.values(), return_exceptions=True)
        if self._fanout_task:
            self._fanout_task.cancel()

    async def cancel_orphaned_visual_tasks(self) -> int:
        """Drop open visual work and tasks for agents no longer on the roster."""
        cancelled = 0
        for node_id, entry in list(self.board._node_registry.items()):
            if not isinstance(entry, dict) or entry.get("label") != "Task":
                continue
            props = dict(entry.get("properties") or {})
            if props.get("archived"):
                continue
            status = str(props.get("status") or "")
            if status not in ("open", "blocked", "in_progress"):
                continue
            text = str(props.get("text") or "")
            assignee = str(props.get("assignee") or "")
            missing = bool(assignee) and assignee not in self._workers
            visual = (
                is_image_generation_task(text) or is_visual_brief_task(text)
            ) and not is_publish_task(text)
            if not missing and not visual:
                continue
            reason = (
                "Cancelled: assignee no longer on the roster."
                if missing
                else "Cancelled: no separate visuals desk — the publisher generates the cover."
            )
            props["status"] = "cancelled"
            props["result"] = reason
            await self.write_node(
                node_id,
                "Task",
                props,
                agent_name="Office",
                rationale="Cancel orphaned visual / missing-agent task",
            )
            await self.release_dependents(node_id)
            cancelled += 1
            logger.info("Cancelled orphaned task %s (%s)", node_id, reason)
        return cancelled

    async def recover_transient_failures(self) -> int:
        """Reopen tasks that died on Gemini 503 / overload so desks can resume."""
        reopened = 0
        for node_id, entry in list(self.board._node_registry.items()):
            if not isinstance(entry, dict) or entry.get("label") != "Task":
                continue
            props = dict(entry.get("properties") or {})
            if props.get("archived"):
                continue
            status = str(props.get("status") or "")
            blob = f"{props.get('result') or ''} {props.get('text') or ''}"
            if status == "failed" and is_transient_llm_error(blob):
                assignee = str(props.get("assignee") or "")
                props["status"] = "open"
                props["result"] = (
                    "Reaberta automaticamente após falha transitória da API."
                )
                await self.write_node(
                    node_id,
                    "Task",
                    props,
                    agent_name="Office",
                    rationale="Recover transient LLM failure — reopen task",
                )
                reopened += 1
                logger.info("Recovered failed task %s for %s", node_id, assignee or "?")
            elif status == "in_progress":
                # Crash mid-flight: put back to open so a desk picks it up again.
                worker = self._workers.get(str(props.get("assignee") or ""))
                if worker and node_id in worker._in_flight:
                    continue
                props["status"] = "open"
                props["result"] = "Reaberta após interrupção (estava in_progress)."
                await self.write_node(
                    node_id,
                    "Task",
                    props,
                    agent_name="Office",
                    rationale="Recover stuck in_progress task",
                )
                reopened += 1
        return reopened

    async def resume_open_tasks(self) -> int:
        """After hydration, wake desks for tasks that were still open last session."""
        resumed = 0
        for node_id, entry in list(self.board._node_registry.items()):
            if not isinstance(entry, dict) or entry.get("label") != "Task":
                continue
            props = dict(entry.get("properties") or {})
            if props.get("archived"):
                continue
            status = str(props.get("status") or "")
            if status not in ("open", "blocked"):
                continue
            assignee = str(props.get("assignee") or "")
            worker = self._workers.get(assignee)
            if worker is None:
                continue
            if node_id in worker._in_flight:
                continue
            depends_on = str(props.get("depends_on") or "").strip()
            if status == "blocked" or (depends_on and not self.is_task_complete(depends_on)):
                worker._waiting[node_id] = dict(props)
                continue
            # Fire-and-forget so start() isn't blocked on long agent loops.
            asyncio.create_task(
                worker._run_task(node_id, dict(props)),
                name=f"resume-{node_id[-12:]}",
            )
            resumed += 1
        return resumed

    async def _recover_loop(self) -> None:
        interval = max(15.0, float(os.getenv("OFFICE_RECOVER_INTERVAL_SEC", "60")))
        await asyncio.sleep(20)
        while True:
            try:
                n = await self.recover_transient_failures()
                if n:
                    logger.info("Recovery sweep reopened %s task(s)", n)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Recovery sweep failed")
            await asyncio.sleep(interval)

    def _yaml_specs(self) -> List[Dict[str, Any]]:
        agents = self.config.get("agents") or []
        loaded: List[Dict[str, Any]] = []
        for item in agents:
            if not isinstance(item, dict) or not item.get("id"):
                continue
            spec = dict(item)
            tools = [t for t in (spec.get("tools") or ALL_TOOLS) if t in TOOL_FACTORIES or t in ALL_TOOLS]
            spec["tools"] = tools or list(ALL_TOOLS)
            loaded.append(spec)
        return loaded

    def _load_specs(self) -> List[Dict[str, Any]]:
        # YAML is the source of truth for this autonomous office.
        yaml_specs = self._yaml_specs()
        if yaml_specs:
            return yaml_specs
        if os.path.exists(self._agents_path):
            with open(self._agents_path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
            if isinstance(data, dict) and data.get("team") == self.team_id and data.get("agents"):
                return [dict(item) for item in data["agents"]]
        return [dict(item) for item in SEED_AGENTS]

    def _persist_specs(self) -> None:
        with open(self._agents_path, "w", encoding="utf-8") as handle:
            json.dump(
                {"team": self.team_id, "agents": list(self.agents.values())},
                handle,
                ensure_ascii=False,
                indent=2,
            )

    def find_agent(self, name: str) -> Optional[Dict[str, Any]]:
        wanted = (name or "").strip().lower()
        if not wanted:
            return None
        # Exact name / id first
        for spec in self.agents.values():
            full = (spec.get("name") or "").lower()
            if wanted == full or wanted == (spec.get("id") or "").lower():
                return spec
        # First name unique match
        first_hits = []
        for spec in self.agents.values():
            full = (spec.get("name") or "").lower()
            first = full.split(" ")[0] if full else ""
            if wanted == first:
                first_hits.append(spec)
        if len(first_hits) == 1:
            return first_hits[0]
        if len(first_hits) > 1:
            return None  # ambiguous — caller should list options
        # Substring on name
        name_hits = []
        for spec in self.agents.values():
            full = (spec.get("name") or "").lower()
            if full and (wanted in full or full in wanted):
                name_hits.append(spec)
        if len(name_hits) == 1:
            return name_hits[0]
        if len(name_hits) > 1:
            return None
        # Role match — prefer the longest (most specific) role that contains wanted
        role_hits: List[tuple[int, Dict[str, Any]]] = []
        for spec in self.agents.values():
            role = (spec.get("role") or "").lower()
            if not role:
                continue
            if wanted == role or wanted in role or role in wanted:
                role_hits.append((len(role), spec))
        if not role_hits:
            return None
        role_hits.sort(key=lambda item: item[0], reverse=True)
        # If top two roles are equally long and both match, ambiguous
        if len(role_hits) > 1 and role_hits[0][0] == role_hits[1][0]:
            a = role_hits[0][1].get("name")
            b = role_hits[1][1].get("name")
            if a != b:
                return None
        return role_hits[0][1]

    def _team_options_text(self, exclude_id: str = "") -> str:
        return ", ".join(
            f"{s['name']} ({s.get('role') or '?'})"
            for s in self.agents.values()
            if s.get("id") != exclude_id
        )

    async def _fanout(self, queue: asyncio.Queue) -> None:
        while True:
            event = await queue.get()
            message = self._event_message(event)
            for listener in list(self._listeners):
                try:
                    listener.put_nowait(message)
                except asyncio.QueueFull:
                    pass

    def _event_message(self, event) -> Dict[str, Any]:
        payload = event.payload if isinstance(event.payload, dict) else {}
        label = str(payload.get("label") or "")
        properties = {
            key: value
            for key, value in payload.items()
            if key not in ("node_id", "label", "_lineage")
        }
        node = _public_node(event.key_path, label, properties)
        return {
            "kind": "node",
            "event_type": event.event_type.value if isinstance(event.event_type, EventType) else str(event.event_type),
            "key_path": event.key_path,
            "epoch_version": event.epoch_version,
            "node": node,
        }

    def subscribe(self) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue(maxsize=200)
        self._listeners.append(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue) -> None:
        if queue in self._listeners:
            self._listeners.remove(queue)

    async def write_node(
        self,
        node_id: str,
        label: str,
        properties: Dict[str, Any],
        agent_name: str,
        rationale: str,
        triggering_event_id: str = "",
    ):
        event = await self.board.write_node(
            node_id,
            label,
            properties,
            lineage=_lineage(agent_name, rationale, triggering_event_id),
        )
        await self.router.publish(event)
        return event

    async def _register(self, spec: Dict[str, Any], persist: bool) -> Dict[str, Any]:
        incoming_tools = spec.get("tools")
        if isinstance(incoming_tools, list) and incoming_tools:
            tools = [t for t in incoming_tools if t in ALL_TOOLS]
        else:
            tools = list(ALL_TOOLS)
        incoming_voice = str(spec.get("voiceName") or "").strip()
        spec = {
            "status": "idle",
            "currentActivity": "using_computer",
            "activeItem": "computer",
            "gender": "male",
            "skills": [],
            "model": default_text_model(),
            **spec,
            "tools": tools or list(ALL_TOOLS),
        }
        # voiceGender is the old name for the avatar/voice gender.
        if not spec.get("gender") and spec.get("voiceGender"):
            spec["gender"] = spec.get("voiceGender")
        spec.pop("voiceGender", None)
        spec["voiceName"] = incoming_voice or ("Kore" if spec.get("gender") == "female" else "Puck")
        for key, low, high in (("voicePitch", 0.5, 2.0), ("voiceRate", 0.6, 1.4)):
            raw = spec.get(key)
            if raw is None or raw == "":
                spec.pop(key, None)
                continue
            try:
                spec[key] = min(high, max(low, float(raw)))
            except (TypeError, ValueError):
                spec.pop(key, None)
        self.agents[spec["id"]] = spec
        if persist:
            self._persist_specs()
        await self.write_node(
            f"global:agent:{spec['id']}",
            "Agent",
            {
                "title": spec["name"],
                "text": spec.get("role", ""),
                "author": "Office",
                "agent_id": spec["id"],
            },
            agent_name="Office",
            rationale="Agent registered on the blackboard",
        )
        await self._restart_worker(spec)
        return spec

    async def _restart_worker(self, spec: Dict[str, Any]) -> None:
        agent_id = spec["id"]
        existing = self._worker_tasks.pop(agent_id, None)
        if existing:
            existing.cancel()
            await asyncio.gather(existing, return_exceptions=True)
        worker = DeskAgent(spec, self.router, self.board, self)
        self._workers[agent_id] = worker
        self._worker_tasks[agent_id] = asyncio.create_task(worker.start())
        await asyncio.sleep(0)

    async def create_agent(self, spec: Dict[str, Any]) -> Dict[str, Any]:
        if not spec.get("id"):
            spec["id"] = f"agent-{uuid.uuid4().hex[:8]}"
        return await self._register(spec, persist=True)

    async def update_agent(self, agent_id: str, spec: Dict[str, Any]) -> Dict[str, Any]:
        current = dict(self.agents[agent_id])
        current.update(spec)
        current["id"] = agent_id
        return await self._register(current, persist=True)

    async def delete_agent(self, agent_id: str) -> None:
        task = self._worker_tasks.pop(agent_id, None)
        if task:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        self._workers.pop(agent_id, None)
        self.agents.pop(agent_id, None)
        self._persist_specs()

    async def set_status(self, agent_id: str, status: str, activity: str) -> None:
        spec = self.agents.get(agent_id)
        if not spec:
            return
        spec["status"] = status
        spec["currentActivity"] = activity
        await self.write_node(
            f"global:agentstatus:{agent_id}",
            "AgentStatus",
            {
                "title": spec["name"],
                "text": status,
                "author": spec["name"],
                "agent_id": agent_id,
                "status": status,
                "activity": activity,
            },
            agent_name=spec["name"],
            rationale="Agent activity changed",
        )

    async def update_task_status(
        self,
        node_id: str,
        original: Dict[str, Any],
        status: str,
        agent_name: str,
        triggering_event_id: str,
        result: str = "",
    ) -> None:
        properties = {
            key: value
            for key, value in original.items()
            if key not in ("node_id", "label", "_lineage")
        }
        properties["status"] = status
        if status == "in_progress":
            properties["started_by"] = agent_name
        if result:
            properties["result"] = result[:1000]
        await self.write_node(
            node_id,
            "Task",
            properties,
            agent_name=agent_name,
            rationale=f"Task moved to {status}",
            triggering_event_id=triggering_event_id,
        )

    async def dispatch_task(
        self,
        agent_id: str,
        prompt: str,
        author: str = "User",
        depth: int = 0,
        title: str = "",
        depends_on: str = "",
        *,
        allow_duplicate: bool = False,
        reopen_blocked: bool = False,
        job_id: str = "",
        parent_task_id: str = "",
    ) -> Dict[str, Any]:
        spec = self.agents[agent_id]

        # Teammate hand_offs must always reach the new assignee. Prior "already done"
        # checks blocked reassignment when the wrong person got the task first.
        if not allow_duplicate:
            prior = self.find_similar_delivery(prompt, assignee_id=agent_id, threshold=0.95)
            if prior:
                logger.info(
                    "Not dispatching duplicate task to %s — matches their prior %s (%s)",
                    spec["name"],
                    prior["id"],
                    prior["kind"],
                )
                return {
                    "id": prior["id"],
                    "assignee": agent_id,
                    "assignee_name": spec["name"],
                    "status": "duplicate",
                    "depends_on": None,
                    "job_id": prior.get("job_id") or job_id or None,
                    "duplicate_of": prior,
                    "message": (
                        f"{spec['name']} already completed an equivalent task "
                        f"('{prior['title']}', similarity {prior['score']:.2f}). "
                        "Not creating another for them. Pick a different teammate "
                        "if this work belongs to someone else."
                    ),
                }

        twin = self.find_similar_open_task(prompt, assignee_id=agent_id, threshold=0.92)
        if twin:
            twin_id = twin["id"]
            twin_status = twin["status"]
            # Misrouted work often leaves a blocked twin waiting on the wrong agent.
            # Reopen it so the correct person can actually start.
            if twin_status == "blocked" and (reopen_blocked or allow_duplicate):
                entry = self.board._node_registry.get(twin_id) or {}
                props = dict(entry.get("properties") or {})
                props["status"] = "open"
                props.pop("depends_on", None)
                props["text"] = prompt
                props["author"] = author
                props["depth"] = depth
                if job_id:
                    props["job_id"] = job_id
                if parent_task_id:
                    props["parent_task_id"] = parent_task_id
                await self.write_node(
                    twin_id,
                    "Task",
                    props,
                    agent_name=author,
                    rationale="Reopened blocked handoff so the correct assignee can start",
                )
                return {
                    "id": twin_id,
                    "assignee": agent_id,
                    "assignee_name": spec["name"],
                    "status": "open",
                    "depends_on": None,
                    "job_id": props.get("job_id") or job_id or twin_id,
                    "message": (
                        f"Reopened blocked task {twin_id.split(':')[-1]} for {spec['name']} "
                        f"('{twin['title']}'). They can start now."
                    ),
                }
            # Reuse an already-open twin instead of spawning a clone
            if twin_status in ("open", "in_progress") or not allow_duplicate:
                twin_entry = self.board._node_registry.get(twin_id) or {}
                twin_job = str((twin_entry.get("properties") or {}).get("job_id") or twin_id)
                return {
                    "id": twin_id,
                    "assignee": agent_id,
                    "assignee_name": spec["name"],
                    "status": twin_status,
                    "depends_on": twin.get("depends_on"),
                    "job_id": twin_job,
                    "duplicate_of": twin,
                    "message": (
                        f"Similar task already {twin_status} for {spec['name']} "
                        f"(id={twin_id.split(':')[-1]}): '{twin['title']}'. "
                        "Reusing that assignment."
                    ),
                }

        node_id = f"global:task:{uuid.uuid4().hex[:12]}"
        # Root user requests open a Job; handoffs inherit the parent's job_id.
        if job_id:
            resolved_job = self._normalize_job_id(job_id)
        elif parent_task_id:
            parent_entry = self.board._node_registry.get(parent_task_id) or {}
            parent_job = str(
                (parent_entry.get("properties") or {}).get("job_id") or parent_task_id
            )
            resolved_job = self._normalize_job_id(parent_job)
        else:
            resolved_job = self._normalize_job_id(node_id)
        task_title = title.strip() or f"{spec['name']}: {prompt[:70]}"
        dep = (depends_on or "").strip()
        blocked = bool(dep) and not self.is_task_complete(dep)
        properties: Dict[str, Any] = {
            "title": task_title,
            "text": prompt,
            "author": author,
            "assignee": agent_id,
            "assignee_name": spec["name"],
            "depth": depth,
            "status": "blocked" if blocked else "open",
            "job_id": resolved_job,
        }
        if parent_task_id:
            properties["parent_task_id"] = parent_task_id
        if dep:
            properties["depends_on"] = dep
        await self.write_node(
            node_id,
            "Task",
            properties,
            agent_name=author,
            rationale="Task assigned on the shared blackboard",
        )
        # Create / refresh the Job card so the board shows the whole request
        if not parent_task_id:
            await self.ensure_job(
                resolved_job,
                title=prompt[:100] or task_title,
                brief=prompt[:800],
                author=author,
                seed_task_id=node_id,
            )
        else:
            await self.refresh_job_status(resolved_job, agent_name=author)
        return {
            "id": node_id,
            "assignee": agent_id,
            "assignee_name": spec["name"],
            "status": properties["status"],
            "depends_on": dep or None,
            "job_id": resolved_job,
        }

    def find_similar_delivery(
        self,
        prompt: str,
        assignee_id: str = "",
        exclude_task_key: str = "",
        threshold: float = 0.88,
    ) -> Optional[Dict[str, Any]]:
        """Return a prior completed Task by the same assignee that matches this prompt.

        Only compares assignment text (not results, other agents, or files) so a
        researcher's finished work does not cancel a writer's related task.
        """
        from runtime.text_similarity import best_similar_match

        # Same article path already published by this assignee → block republication.
        paths = article_paths_in(prompt) if is_publish_task(prompt) else []
        if paths:
            needle = paths[0]
            for node_id, entry in self.board._node_registry.items():
                if exclude_task_key and node_id == exclude_task_key:
                    continue
                if (entry.get("label") or "") != "Task":
                    continue
                props = entry.get("properties") or {}
                status = str(props.get("status") or "")
                if status not in ("completed", "done"):
                    continue
                if assignee_id and str(props.get("assignee") or "") != assignee_id:
                    continue
                blob = f"{props.get('text') or ''} {props.get('result') or ''}".lower()
                if needle in blob:
                    return {
                        "id": node_id,
                        "kind": "Task",
                        "title": str(props.get("title") or "")[:120],
                        "job_id": props.get("job_id"),
                        "summary": str(props.get("result") or props.get("text") or "")[:280],
                        "score": 1.0,
                    }

        candidates = []
        for node_id, entry in self.board._node_registry.items():
            if exclude_task_key and node_id == exclude_task_key:
                continue
            if (entry.get("label") or "") != "Task":
                continue
            props = entry.get("properties") or {}
            status = str(props.get("status") or "")
            if status not in ("completed", "done"):
                continue
            if assignee_id and str(props.get("assignee") or "") != assignee_id:
                continue
            # Compare only the original assignment text — not titles/results,
            # which share topic words across different roles.
            blob = str(props.get("text") or "").strip()
            if blob:
                candidates.append((node_id, blob[:1200]))

        match = best_similar_match(prompt, candidates, threshold=threshold)
        if not match:
            return None
        node_id, label_text, score = match
        entry = self.board._node_registry.get(node_id) or {}
        props = entry.get("properties") or {}
        return {
            "id": node_id,
            "kind": "Task",
            "title": str(props.get("title") or label_text)[:120],
            "job_id": props.get("job_id"),
            "summary": str(props.get("result") or props.get("text") or label_text)[:280],
            "score": score,
        }

    def find_similar_open_task(
        self,
        prompt: str,
        assignee_id: str = "",
        threshold: float = 0.9,
    ) -> Optional[Dict[str, Any]]:
        from runtime.text_similarity import best_similar_match

        paths = article_paths_in(prompt) if is_publish_task(prompt) else []
        if paths:
            needle = paths[0]
            for node_id, entry in self.board._node_registry.items():
                if entry.get("label") != "Task":
                    continue
                props = entry.get("properties") or {}
                status = str(props.get("status") or "open")
                if status not in ("open", "in_progress", "blocked"):
                    continue
                if assignee_id and str(props.get("assignee") or "") != assignee_id:
                    continue
                if needle in str(props.get("text") or "").lower():
                    return {
                        "id": node_id,
                        "title": str(props.get("title") or "")[:120],
                        "status": status,
                        "depends_on": props.get("depends_on"),
                    }

        candidates = []
        meta: Dict[str, Dict[str, Any]] = {}
        for node_id, entry in self.board._node_registry.items():
            if entry.get("label") != "Task":
                continue
            props = entry.get("properties") or {}
            status = str(props.get("status") or "open")
            if status not in ("open", "in_progress", "blocked"):
                continue
            if assignee_id and str(props.get("assignee") or "") != assignee_id:
                continue
            blob = str(props.get("text") or "").strip()
            if not blob:
                continue
            candidates.append((node_id, blob[:1000]))
            meta[node_id] = {
                "id": node_id,
                "title": str(props.get("title") or "")[:120],
                "status": status,
                "depends_on": props.get("depends_on"),
            }
        match = best_similar_match(prompt, candidates, threshold=threshold)
        if not match:
            return None
        return meta.get(match[0])

    def is_task_complete(self, task_key: str) -> bool:
        if not task_key:
            return True
        entry = self.board._node_registry.get(task_key)
        if not entry:
            return False
        props = entry.get("properties") or {}
        return str(props.get("status") or "") in (
            "completed",
            "failed",
            "done",
            "reassigned",
            "cancelled",
        )

    async def release_dependents(self, parent_key: str) -> None:
        """Flip blocked tasks that waited on parent_key to open so desks wake up."""
        if not parent_key:
            return
        for node_id, entry in list(self.board._node_registry.items()):
            if entry.get("label") != "Task":
                continue
            props = dict(entry.get("properties") or {})
            if str(props.get("depends_on") or "") != parent_key:
                continue
            if str(props.get("status") or "") != "blocked":
                continue
            props["status"] = "open"
            await self.write_node(
                node_id,
                "Task",
                props,
                agent_name="Office",
                rationale=f"Dependency {parent_key} completed — task unblocked",
            )

    def latest_task_key_for(
        self,
        assignee_id: str,
        author: str = "",
    ) -> Optional[str]:
        """Most recent Task node for an assignee (optionally authored by someone)."""
        matches: List[tuple[str, Dict[str, Any]]] = []
        for node_id, entry in self.board._node_registry.items():
            if entry.get("label") != "Task":
                continue
            props = entry.get("properties") or {}
            if str(props.get("assignee") or "") != assignee_id:
                continue
            if author and str(props.get("author") or "") != author:
                continue
            matches.append((node_id, props))
        if not matches:
            return None
        return matches[-1][0]

    def handoff_tool(self, worker: DeskAgent, depth: int, parent_task_key: str = ""):
        from runtime.tools import Tool
        from pydantic import BaseModel, Field

        office = self
        roster = format_team_roster(self.agents, exclude_id=worker.agent_id)
        colleague_hint = roster_colleague_hint(self.agents, exclude_id=worker.agent_id)

        class HandOffArgs(BaseModel):
            colleague: str = Field(
                description=(
                    "Exact teammate NAME from the roster (not a vague label). "
                    f"Valid teammates: {colleague_hint}. "
                    "Pick the person whose Role/Description matches this subtask."
                )
            )
            task: str = Field(description="Only their part of the work, in one or two sentences.")
            wait_for: str = Field(
                default="",
                description=(
                    "Optional. Teammate name who must finish first, or 'me' to wait until "
                    "your current task finishes. Leave empty if they can start immediately."
                ),
            )

        async def hand_off(colleague: str, task: str, wait_for: str = "") -> str:
            """Pass one part of the work to a teammate by exact name."""
            if depth >= 8:
                return "Handoff limit reached. Finish your own part and report it on the board."
            target = office.find_agent(colleague)
            if target is None:
                return (
                    f"No unique teammate matched '{colleague}'. "
                    f"Use an exact name from this roster:\n{roster}"
                )
            if target["id"] == worker.agent_id:
                return "That task is already yours. Do it yourself or pick a colleague."

            parent_props = {}
            if parent_task_key:
                parent_entry = office.board._node_registry.get(parent_task_key) or {}
                parent_props = dict(parent_entry.get("properties") or {})
            job_id = str(parent_props.get("job_id") or parent_task_key or "").strip()
            if job_id:
                job_id = office._normalize_job_id(job_id)

            depends_on = ""
            wait = (wait_for or "").strip()
            task_text = (task or "").strip()

            # Optional: block visual handoffs when configured for this office.
            if skip_visual_handoffs(office.config) and (
                is_image_generation_task(task_text) or is_visual_brief_task(task_text)
            ):
                return blocked_visual_handoff_reply(office.config)

            # Status notes ("concluído / salvo em …") must not block the recipient.
            if is_status_handoff(task_text):
                wait = ""

            # Publish work must never wait on Junior / binary visuals.
            if re.search(
                r"publicar|publique|postar|blogger|publish|post_to_blogger",
                task_text,
                re.IGNORECASE,
            ):
                if wait:
                    dep_probe = office.find_agent(wait)
                    if is_visual_designer(dep_probe) or wait.lower() in (
                        "junior",
                        "artes",
                        "designer",
                    ):
                        wait = ""

            if wait:
                wait_l = wait.lower()
                if wait_l in ("me", "eu", "mim", "myself", "self"):
                    # Completion notices with wait_for='me' deadlocked Junior:
                    # parent never completed because hand_off reassigned it.
                    if is_status_handoff(task_text) or is_visual_designer(
                        {"id": worker.agent_id, "name": worker.spec.get("name"), "role": worker.spec.get("role")}
                    ):
                        wait = ""
                        depends_on = ""
                    else:
                        depends_on = parent_task_key
                else:
                    dep_agent = office.find_agent(wait)
                    if dep_agent is None:
                        return (
                            f"wait_for '{wait}' not found. Use a teammate name "
                            f"or 'me'. Available: {office._team_options_text(worker.agent_id)}"
                        )
                    # Never gate publishing (or anyone) on the visuals desk.
                    if is_visual_designer(dep_agent):
                        depends_on = ""
                    else:
                        depends_on = (
                            office.latest_task_key_for(dep_agent["id"], author=worker.spec["name"])
                            or office.latest_task_key_for(dep_agent["id"])
                            or ""
                        )
                        if not depends_on:
                            return (
                                f"No prior task found for {dep_agent['name']} to wait on. "
                                f"Assign their work first, then hand_off with wait_for='{dep_agent['name']}'."
                            )

            posted = await office.dispatch_task(
                target["id"],
                task_text,
                author=worker.spec["name"],
                depth=depth + 1,
                depends_on=depends_on,
                # Publish handoffs must not spawn a second Bete task for the same article.
                allow_duplicate=not is_publish_task(task_text),
                reopen_blocked=True,
                job_id=job_id,
                parent_task_id=parent_task_key,
            )
            who = f"{target['name']} ({target.get('role') or 'no role'})"
            if posted.get("status") == "duplicate":
                return posted.get("message") or (
                    f"{who} already has equivalent work — not creating another publish task."
                )
            await office.write_node(
                f"global:handoff:{uuid.uuid4().hex[:12]}",
                "Handoff",
                {
                    "title": f"{worker.spec['name']} → {target['name']}",
                    "text": task_text,
                    "author": worker.spec["name"],
                    "assignee": target["id"],
                    "assignee_name": target["name"],
                    "depends_on": depends_on or None,
                    "task_id": posted["id"],
                    "job_id": posted.get("job_id") or job_id,
                },
                agent_name=worker.spec["name"],
                rationale="Teammate handoff posted on the blackboard",
            )
            child_short = posted["id"].split(":")[-1]
            job_short = str(posted.get("job_id") or job_id or "").split(":")[-1]

            # Full transfer of our current assignment (common misroute fix):
            # close our task as reassigned so we don't stay stuck and so any
            # dependents waiting on us can move to the real owner's work.
            # Skip status/completion notices — those used to mark Junior's finished
            # briefs as "reassigned" instead of completed.
            if (
                parent_task_key
                and not depends_on
                and not is_status_handoff(task_text)
            ):
                from runtime.text_similarity import text_similarity

                parent_entry = office.board._node_registry.get(parent_task_key) or {}
                parent_props = parent_entry.get("properties") or {}
                parent_text = str(parent_props.get("text") or "")
                if text_similarity(parent_text, task_text) >= 0.6:
                    note = (
                        f"Reassigned to {target['name']} "
                        f"(task {posted['id']}). Not this role's work."
                    )
                    await office.update_task_status(
                        parent_task_key,
                        dict(parent_props),
                        "reassigned",
                        worker.spec["name"],
                        "",
                        result=note,
                    )
                    await office.release_dependents(parent_task_key)
                    return (
                        f"Reassigned your current task to {who} as task id={child_short} "
                        f"(job={job_short}, status={posted.get('status')}). "
                        "They own delivery now — you are done with this assignment."
                    )

            if posted.get("status") == "blocked":
                return (
                    f"Assigned to {who} as task id={child_short} (job={job_short}, "
                    f"status=blocked — waiting on dependency before they start)."
                )
            if posted.get("message") and posted.get("status") == "open":
                return (
                    f"{posted['message']} ({who}, task id={child_short}, job={job_short})"
                )
            return (
                f"Assigned to {who} as task id={child_short} "
                f"(job={job_short}, status={posted.get('status')}). "
                "Track it on the blackboard until status=completed."
            )

        return Tool(
            hand_off,
            name="hand_off",
            description=(
                "Delegate a subtask to ONE teammate by their exact name. "
                "Read each teammate's Role and Description below and pick the person "
                "whose job matches this subtask — never a random name.\n"
                "If YOUR current assignment is not your role, hand_off the full task "
                "to the right teammate with wait_for empty so they start immediately "
                "(do not claim the work is already done).\n"
                f"TEAM ROSTER:\n{roster}"
            ),
            args_schema=HandOffArgs,
        )

    async def post_note(
        self,
        title: str,
        content: str,
        author: str,
        assignee_id: str | None = None,
    ) -> str:
        target_agent = None
        if assignee_id and assignee_id in self.agents:
            target_agent = self.agents[assignee_id]
        elif author and author not in ("User", "Admin", "You", "Office"):
            target_agent = self.find_agent(author)

        if target_agent is not None:
            prompt_text = content.strip() or title.strip()
            task_title = title.strip() or f"{target_agent['name']}: {prompt_text[:60]}"
            posted = await self.dispatch_task(
                target_agent["id"],
                prompt_text,
                author="User",
                title=task_title,
            )
            return posted["id"]

        node_id = f"global:note:{uuid.uuid4().hex[:12]}"
        await self.write_node(
            node_id,
            "Note",
            {"title": title or "Nota", "text": content, "author": author or "User"},
            agent_name=author or "User",
            rationale="Note pinned on the shared blackboard",
        )
        return node_id

    async def update_node_text(self, node_id: str, title: str | None, content: str | None) -> None:
        current = self.board._node_registry.get(node_id)
        if not current:
            return
        properties = dict(current.get("properties") or {})
        if title is not None and current.get("label") == "Note":
            properties["title"] = title
        if content is not None:
            if current.get("label") == "Task" and properties.get("status") in {"completed", "failed"}:
                properties["result"] = content
            else:
                properties["text"] = content
        await self.write_node(
            node_id,
            current.get("label") or "Note",
            properties,
            agent_name="User",
            rationale="Board card edited",
        )

    async def archive_node(self, node_id: str, agent_name: str = "User") -> None:
        current = self.board._node_registry.get(node_id)
        if not current:
            return
        properties = dict(current.get("properties") or {})
        if properties.get("archived"):
            return
        properties["archived"] = True
        await self.write_node(
            node_id,
            current.get("label") or "Note",
            properties,
            agent_name=agent_name,
            rationale="Card archived",
        )

    def _normalize_job_id(self, job_id: str) -> str:
        raw = (job_id or "").strip()
        if not raw:
            return ""
        if raw.startswith("global:job:"):
            return raw
        if raw.startswith("global:task:"):
            return f"global:job:{raw.split(':')[-1]}"
        return f"global:job:{raw.split(':')[-1]}"

    def _job_task_key(self, job_id: str) -> str:
        """Root task id that seeded this job (same suffix)."""
        short = self._normalize_job_id(job_id).split(":")[-1]
        return f"global:task:{short}"

    def _nodes_for_job(self, job_id: str) -> List[tuple[str, Dict[str, Any]]]:
        wanted = self._normalize_job_id(job_id)
        short = wanted.split(":")[-1]
        aliases = {wanted, f"global:task:{short}", short}
        found: List[tuple[str, Dict[str, Any]]] = []
        for node_id, entry in self.board._node_registry.items():
            label = entry.get("label") or ""
            if label not in ("Task", "Handoff", "Result", "Job"):
                continue
            props = entry.get("properties") or {}
            raw_job = str(props.get("job_id") or "").strip()
            if label == "Job" and (node_id == wanted or node_id.endswith(f":{short}")):
                found.append((node_id, entry))
                continue
            if not raw_job:
                # Root task without job_id yet: treat matching id as belonging to job
                if label == "Task" and node_id.endswith(f":{short}"):
                    found.append((node_id, entry))
                continue
            norm = self._normalize_job_id(raw_job)
            if norm in aliases or raw_job in aliases or raw_job.endswith(f":{short}") or raw_job == short:
                found.append((node_id, entry))
        return found

    def job_snapshot(self, job_id: str, include_archived: bool = False) -> Dict[str, Any]:
        job_node_id = self._normalize_job_id(job_id)
        tasks = []
        handoffs = []
        results = []
        job_props: Dict[str, Any] = {}
        for node_id, entry in self._nodes_for_job(job_id):
            props = dict(entry.get("properties") or {})
            if props.get("archived") and not include_archived:
                if (entry.get("label") or "") != "Job":
                    continue
            label = entry.get("label") or ""
            item = {
                "id": node_id,
                "title": str(props.get("title") or label)[:120],
                "text": str(props.get("text") or "")[:240],
                "status": str(props.get("status") or ""),
                "assignee_name": str(props.get("assignee_name") or props.get("author") or ""),
                "result": str(props.get("result") or "")[:240],
                "archived": bool(props.get("archived")),
            }
            if label == "Job":
                job_props = props
            elif label == "Task":
                tasks.append(item)
            elif label == "Handoff":
                handoffs.append(item)
            elif label == "Result":
                results.append(item)

        open_tasks = [
            t for t in tasks
            if t["status"] not in _TERMINAL_TASK_STATUSES and not t["archived"]
        ]
        done_tasks = [t for t in tasks if t["status"] in _TERMINAL_TASK_STATUSES]
        if job_props.get("status") in ("delivered", "closed", "archived"):
            status = str(job_props.get("status"))
        elif not tasks:
            status = "open"
        elif open_tasks:
            status = "in_progress"
        else:
            status = "ready_to_close"

        return {
            "id": job_node_id,
            "short_id": job_node_id.split(":")[-1],
            "title": str(job_props.get("title") or (tasks[0]["title"] if tasks else "Job")),
            "brief": str(job_props.get("text") or (tasks[0]["text"] if tasks else "")),
            "status": status,
            "summary": str(job_props.get("result") or job_props.get("summary") or ""),
            "author": str(job_props.get("author") or ""),
            "tasks": tasks,
            "handoffs": handoffs,
            "results": results,
            "open_count": len(open_tasks),
            "done_count": len(done_tasks),
            "task_count": len(tasks),
            "archived": bool(job_props.get("archived")),
        }

    def list_jobs(self, include_archived: bool = False) -> List[Dict[str, Any]]:
        job_ids: set[str] = set()
        for node_id, entry in self.board._node_registry.items():
            label = entry.get("label") or ""
            props = entry.get("properties") or {}
            if props.get("archived") and not include_archived and label != "Job":
                continue
            if label == "Job":
                job_ids.add(node_id)
                continue
            if label in ("Task", "Handoff", "Result"):
                raw = str(props.get("job_id") or "").strip()
                if raw:
                    job_ids.add(self._normalize_job_id(raw))
                elif label == "Task":
                    # legacy root tasks without job_id
                    job_ids.add(self._normalize_job_id(node_id))
        jobs = [self.job_snapshot(jid, include_archived=include_archived) for jid in job_ids]
        jobs.sort(key=lambda j: (j["status"] == "delivered", j["short_id"]), reverse=False)
        # Active first
        order = {"in_progress": 0, "open": 1, "ready_to_close": 2, "delivered": 3, "closed": 4}
        jobs.sort(key=lambda j: order.get(j["status"], 9))
        return jobs

    def jobs_board_text(self, limit: int = 8) -> str:
        lines = []
        for job in self.list_jobs()[:limit]:
            lines.append(
                f"- [Job id={job['short_id']}] status={job['status']} "
                f"tasks={job['done_count']}/{job['task_count']} open={job['open_count']} "
                f"| {job['title'][:80]}"
            )
            for task in job["tasks"][-6:]:
                tshort = task["id"].split(":")[-1]
                lines.append(
                    f"    · Task id={tshort} status={task['status'] or '?'} "
                    f"assignee={task['assignee_name'] or '?'} | {task['title'][:60]}"
                )
            if job.get("summary"):
                lines.append(f"    delivery: {job['summary'][:140]}")
        return "\n".join(lines) if lines else "(no jobs yet)"

    async def ensure_job(
        self,
        job_id: str,
        *,
        title: str,
        brief: str,
        author: str,
        seed_task_id: str = "",
    ) -> str:
        node_id = self._normalize_job_id(job_id)
        current = self.board._node_registry.get(node_id)
        if current:
            props = dict(current.get("properties") or {})
            if seed_task_id and not props.get("seed_task_id"):
                props["seed_task_id"] = seed_task_id
            props["job_id"] = node_id
            await self.write_node(
                node_id,
                "Job",
                props,
                agent_name=author or "Office",
                rationale="Job card refreshed",
            )
            return node_id
        await self.write_node(
            node_id,
            "Job",
            {
                "title": (title or "Job")[:120],
                "text": (brief or title or "")[:800],
                "author": author or "User",
                "status": "open",
                "job_id": node_id,
                "seed_task_id": seed_task_id or self._job_task_key(node_id),
            },
            agent_name=author or "Office",
            rationale="Job opened on the shared blackboard",
        )
        return node_id

    async def refresh_job_status(self, job_id: str, agent_name: str = "Office") -> Optional[Dict[str, Any]]:
        if not job_id:
            return None
        snap = self.job_snapshot(job_id, include_archived=True)
        node_id = snap["id"]
        current = self.board._node_registry.get(node_id)
        props = dict((current or {}).get("properties") or {})
        if not current:
            await self.ensure_job(
                node_id,
                title=snap["title"],
                brief=snap["brief"],
                author=agent_name,
            )
            props = dict((self.board._node_registry.get(node_id) or {}).get("properties") or {})
        if props.get("status") in ("delivered", "closed"):
            return snap
        props["status"] = snap["status"] if snap["status"] != "ready_to_close" else "ready_to_close"
        props["title"] = props.get("title") or snap["title"]
        props["text"] = props.get("text") or snap["brief"]
        props["job_id"] = node_id
        props["open_count"] = snap["open_count"]
        props["done_count"] = snap["done_count"]
        props["task_count"] = snap["task_count"]
        await self.write_node(
            node_id,
            "Job",
            props,
            agent_name=agent_name,
            rationale="Job progress updated",
        )
        return self.job_snapshot(node_id)

    async def close_job(
        self,
        job_id: str,
        *,
        summary: str = "",
        archive_tickets: bool = True,
        agent_name: str = "Office",
        force: bool = False,
    ) -> Dict[str, Any]:
        snap = self.job_snapshot(job_id, include_archived=True)
        if snap["open_count"] > 0 and not force:
            open_ids = [
                t["id"].split(":")[-1]
                for t in snap["tasks"]
                if t["status"] not in _TERMINAL_TASK_STATUSES and not t["archived"]
            ]
            return {
                "ok": False,
                "error": (
                    f"Job {snap['short_id']} still has {snap['open_count']} open task(s): "
                    f"{', '.join(open_ids)}. Finish or reassign them first, or pass force=true."
                ),
                "job": snap,
            }

        node_id = snap["id"]
        current = self.board._node_registry.get(node_id)
        props = dict((current or {}).get("properties") or {})
        props.update(
            {
                "title": props.get("title") or snap["title"],
                "text": props.get("text") or snap["brief"],
                "author": props.get("author") or agent_name,
                "status": "delivered",
                "result": (summary or snap.get("summary") or "Job delivered.")[:1000],
                "job_id": node_id,
                "open_count": 0,
                "done_count": snap["task_count"],
                "task_count": snap["task_count"],
                "closed_by": agent_name,
            }
        )
        await self.write_node(
            node_id,
            "Job",
            props,
            agent_name=agent_name,
            rationale="Job marked delivered",
        )

        archived = []
        if archive_tickets:
            for node_id_child, entry in self._nodes_for_job(job_id):
                label = entry.get("label") or ""
                if label == "Job":
                    continue
                child_props = entry.get("properties") or {}
                if child_props.get("archived"):
                    continue
                await self.archive_node(node_id_child, agent_name=agent_name)
                archived.append(node_id_child)

        return {
            "ok": True,
            "job": self.job_snapshot(job_id, include_archived=True),
            "archived": archived,
            "message": (
                f"Job {snap['short_id']} delivered. "
                f"Archived {len(archived)} ticket(s). Job card kept on the board as ENTREGUE."
            ),
        }

    def job_tools(self, worker: "DeskAgent"):
        from runtime.tools import Tool
        from pydantic import BaseModel, Field

        office = self

        class CloseJobArgs(BaseModel):
            job_id: str = Field(
                description="Job id (short hex or global:job:...). From the blackboard Jobs list."
            )
            summary: str = Field(
                description="Short delivery summary for the whole job (what was produced)."
            )
            archive_tickets: bool = Field(
                default=True,
                description="Archive finished Task/Handoff/Result tickets under this job.",
            )
            force: bool = Field(
                default=False,
                description="Close even if some tasks are still open (avoid unless stuck).",
            )

        class ListJobsArgs(BaseModel):
            include_archived: bool = Field(
                default=False,
                description="Include tickets already archived under each job.",
            )

        async def list_jobs(include_archived: bool = False) -> str:
            """List company jobs with task ids and status."""
            jobs = office.list_jobs(include_archived=include_archived)
            if not jobs:
                return "No jobs on the board yet."
            lines = []
            for job in jobs:
                lines.append(
                    f"Job id={job['short_id']} status={job['status']} "
                    f"progress={job['done_count']}/{job['task_count']} "
                    f"| {job['title']}"
                )
                for task in job["tasks"]:
                    mark = "✓" if task["status"] in _TERMINAL_TASK_STATUSES else "…"
                    lines.append(
                        f"  {mark} task={task['id'].split(':')[-1]} "
                        f"[{task['status']}] {task['assignee_name']}: {task['title'][:70]}"
                    )
                if job.get("summary"):
                    lines.append(f"  delivery: {job['summary'][:200]}")
            return "\n".join(lines)

        async def close_job(
            job_id: str,
            summary: str,
            archive_tickets: bool = True,
            force: bool = False,
        ) -> str:
            """Mark a job delivered and optionally archive its tickets."""
            result = await office.close_job(
                job_id,
                summary=summary,
                archive_tickets=archive_tickets,
                agent_name=worker.spec.get("name") or "Agent",
                force=force,
            )
            if not result.get("ok"):
                return result.get("error") or "Could not close job."
            return result["message"]

        return [
            Tool(
                list_jobs,
                name="list_jobs",
                description=(
                    "Show jobs on the blackboard: each job groups related tasks created "
                    "during one company request. Check this before claiming work is done."
                ),
                args_schema=ListJobsArgs,
            ),
            Tool(
                close_job,
                name="close_job",
                description=(
                    "When the overall company request is delivered, close the job: "
                    "write a delivery summary and archive finished tickets. "
                    "Only after every task under that job is completed/reassigned "
                    "(or use force if stuck). Keeps the Job card as ENTREGUE."
                ),
                args_schema=CloseJobArgs,
            ),
        ]

    def snapshot(self) -> Dict[str, Any]:
        nodes = []
        for node_id, entry in self.board._node_registry.items():
            nodes.append(_public_node(node_id, entry.get("label") or "", entry.get("properties") or {}))
        return {
            "agents": list(self.agents.values()),
            "nodes": nodes,
            "jobs": self.list_jobs(include_archived=True),
        }

    def board_summary(self, limit: int = 24) -> str:
        """Human-readable board with task IDs and lifecycle status for agents."""
        items: List[tuple[int, str]] = []
        for idx, (node_id, entry) in enumerate(self.board._node_registry.items()):
            label = entry.get("label") or ""
            if label in HIDDEN_LABELS:
                continue
            props = entry.get("properties") or {}
            if props.get("archived"):
                continue
            short_id = node_id.split(":")[-1] if ":" in node_id else node_id
            title = str(props.get("title") or label)[:80]
            text = str(props.get("text") or "")[:140]
            if label == "Task":
                status = str(props.get("status") or "open")
                who = str(props.get("assignee_name") or props.get("assignee") or "?")
                dep = str(props.get("depends_on") or "").strip()
                dep_short = dep.split(":")[-1] if dep else ""
                job = str(props.get("job_id") or "").strip()
                job_short = job.split(":")[-1] if job else short_id
                result = str(props.get("result") or "").strip()[:120]
                line = (
                    f"- [Task id={short_id} job={job_short}] "
                    f"status={status} assignee={who}"
                )
                if dep_short:
                    line += f" waits_on={dep_short}"
                line += f" | {title}"
                if text:
                    line += f" — {text}"
                if result and status in (
                    "completed", "failed", "done", "reassigned", "cancelled",
                ):
                    line += f" | delivery: {result}"
            elif label == "Result":
                who = str(props.get("assignee_name") or props.get("author") or "?")
                linked = str(props.get("task_key") or "").split(":")[-1]
                line = f"- [Result id={short_id}] by={who}"
                if linked:
                    line += f" for_task={linked}"
                line += f" | {title}: {text}"
            elif label == "Handoff":
                who = str(props.get("assignee_name") or "?")
                linked = str(props.get("task_id") or "").split(":")[-1]
                line = f"- [Handoff id={short_id}] → {who}"
                if linked:
                    line += f" created_task={linked}"
                line += f" | {title}: {text}"
            elif label == "Job":
                status = str(props.get("status") or "open")
                done = props.get("done_count")
                total = props.get("task_count")
                progress = f" {done}/{total}" if total is not None else ""
                line = (
                    f"- [Job id={short_id}] status={status}{progress} | {title}: {text}"
                )
                result = str(props.get("result") or "").strip()[:120]
                if result:
                    line += f" | delivery: {result}"
            else:
                line = f"- [{label} id={short_id}] {title}: {text}"
            items.append((idx, line))

        # Prefer recent items, but always keep open/in_progress/blocked tasks visible
        recent = items[-limit:]
        recent_ids = {line for _, line in recent}
        sticky = []
        for idx, line in items:
            if line in recent_ids:
                continue
            if "status=open" in line or "status=in_progress" in line or "status=blocked" in line:
                sticky.append((idx, line))
        chosen = (sticky + recent)[-limit:]
        return "\n".join(line for _, line in chosen) if chosen else "(blackboard vazio)"

    def task_brief(self, task_key: str) -> str:
        """One-line status of a task for prompts."""
        if not task_key:
            return "(no task id)"
        entry = self.board._node_registry.get(task_key) or {}
        props = entry.get("properties") or {}
        short = task_key.split(":")[-1]
        status = str(props.get("status") or "open")
        who = str(props.get("assignee_name") or "?")
        job = str(props.get("job_id") or task_key).split(":")[-1]
        dep = str(props.get("depends_on") or "").strip()
        dep_s = f" waits_on={dep.split(':')[-1]}" if dep else ""
        return f"id={short} job={job} status={status} assignee={who}{dep_s}"

    async def complete_task(
        self,
        worker: DeskAgent,
        prompt: str,
        depth: int = 0,
        parent_task_key: str = "",
    ) -> str:
        spec = worker.spec
        model = desk_model(spec)

        roster = format_team_roster(self.agents, exclude_id=worker.agent_id)
        my_task = self.task_brief(parent_task_key) if parent_task_key else "(ad-hoc)"
        parent_props = {}
        if parent_task_key:
            parent_props = dict(
                (self.board._node_registry.get(parent_task_key) or {}).get("properties") or {}
            )
        job_id = str(parent_props.get("job_id") or parent_task_key or "")
        job_short = job_id.split(":")[-1] if job_id else "?"

        system = (
            f"You are {spec.get('name')}, role: {spec.get('role')}.\n"
            f"Your description / instructions:\n{spec.get('systemPrompt') or '(none)'}\n\n"
            f"YOUR TEAMMATES (read carefully before any hand_off):\n{roster}\n\n"
            f"{language_rule()}\n"
            "Follow your own role and description. Do the work that belongs to you.\n"
            "When handing off, call hand_off with the teammate's EXACT name from the roster above, "
            "choosing the person whose Role and Description match that subtask.\n"
            "If this assignment is clearly NOT your role, hand_off the full task immediately "
            "with wait_for empty — never say the work is already done, and never wait on yourself.\n"
            "Speak in short plain sentences. No markdown in spoken replies. "
            "Never introduce yourself by name or role (no 'As X, the Y…' openings). "
            "The wall whiteboard is the shared blackboard.\n"
            "Every Task belongs to a Job (job id). Related handoffs share the same job. "
            "Statuses: open → in_progress → completed (or failed / reassigned). "
            "Your final answer is the delivery for YOUR task id — be explicit about what you "
            "produced (file paths, posts, decisions). Call list_jobs to see the whole request. "
            "When every task under the job is done and the company request is delivered, "
            "call close_job(job_id, summary=...) to archive the tickets and leave the Job as ENTREGUE.\n"
            "If teammate B needs teammate A's output, call hand_off(..., wait_for='A').\n"
            "Teammates' completed tasks are not yours — do your own role even if the "
            "topic overlaps. Only skip writing a second copy of a file YOU already saved "
            "(check your list/read tools for your own prior output).\n"
            + (
                f"{shared_instructions_from(self.config)}\n"
                if shared_instructions_from(self.config)
                else ""
            )
            + "Cookie banners are accepted automatically after navigation."
        )
        board = self.board_summary()
        jobs = self.jobs_board_text(limit=6)
        wanted = set(spec.get("tools") or ALL_TOOLS)
        tools = []
        md = markdown_tools(self.workspace)
        if any(name in wanted for name in ("save_markdown", "read_markdown", "list_markdown")):
            tools.extend([t for t in md if getattr(t, "name", None) in wanted])
        if any(
            name in wanted
            for name in (
                "navigate_browser",
                "click_element",
                "input_text",
                "extract_text",
                "scroll_page",
                "press_key",
                "current_webpage",
            )
        ):
            tools.extend(
                [t for t in browser_tools(worker.agent_id) if getattr(t, "name", None) in wanted]
            )
        if any(name in wanted for name in ("post_to_blogger", "list_blogger_posts", "get_blog_info")):
            from runtime.blogger import blogger_tools

            tools.extend([t for t in blogger_tools() if getattr(t, "name", None) in wanted])
        if "generate_blog_image" in wanted:
            from blog_images import blog_image_tools

            # Covers need Gemini. Without it the tool tells Bete to publish text only.
            images = self._gemini if gemini_extras_available() else None
            tools.extend(blog_image_tools(self.workspace, images))
        if any(name in wanted for name in ("list_jobs", "close_job")):
            tools.extend([t for t in self.job_tools(worker) if getattr(t, "name", None) in wanted])
        if depth < 8 and "hand_off" in wanted:
            tools.append(self.handoff_tool(worker, depth, parent_task_key=parent_task_key))
        dep_context = ""
        if parent_task_key:
            upstream = str(parent_props.get("depends_on") or "").strip()
            if upstream:
                up = self.board._node_registry.get(upstream) or {}
                up_props = up.get("properties") or {}
                up_result = str(up_props.get("result") or "")[:2000]
                if up_result:
                    dep_context = (
                        f"\nUpstream dependency ({self.task_brief(upstream)}):\n{up_result}\n"
                    )
        image_nudge = ""
        if is_image_generation_task(prompt) or is_visual_brief_task(prompt):
            skip_nudge = task_nudge_from(self.config, "skip_visuals")
            if skip_nudge:
                image_nudge = f"\n{skip_nudge}\n"
        publish_nudge = ""
        if is_publish_task(prompt or ""):
            pub = task_nudge_from(self.config, "publish")
            if pub:
                publish_nudge = f"\n{pub}\n"

        shared = shared_instructions_from(self.config)
        shared_block = f"{shared}\n" if shared else ""

        from runtime.crew_loop import run_desk_task

        return await run_desk_task(
            agent_name=spec.get("name") or "Agent",
            role=spec.get("role") or "",
            backstory=system,
            model=model,
            tools=tools,
            max_iterations=40,
            memory=self.board,
            prompt=(
                f"Shared company workspace: {self.workspace}\n"
                f"JOBS (request → tasks):\n{jobs}\n\n"
                f"Blackboard detail:\n{board}\n"
                f"{dep_context}\n"
                f"{language_rule()}\n"
                f"{shared_block}"
                f"{image_nudge}"
                f"{publish_nudge}"
                f"You are {spec.get('name')}.\n"
                f"Your role: {spec.get('role')}\n"
                f"Your instructions: {spec.get('systemPrompt') or '(none)'}\n\n"
                f"YOUR CURRENT TASK: {my_task}\n"
                f"Job id={job_short} (all related handoffs share this job).\n"
                f"Teammates you may hand_off to (name + role + description):\n{roster}\n\n"
                f"Your assigned work:\n{prompt}\n\n"
                "Do the parts that match your role yourself. "
                "For hand_off, use an exact teammate name from the list above whose role fits the subtask. "
                "If this whole assignment is not your role, hand_off it with wait_for empty and stop. "
                "Do not report the task as already completed unless you personally finished it. "
                "Use save_markdown for documents you create. Include saved paths in your final response. "
                "When you are done, your final answer IS the delivery for this task id — "
                "summarize what was delivered and any child task ids you created. "
                "If you are wrapping up the whole company request and list_jobs shows no open tasks, "
                "call close_job to archive tickets. "
                "The office will mark this task completed when you finish."
            ),
        )

    async def complete_text(self, spec: Dict[str, Any], prompt: str) -> str:
        resolved = resolve(desk_model(spec))
        system = (
            f"You are {spec.get('name')}, {spec.get('role')}. "
            f"{spec.get('systemPrompt') or ''} "
            f"{language_rule()} "
            "Answer in at most 4 sentences, with no markdown. "
            "Do not introduce yourself by name or role. "
            "What you conclude should be readable on the shared blackboard."
        )
        board = self.board_summary()
        user = f"Quadro agora:\n{board}\n\nTarefa:\n{prompt}"
        if resolved.provider == "gemini":
            require(resolved)
            response = await self._gemini().aio.models.generate_content(
                model=resolved.model,
                contents=user,
                config={"system_instruction": system, "temperature": 0.6},
            )
            text = (getattr(response, "text", None) or "").strip()
        else:
            llm = crew_llm(resolved, temperature=0.6)
            reply = await llm.acall([
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ])
            text = str(reply or "").strip()
        if not text:
            raise RuntimeError(f"Modelo {resolved.label} não devolveu texto")
        return text
