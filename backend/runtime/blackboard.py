"""Durable newsroom board backed by EpochDB.

Jobs, tasks, handoffs, and results are memory atoms. The in-memory registry
is rebuilt from ``wal.jsonl`` on boot so a restart does not drop open work.
Semantic recall uses the same store: completed deliveries are embedded and
can be queried before the next CrewAI kickoff.
"""

from __future__ import annotations

import asyncio
import datetime
import json
import logging
import os
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from runtime.events import BlackboardEvent, EventType, LineageContext
from runtime.providers import ResolvedModel, api_key, is_allowed, resolve

logger = logging.getLogger("office.blackboard")

_EMBED_LABELS = {"Task", "Result", "Job", "Handoff", "Note"}


def _embedding_model() -> tuple[Optional[str], Optional[ResolvedModel]]:
    """(EpochDB model name, remote endpoint). A remote endpoint is served by the
    office's own embedder instead of one EpochDB loads."""
    chosen = os.getenv("OFFICE_EMBED_MODEL", "").strip()
    if chosen:
        if chosen.lower() == "off":
            return None, None
        prefix, sep, rest = chosen.partition("/")
        if sep and prefix.strip().lower() == "local":
            # Any sentence-transformers model. EpochDB runs it on this machine.
            return (rest.strip() or None), None
        resolved = resolve(chosen)
        if resolved.provider == "gemini":
            return f"google:{resolved.model}", None
        if resolved.provider == "deepseek":
            raise ValueError(
                "OFFICE_EMBED_MODEL: DeepSeek has no embeddings endpoint. "
                "Use gemini/, fireworks/, openai/, local/, or off."
            )
        return resolved.label, resolved
    model = (
        os.getenv("EPOCHDB_MODEL")
        or os.getenv("EPOCHDB_EMBEDDING_MODEL")
        or ""
    ).strip()
    if not model and api_key("gemini") and is_allowed("gemini"):
        model = "google:gemini-embedding-2"
    return (model or None), None


def _embedding_settings() -> tuple[Optional[str], int, Optional[ResolvedModel]]:
    model, remote = _embedding_model()
    name = model or ""
    dim_raw = os.getenv("EPOCHDB_DIM", "").strip()
    if dim_raw:
        dim = int(dim_raw)
    elif remote is not None:
        dim = 768
    elif "gemini-embedding-2" in name or "gemini-embedding-001" in name or "text-embedding-004" in name:
        dim = 768
    elif "text-embedding-3-small" in name:
        dim = 1536
    elif "gemini-embedding" in name or "text-embedding-3-large" in name:
        dim = 3072
    else:
        dim = 384
    return model, dim, remote


def _embedding_provider(model: Optional[str], remote: Optional[ResolvedModel]) -> Optional[str]:
    """The provider board embeddings call. None means local or off."""
    if remote is not None:
        return remote.provider
    name = (model or "").lower()
    # EpochDB sends these names to Google.
    if name.startswith(("google:", "gemini-embedding", "text-embedding-")):
        return "gemini"
    if name.startswith("openai:"):
        return "openai"
    return None


class EpochBlackboard:
    def __init__(self, storage_dir: str, dim: Optional[int] = None, model: Optional[str] = None):
        default_model, default_dim, remote = _embedding_settings()
        self._model = model if model is not None else default_model
        # An explicit model is an EpochDB name; only the configured default can be remote.
        self._remote = remote if model is None else None
        self._dim = int(dim) if dim is not None else default_dim
        self._storage_path = os.path.abspath(os.path.join(storage_dir, "global"))
        os.makedirs(self._storage_path, exist_ok=True)
        self._db = None
        self._version_store: Dict[str, int] = {}
        self._node_registry: Dict[str, Dict[str, Any]] = {}
        self._embed_cache: Dict[str, np.ndarray] = {}

    def _engine(self):
        if self._db is None:
            from epochdb import EpochDB

            kwargs: Dict[str, Any] = {
                "storage_dir": self._storage_path,
                "dim": self._dim,
                "model": self._model,
                "wal_use_uring": False,
                "auto_extract": False,
            }
            try:
                self._db = EpochDB(**kwargs)
            except TypeError:
                kwargs.pop("wal_use_uring", None)
                kwargs.pop("auto_extract", None)
                self._db = EpochDB(**kwargs)
            self._dim = int(getattr(self._db, "dim", self._dim))
            if self._remote is not None:
                from runtime.embeddings import OpenAICompatibleEmbedder

                # EpochDB has no embedder parameter. It uses a preset one instead of loading its own.
                self._db._embedder = OpenAICompatibleEmbedder(self._remote, dim=self._dim)
            logger.info(
                "EpochDB blackboard at %s (model=%s, dim=%s)",
                self._storage_path,
                self._model or "none",
                self._dim,
            )
        return self._db

    @property
    def embedding_provider(self) -> Optional[str]:
        """The provider board embeddings call, or None when they are local or off."""
        return _embedding_provider(self._model, self._remote)

    def _embed(self, text: str) -> np.ndarray:
        raw = (text or "").strip()
        if not raw or not self._model:
            return np.zeros(self._dim, dtype=np.float32)
        cached = self._embed_cache.get(raw)
        if cached is not None:
            return cached
        try:
            embedder = self._engine()._get_embedder()
            vector = np.array(
                embedder.encode(raw[:4000], normalize_embeddings=True),
                dtype=np.float32,
            )
        except Exception as exc:
            logger.warning("EpochDB embed failed; storing a zero vector: %s", exc)
            vector = np.zeros(self._dim, dtype=np.float32)
        if vector.shape[0] != self._dim:
            fitted = np.zeros(self._dim, dtype=np.float32)
            n = min(self._dim, int(vector.shape[0]))
            fitted[:n] = vector[:n]
            vector = fitted
        self._embed_cache[raw] = vector
        return vector

    async def write_node(
        self,
        node_id: str,
        label: str,
        properties: Dict[str, Any],
        vector: Optional[List[float]] = None,
        lineage: Optional[LineageContext] = None,
        expected_version: Optional[int] = None,
        priority: int = 100,
    ) -> BlackboardEvent:
        if node_id.startswith("global:") and lineage is None:
            raise ValueError(f"Global write {node_id} requires lineage")

        is_update = node_id in self._version_store
        if is_update:
            current = self._version_store[node_id]
            if expected_version is not None and expected_version != current:
                raise ValueError(
                    f"Version conflict on {node_id}: expected v{expected_version}, current v{current}"
                )
            new_version = current + 1
            event_type = EventType.NODE_UPDATED
        else:
            new_version = 1
            event_type = EventType.NODE_INSERTED

        payload = {"node_id": node_id, "label": label, **properties}
        if lineage:
            payload["_lineage"] = {
                "triggering_event_id": lineage.triggering_event_id,
                "agent_name": lineage.agent_name,
                "rationale": lineage.rationale,
                "sources_used": list(lineage.sources_used),
            }

        text = str(
            properties.get("text")
            or properties.get("content")
            or properties.get("result")
            or properties.get("title")
            or ""
        )
        should_embed = label in _EMBED_LABELS and len(text.strip()) > 24
        if vector is not None:
            embedding = np.array(vector, dtype=np.float32)
        elif should_embed:
            embedding = await asyncio.to_thread(self._embed, text)
        else:
            embedding = np.zeros(self._dim, dtype=np.float32)

        triples = [(node_id, "label", label)]
        for key, value in properties.items():
            if isinstance(value, str) and 0 < len(value) < 180:
                triples.append((node_id, key, value))

        def _commit() -> str:
            db = self._engine()
            if is_update and hasattr(db, "replace_memory"):
                return db.replace_memory(node_id, payload, embedding, triples)
            return db.add_memory(payload, embedding, triples, atom_id=node_id)

        await asyncio.to_thread(_commit)
        self._version_store[node_id] = new_version
        self._node_registry[node_id] = {
            "id": node_id,
            "label": label,
            "properties": properties,
            "atom_id": node_id,
            "priority": priority,
        }
        return BlackboardEvent(
            event_id=f"EV-{uuid.uuid4().hex[:8]}",
            event_type=event_type,
            key_path=node_id,
            payload=payload,
            epoch_version=new_version,
            timestamp=datetime.datetime.now(datetime.timezone.utc),
            priority=priority,
        )

    async def hydrate_registry(self, wal_path: Optional[str] = None) -> int:
        path = Path(wal_path) if wal_path else Path(self._storage_path) / "wal.jsonl"
        if not path.exists():
            logger.info("No EpochDB WAL to hydrate at %s", path)
            return 0
        latest, versions = await asyncio.to_thread(_scan_wal, path)
        self._node_registry.clear()
        self._version_store.clear()
        self._node_registry.update(latest)
        self._version_store.update(versions)
        logger.info("Hydrated %s blackboard nodes from %s", len(latest), path)
        return len(latest)

    async def recall(self, query: str, k: int = 4) -> str:
        """Semantic notes from EpochDB, injected into the next CrewAI task."""
        text = (query or "").strip()
        if not text or not self._model:
            return ""

        def _search() -> list:
            db = self._engine()
            if not hasattr(db, "recall_text"):
                return []
            return list(db.recall_text(text, top_k=k) or [])

        try:
            atoms = await asyncio.to_thread(_search)
        except Exception as exc:
            logger.warning("EpochDB recall failed: %s", exc)
            return ""

        lines: List[str] = []
        for atom in atoms:
            payload = getattr(atom, "payload", None)
            if isinstance(payload, dict):
                label = str(payload.get("label") or "Memory")
                body = str(
                    payload.get("text")
                    or payload.get("result")
                    or payload.get("title")
                    or ""
                ).strip()
                if not body:
                    continue
                lines.append(f"- [{label}] {body[:360]}")
            elif isinstance(payload, str) and payload.strip():
                lines.append(f"- {payload.strip()[:360]}")
        return "\n".join(lines[:k])

    async def remember_outcome(self, agent_name: str, task: str, outcome: str) -> None:
        """Store a prose delivery so later desks can recall it with query()."""
        body = (outcome or "").strip()
        if not body or not self._model:
            return
        note = (
            f"{agent_name} delivered: {body[:1800]}\n"
            f"Assignment: {(task or '').strip()[:400]}"
        )

        def _remember() -> None:
            db = self._engine()
            db.remember(
                note,
                metadata={
                    "triples": [(agent_name, "delivered", body[:120])],
                    "author": agent_name,
                    "kind": "delivery",
                },
            )

        try:
            await asyncio.to_thread(_remember)
        except Exception:
            logger.warning("EpochDB remember failed for %s", agent_name, exc_info=True)

    async def close(self) -> None:
        db = self._db
        self._db = None
        if db is not None and hasattr(db, "close"):
            await asyncio.to_thread(db.close)


def _scan_wal(path: Path) -> tuple[Dict[str, Dict[str, Any]], Dict[str, int]]:
    latest: Dict[str, Dict[str, Any]] = {}
    versions: Dict[str, int] = {}
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            op = event.get("op")
            data = event.get("data") if isinstance(event.get("data"), dict) else {}
            if op == "ADD":
                payload = data.get("payload") if isinstance(data.get("payload"), dict) else {}
                node_id = str(data.get("id") or payload.get("node_id") or "").strip()
                if not node_id or not payload.get("label"):
                    # Prose remember() atoms are not board cards.
                    continue
                label = str(payload.get("label") or "Unknown")
                properties = {
                    key: value
                    for key, value in payload.items()
                    if key not in ("node_id", "label", "_lineage")
                }
                versions[node_id] = versions.get(node_id, 0) + 1
                latest[node_id] = {
                    "id": node_id,
                    "label": label,
                    "properties": properties,
                    "atom_id": data.get("atom_id") or data.get("id"),
                    "priority": int(properties.get("priority") or 100),
                }
            elif op == "DELETE":
                node_id = str(data.get("id") or "").strip()
                if node_id:
                    latest.pop(node_id, None)
                    versions.pop(node_id, None)
    return latest, versions
