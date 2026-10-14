"""Board embeddings from any OpenAI-compatible /embeddings endpoint.

EpochDB calls ``encode`` and ``encode_batch``. This class answers with the
same shapes as EpochDB's built-in embedders, so Fireworks or a local server
can back board recall without EpochDB reading global OPENAI_* variables.
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

import httpx
import numpy as np

from runtime.providers import ResolvedModel, require

logger = logging.getLogger("office.embeddings")


class OpenAICompatibleEmbedder:
    def __init__(self, resolved: ResolvedModel, dim: int, timeout: float = 30.0):
        self.resolved = resolved
        self.dim = int(dim)
        self._timeout = timeout
        self._send_dimensions = True

    def _post(self, texts: list[str]) -> list[list[float]]:
        key = require(self.resolved)
        url = f"{self.resolved.base_url}/embeddings"
        headers = {"Authorization": f"Bearer {key}"}
        payload: dict[str, Any] = {"model": self.resolved.model, "input": texts}
        if self._send_dimensions:
            payload["dimensions"] = self.dim
        response = httpx.post(url, json=payload, headers=headers, timeout=self._timeout)
        if response.status_code == 400 and self._send_dimensions and "resiz" in response.text.lower():
            # Fixed-size models reject `dimensions`. Take their native size and fit it below.
            logger.info("%s has a fixed size; fitting its vectors to %s", self.resolved.label, self.dim)
            self._send_dimensions = False
            payload.pop("dimensions", None)
            response = httpx.post(url, json=payload, headers=headers, timeout=self._timeout)
        response.raise_for_status()
        rows = sorted(response.json().get("data") or [], key=lambda row: row.get("index", 0))
        if len(rows) != len(texts):
            raise RuntimeError(f"{self.resolved.label} returned {len(rows)} embeddings for {len(texts)} inputs")
        return [row["embedding"] for row in rows]

    def _fit(self, vector: Sequence[float], normalize: bool) -> np.ndarray:
        fitted = np.zeros(self.dim, dtype=np.float32)
        n = min(self.dim, len(vector))
        fitted[:n] = np.asarray(vector[:n], dtype=np.float32)
        if normalize:
            norm = float(np.linalg.norm(fitted))
            if norm > 1e-10:
                fitted /= norm
        return fitted

    def encode(self, text: str | Sequence[str], normalize_embeddings: bool = True, **_: Any) -> np.ndarray:
        if isinstance(text, (list, tuple)):
            return self.encode_batch(list(text), normalize_embeddings=normalize_embeddings)
        return self.encode_batch([str(text)], normalize_embeddings=normalize_embeddings)[0]

    def encode_batch(self, texts: Sequence[str], normalize_embeddings: bool = True, **_: Any) -> np.ndarray:
        items = [str(text) for text in texts]
        if not items:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.stack([self._fit(vector, normalize_embeddings) for vector in self._post(items)])
