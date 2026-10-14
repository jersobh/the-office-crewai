"""One cover image per Blogger post: Gemini generates it, Blogger hosts it."""

from __future__ import annotations

import base64
import html
import logging
import os
import re
import unicodedata
from pathlib import Path
from typing import Any, Callable

import httpx

logger = logging.getLogger("office.images")

# Blogger's API does not accept image bytes. The compose UI uploads through
# this resumable photos endpoint and returns a googleusercontent URL.
_UPLOAD_START = "https://docs.google.com/upload/blogger/photos/resumable"

_DEFAULT_MODELS = ("gemini-3.1-flash-image", "gemini-2.5-flash-image")


def image_models() -> list[str]:
    primary = os.getenv("GEMINI_IMAGE_MODEL", _DEFAULT_MODELS[0]).strip()
    names: list[str] = []
    for name in (primary, *_DEFAULT_MODELS):
        if name and name not in names:
            names.append(name)
    return names


def slugify(title: str) -> str:
    text = unicodedata.normalize("NFKD", title or "")
    text = text.encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^a-zA-Z0-9]+", "-", text).strip("-").lower()
    return (text[:80] or "capa")


def cover_prompt(title: str, description: str) -> str:
    scene = (description or title or "technology news").strip()
    return (
        "Editorial illustration for a Brazilian technology blog. "
        "Wide composition, clean modern style, concrete objects and a clear focal point. "
        "No text, no letters, no numbers, no logos, no watermarks, no UI screenshots. "
        f"Article title: {(title or '').strip()}. Scene: {scene}."
    )


def cover_markdown(url: str) -> str:
    """HTML so the image survives both the Markdown package and the fallback converter."""
    safe = html.escape((url or "").strip(), quote=True)
    return f'<img src="{safe}" alt="capa">'


def public_image_url(raw: str) -> str:
    """Prefer the full-size Blogger image (s0) instead of a resized variant."""
    url = (raw or "").strip()
    if not url.startswith("http"):
        raise ValueError("upload did not return an image URL")
    if "/s0/" in url:
        return url
    parts = url.split("/")
    if len(parts) < 4:
        return url
    return "/".join(parts[:-1] + ["s0", parts[-1]])


def find_hosted_url(payload: Any) -> str:
    if isinstance(payload, str) and payload.startswith("http") and "googleusercontent.com" in payload:
        return payload
    if isinstance(payload, dict):
        for value in payload.values():
            found = find_hosted_url(value)
            if found:
                return found
    elif isinstance(payload, list):
        for value in payload:
            found = find_hosted_url(value)
            if found:
                return found
    return ""


def image_url_from_upload(payload: Any) -> str:
    hosted = find_hosted_url(payload)
    if not hosted:
        raise ValueError("Blogger upload response did not include an image URL")
    return public_image_url(hosted)


def cover_path(workspace: Path, title: str) -> Path:
    root = workspace.resolve()
    target = (root / "imagens" / f"{slugify(title)}.png").resolve()
    if root != target and root not in target.parents:
        raise ValueError("Cover path must stay inside the workspace")
    return target


def _part_bytes(part: Any) -> tuple[bytes, str] | None:
    inline = getattr(part, "inline_data", None)
    if inline is None:
        return None
    raw = getattr(inline, "data", None)
    if not raw:
        return None
    if isinstance(raw, str):
        raw = base64.b64decode(raw)
    mime = str(getattr(inline, "mime_type", None) or "image/png")
    return bytes(raw), mime


def _is_missing_model(exc: BaseException) -> bool:
    text = str(exc).lower()
    return any(token in text for token in ("404", "not_found", "not found", "unknown model", "invalid model"))


async def generate_cover_bytes(client: Any, title: str, description: str) -> tuple[bytes, str]:
    from google.genai import types

    prompt = cover_prompt(title, description)
    config = types.GenerateContentConfig(
        response_modalities=["IMAGE"],
        image_config=types.ImageConfig(aspect_ratio="16:9"),
    )
    last_error: BaseException | None = None
    for model in image_models():
        try:
            response = await client.aio.models.generate_content(
                model=model,
                contents=prompt,
                config=config,
            )
        except Exception as exc:
            last_error = exc
            if _is_missing_model(exc):
                logger.warning("Image model %s unavailable: %s", model, exc)
                continue
            raise
        candidates = getattr(response, "candidates", None) or []
        parts = []
        if candidates:
            content = getattr(candidates[0], "content", None)
            parts = list(getattr(content, "parts", None) or [])
        for part in parts:
            extracted = _part_bytes(part)
            if extracted and extracted[0]:
                return extracted
        last_error = RuntimeError(f"Model {model} returned no image")
        logger.warning("%s", last_error)
    raise RuntimeError(f"Image generation failed: {last_error}")


async def upload_blogger_image(access_token: str, data: bytes, filename: str, mime: str = "image/png") -> str:
    """Upload bytes to the blog's photo storage and return a public URL."""
    payload = {
        "protocolVersion": "0.8",
        "createSessionRequest": {
            "fields": [
                {
                    "external": {
                        "name": "file",
                        "filename": filename,
                        "put": {},
                        "size": len(data),
                    }
                },
                {"inlined": {"name": "title", "content": filename, "contentType": "text/plain"}},
                {"inlined": {"name": "album_mode", "content": "permanent", "contentType": "text/plain"}},
                {"inlined": {"name": "onepick_host_id", "content": "10", "contentType": "text/plain"}},
                {
                    "inlined": {
                        "name": "onepick_host_usecase",
                        "content": "RichEditor",
                        "contentType": "text/plain",
                    }
                },
            ]
        },
    }
    start_headers = {
        "Authorization": f"Bearer {access_token}",
        "X-Goog-Upload-Command": "start",
        "X-Goog-Upload-Header-Content-Length": str(len(data)),
        "X-Goog-Upload-Header-Content-Type": mime,
        "X-Goog-Upload-Protocol": "resumable",
        "Content-Type": "application/json",
    }
    async with httpx.AsyncClient(timeout=60.0) as http:
        started = await http.post(_UPLOAD_START, params={"authuser": "0"}, json=payload, headers=start_headers)
        if started.status_code not in (200, 201):
            raise RuntimeError(
                f"Blogger image upload session failed ({started.status_code}): {started.text[:400]}"
            )
        if "AUTH_REQUIRED" in (started.text or ""):
            raise RuntimeError(
                "Blogger image upload rejected the access token (AUTH_REQUIRED). "
                "Refresh OAuth credentials and retry."
            )
        upload_url = started.headers.get("x-goog-upload-url") or ""
        if not upload_url:
            raise RuntimeError(
                "Blogger image upload did not return a session URL: "
                f"{started.text[:400]}"
            )
        uploaded = await http.post(
            upload_url,
            content=data,
            headers={
                "Authorization": f"Bearer {access_token}",
                "Content-Type": mime,
                "X-Goog-Upload-Command": "upload, finalize",
                "X-Goog-Upload-Offset": "0",
            },
        )
        if uploaded.status_code not in (200, 201):
            raise RuntimeError(
                f"Blogger image upload failed ({uploaded.status_code}): {uploaded.text[:400]}"
            )
        try:
            body = uploaded.json()
        except Exception as exc:
            raise RuntimeError("Blogger image upload returned a non-JSON body") from exc
        return image_url_from_upload(body)


def blog_image_tools(workspace: Path, client_factory: Callable[[], Any] | None) -> list:
    """Tool Bete uses once, just before post_to_blogger.

    ``client_factory`` returns a Gemini client. It is None when covers are off:
    no Gemini key, or anonymous mode without Gemini.
    """
    from runtime.tools import Tool

    async def generate_blog_image(title: str, description: str) -> str:
        """Generate one 16:9 cover and upload it so the post can embed it."""
        if client_factory is None:
            return (
                "Capas desligadas nesta redação: não há provedor de imagem. "
                "Publique o artigo só com texto, sem linha <img>, e registre isso no relatório."
            )
        title = (title or "").strip()
        description = (description or "").strip()
        if not title:
            return "Título obrigatório para gerar a capa."
        try:
            data, mime = await generate_cover_bytes(client_factory(), title, description)
        except Exception as exc:
            logger.exception("Cover generation failed")
            return (
                f"Falha ao gerar a capa: {exc}. "
                "Publique o artigo mesmo assim, só com texto, e registre a falha no relatório."
            )

        path = cover_path(workspace, title)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Gemini sometimes returns jpeg bytes with a png name; keep the real type.
        suffix = ".jpg" if "jpeg" in mime or "jpg" in mime else ".png"
        if path.suffix != suffix:
            path = path.with_suffix(suffix)
        path.write_bytes(data)
        relative = path.relative_to(workspace.resolve())

        try:
            from runtime.blogger import default_blogger_client

            token = await default_blogger_client.get_valid_access_token()
            url = await upload_blogger_image(token, data, path.name, mime=mime or "image/png")
        except Exception as exc:
            logger.exception("Cover upload failed")
            return (
                f"Capa salva em {relative}, mas o envio ao Blogger falhou: {exc}. "
                "Publique o artigo mesmo assim, só com texto, e registre a falha no relatório."
            )

        return (
            "Capa pronta. Cole esta linha como PRIMEIRA linha do corpo em post_to_blogger "
            "(não gere outra imagem para este artigo):\n"
            f"{cover_markdown(url)}\n"
            f"Arquivo local: {relative}\n"
            f"URL: {url}"
        )

    return [
        Tool(
            generate_blog_image,
            name="generate_blog_image",
            description=(
                "Generate one cover illustration for a Blogger post and upload it. "
                "Call once per article, before post_to_blogger. "
                "Pass the post title and a short scene description with no words to render. "
                "Put the returned <img> line first in the post body. "
                "If this fails, publish the text anyway."
            ),
        )
    ]
