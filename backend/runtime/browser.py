"""One headless Chromium page per desk, mirrored onto that monitor."""

from __future__ import annotations

import inspect
import logging
import os
from pathlib import Path
from typing import Any, Callable
from urllib.parse import unquote, urlparse

from pydantic import BaseModel, Field

from runtime.tools import Tool

logger = logging.getLogger("office.browser")

_browser_states: dict[str, dict[str, Any]] = {}
_latest_frames: dict[str, dict[str, Any]] = {}
_frame_listeners: list[Callable[[dict[str, Any]], Any]] = []

_COOKIE_JS = """() => {
  const re = /(accept|agree|allow|consent|aceitar|concordo|permitir|got it|^ok$)/i;
  const nodes = Array.from(document.querySelectorAll(
    'button, [role="button"], input[type="button"], input[type="submit"], a'
  ));
  for (const el of nodes) {
    const text = (el.innerText || el.value || el.getAttribute('aria-label') || '').trim();
    if (!text || text.length > 48) continue;
    if (!re.test(text)) continue;
    const style = window.getComputedStyle(el);
    if (style.display === 'none' || style.visibility === 'hidden') continue;
    el.click();
    return text;
  }
  return '';
}"""


def _fix_playwright_browser_path() -> None:
    candidates = [
        Path.home() / ".cache" / "ms-playwright",
        Path(os.environ.get("PLAYWRIGHT_BROWSERS_PATH") or ""),
    ]
    for base in candidates:
        if not base or not base.is_dir():
            continue
        has_chrome = any(base.glob("chromium-*/chrome-linux*/chrome"))
        has_shell = any(
            base.glob("chromium_headless_shell-*/chrome-headless-shell-linux*/chrome-headless-shell")
        )
        if has_chrome or has_shell:
            os.environ["PLAYWRIGHT_BROWSERS_PATH"] = str(base)
            break
    os.environ["PLAYWRIGHT_CHROMIUM_USE_HEADLESS_SHELL"] = "0"


def _get_browser_state(context_name: str) -> dict[str, Any]:
    state = _browser_states.get(context_name)
    if state is None:
        state = {
            "playwright": None,
            "browser": None,
            "page": None,
            "headless": True,
            "mirror_monitor": True,
        }
        _browser_states[context_name] = state
    return state


def add_frame_listener(listener: Callable[[dict[str, Any]], Any]) -> None:
    if listener not in _frame_listeners:
        _frame_listeners.append(listener)


def remove_frame_listener(listener: Callable[[dict[str, Any]], Any]) -> None:
    if listener in _frame_listeners:
        _frame_listeners.remove(listener)


async def _emit_frame(context_name: str, jpeg: bytes, url: str) -> None:
    frame = {"context": context_name, "jpeg": jpeg, "url": url}
    _latest_frames[context_name] = frame
    for listener in list(_frame_listeners):
        try:
            result = listener(frame)
            if inspect.isawaitable(result):
                await result
        except Exception:
            logger.debug("Monitor frame listener failed", exc_info=True)


async def _publish_snapshot(page: Any, context_name: str) -> None:
    state = _get_browser_state(context_name)
    if not state.get("mirror_monitor") or page is None:
        return
    try:
        jpeg = await page.screenshot(type="jpeg", quality=55, animations="disabled")
        url = ""
        try:
            url = page.url or ""
        except Exception:
            url = ""
        if jpeg:
            await _emit_frame(context_name, jpeg, url)
    except Exception:
        logger.debug("Desk monitor snapshot failed for %s", context_name, exc_info=True)


def _proxy_settings(raw: str) -> dict[str, str]:
    """Playwright proxy settings from OFFICE_BROWSER_PROXY, so desk browsing
    leaves through it instead of this machine's address."""
    value = (raw or "").strip()
    if not value:
        return {}
    parsed = urlparse(value)
    if parsed.scheme not in ("http", "https", "socks5") or not parsed.hostname:
        raise ValueError("OFFICE_BROWSER_PROXY must look like http://host:port or socks5://host:port")
    server = f"{parsed.scheme}://{parsed.hostname}"
    if parsed.port:
        server += f":{parsed.port}"
    settings = {"server": server}
    # Playwright takes credentials as separate fields, not inside the URL.
    if parsed.username:
        settings["username"] = unquote(parsed.username)
    if parsed.password:
        settings["password"] = unquote(parsed.password)
    return settings


async def _page(context_name: str) -> Any:
    from playwright.async_api import async_playwright

    state = _get_browser_state(context_name)
    page = state.get("page")
    if page is not None:
        return page
    playwright = state.get("playwright")
    if playwright is None:
        playwright = await async_playwright().start()
        state["playwright"] = playwright
    browser = state.get("browser")
    if browser is None:
        launch: dict[str, Any] = {"headless": bool(state.get("headless", True))}
        proxy = _proxy_settings(os.getenv("OFFICE_BROWSER_PROXY", ""))
        if proxy:
            launch["proxy"] = proxy
        browser = await playwright.chromium.launch(**launch)
        state["browser"] = browser
    page = await browser.new_page(viewport={"width": 1280, "height": 800})
    state["page"] = page
    return page


async def _accept_cookies(page: Any) -> str:
    try:
        clicked = await page.evaluate(_COOKIE_JS)
        return f"accepted '{clicked}'" if clicked else "no banner"
    except Exception:
        return "no banner"


def _check_http_url(url: str) -> str:
    parsed = urlparse((url or "").strip())
    if parsed.scheme not in ("http", "https"):
        raise ValueError("URL scheme must be http or https")
    return url.strip()


class NavigateInput(BaseModel):
    url: str = Field(description="http(s) URL to open on this desk's monitor")


class ClickInput(BaseModel):
    selector: str = Field(description="CSS selector of the element to click")


class InputTextInput(BaseModel):
    selector: str = Field(description="CSS selector of the field")
    text: str = Field(description="Text to type")


class ExtractInput(BaseModel):
    selector: str = Field(default="", description="Optional CSS selector. Empty reads the page body.")


class ScrollInput(BaseModel):
    direction: str = Field(default="down", description="down or up")
    pixels: int = Field(default=900, description="How far to scroll")


class PressKeyInput(BaseModel):
    key: str = Field(description="Key name, for example Enter, Escape, or PageDown")
    selector: str = Field(default="", description="Optional CSS selector to focus first")


def browser_tool_list(agent_id: str) -> list[Tool]:
    """Tools bound to one agent's Chromium page."""

    async def navigate_browser(url: str) -> str:
        target = _check_http_url(url)
        page = await _page(agent_id)
        response = await page.goto(target, wait_until="domcontentloaded", timeout=60000)
        status = response.status if response else "unknown"
        cookies = await _accept_cookies(page)
        await _publish_snapshot(page, agent_id)
        return f"Navigating to {target} returned status code {status}. Cookie handling: {cookies}."

    async def click_element(selector: str) -> str:
        page = await _page(agent_id)
        await page.click(selector, timeout=8000)
        await _publish_snapshot(page, agent_id)
        return f"Clicked {selector}."

    async def input_text(selector: str, text: str) -> str:
        page = await _page(agent_id)
        await page.fill(selector, text, timeout=8000)
        await _publish_snapshot(page, agent_id)
        return f"Typed into {selector}."

    async def extract_text(selector: str = "") -> str:
        page = await _page(agent_id)
        if selector:
            locator = page.locator(selector)
            if await locator.count() == 0:
                return f"No element found for selector: {selector}"
            text = await locator.first.inner_text()
        else:
            text = await page.evaluate("() => document.body ? document.body.innerText : ''")
        text = (text or "").strip()
        if len(text) > 8000:
            text = text[:8000] + "\n…[truncated]"
        await _publish_snapshot(page, agent_id)
        return text or "(empty page)"

    async def scroll_page(direction: str = "down", pixels: int = 900) -> str:
        page = await _page(agent_id)
        delta = abs(int(pixels or 900))
        if str(direction).lower().startswith("up"):
            delta = -delta
        await page.mouse.wheel(0, delta)
        await _publish_snapshot(page, agent_id)
        return f"Scrolled {direction} by {abs(delta)} px."

    async def press_key(key: str, selector: str = "") -> str:
        page = await _page(agent_id)
        if selector:
            await page.focus(selector, timeout=4000)
        await page.keyboard.press(key)
        await _publish_snapshot(page, agent_id)
        return f"Pressed {key}."

    async def current_webpage() -> str:
        page = await _page(agent_id)
        url = ""
        title = ""
        try:
            url = page.url or ""
            title = await page.title()
        except Exception as exc:
            return f"No page yet: {exc}"
        await _publish_snapshot(page, agent_id)
        return f"title={title}\nurl={url}"

    specs = [
        (navigate_browser, "navigate_browser", "Open a URL on this desk monitor. Cookie banners are accepted automatically.", NavigateInput),
        (click_element, "click_element", "Click an element on the current page by CSS selector.", ClickInput),
        (input_text, "input_text", "Type text into a field on the current page.", InputTextInput),
        (extract_text, "extract_text", "Read visible text from the current page, or from one CSS selector.", ExtractInput),
        (scroll_page, "scroll_page", "Scroll the current page up or down so more of the article is visible.", ScrollInput),
        (press_key, "press_key", "Press a keyboard key, optionally after focusing a selector.", PressKeyInput),
        (current_webpage, "current_webpage", "Report the title and URL of the page on this desk monitor.", None),
    ]
    tools = []
    for func, name, description, schema in specs:
        tool = Tool(func, name=name, description=description, args_schema=schema)
        tool.context_name = agent_id
        tools.append(tool)
    return tools
