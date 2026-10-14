"""Run one desk assignment as a CrewAI 1.15 agent.

The newsroom is already a crew: five desks, one task at a time, handoffs on
the EpochDB board. CrewAI owns the tool-calling loop for whichever desk is
awake. Delegation stays off so a desk cannot skip ``hand_off``.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import os
from typing import Any, Optional, get_type_hints

from runtime.tools import Tool

logger = logging.getLogger("office.crew")

# Playwright (and every other async tool) must stay on the office loop.
# CrewAI otherwise runs the sync tool from a worker thread, and asyncio.run()
# opens a loop that dies when the call returns. The next call reuses the page
# and blocks forever.
_office_loop: asyncio.AbstractEventLoop | None = None
_TOOL_TIMEOUT_SEC = float(os.getenv("OFFICE_TOOL_TIMEOUT_SEC", "120"))


def bind_office_loop(loop: asyncio.AbstractEventLoop) -> None:
    global _office_loop
    _office_loop = loop


def _args_schema_from_callable(func: Any) -> Any:
    """Build a Pydantic args model from the real function, not from ``_run``.

    A class body must not assign ``description = description`` (or ``name = name``).
    CrewAI's tool model evaluates those names in the class namespace and raises
    ``NameError`` before the enclosing variable is visible. ``create_model``
    avoids that, including when a parameter itself is named ``description``.
    """
    from pydantic import Field, create_model

    signature = inspect.signature(func)
    try:
        hints = get_type_hints(func)
    except Exception:
        hints = {}
    fields: dict[str, Any] = {}
    for param_name, param in signature.parameters.items():
        if param_name in ("self", "cls"):
            continue
        if param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue
        annotation = hints.get(param_name, param.annotation)
        if annotation is inspect.Parameter.empty or isinstance(annotation, str):
            annotation = Any
        if param.default is inspect.Parameter.empty:
            fields[param_name] = (annotation, Field())
        else:
            fields[param_name] = (annotation, Field(default=param.default))
    if not fields:
        return None
    model_name = "".join(ch if ch.isalnum() else "_" for ch in (func.__name__ or "tool"))
    return create_model(f"{model_name}_Args", **fields)


def _as_crew_tool(tool: Tool) -> Any:
    from crewai.tools import BaseTool

    # These names must not match the class fields. A class body that assigns
    # `description = description` raises NameError under CrewAI's model base.
    tool_name = tool.name
    tool_description = (tool.description or tool_name).strip()
    tool_schema = tool.args_schema or _args_schema_from_callable(tool.func)
    tool_func = tool.func

    class CrewTool(BaseTool):
        name: str = tool_name
        description: str = tool_description
        args_schema: Optional[type] = tool_schema

        def _accepted(self, kwargs: dict) -> dict:
            signature = inspect.signature(tool_func)
            if any(
                param.kind == inspect.Parameter.VAR_KEYWORD
                for param in signature.parameters.values()
            ):
                return kwargs
            return {
                key: value
                for key, value in kwargs.items()
                if key in signature.parameters
            }

        async def _call(self, kwargs: dict) -> str:
            accepted = self._accepted(kwargs)
            try:
                result = tool_func(**accepted)
                if inspect.iscoroutine(result):
                    result = await asyncio.wait_for(result, timeout=_TOOL_TIMEOUT_SEC)
            except asyncio.TimeoutError:
                return (
                    f"Tool {tool_name} timed out after {_TOOL_TIMEOUT_SEC:.0f}s. "
                    "Continue with what you already have, or try a different URL."
                )
            return result if isinstance(result, str) else str(result)

        def _run(self, **kwargs: Any) -> str:
            # CrewAI's sync path runs in a worker thread. Hop back to the office
            # loop so Playwright's page stays on the loop that created it.
            try:
                asyncio.get_running_loop()
                on_loop = True
            except RuntimeError:
                on_loop = False
            if on_loop:
                raise RuntimeError(
                    f"Tool {tool_name} was called synchronously on the office event loop"
                )
            coro = self._call(kwargs)
            loop = _office_loop
            if loop is not None and loop.is_running():
                future = asyncio.run_coroutine_threadsafe(coro, loop)
                return future.result(timeout=_TOOL_TIMEOUT_SEC + 15)
            return asyncio.run(coro)

        async def _arun(self, **kwargs: Any) -> str:
            return await self._call(kwargs)

        def to_structured_tool(self) -> Any:
            # ainvoke() only stays on this loop when func is a coroutine.
            # The default wires func to sync _run, which deadlocks the browser.
            structured = super().to_structured_tool()
            structured.func = self._arun
            return structured

    CrewTool.__name__ = f"CrewTool_{tool.name}"
    return CrewTool()


async def run_desk_task(
    *,
    agent_name: str,
    role: str,
    backstory: str,
    prompt: str,
    tools: Optional[list] = None,
    model: str = "",
    max_iterations: int = 40,
    memory: Any = None,
) -> str:
    """Kick off one CrewAI agent. EpochDB recall is prepended when available.

    ``model`` is ``<provider>/<model>``; empty means the office default.
    """
    from runtime.providers import apply_privacy_defaults, crew_llm, resolve

    # Anonymous mode has to switch CrewAI telemetry off before crewai is imported.
    apply_privacy_defaults()
    from crewai import Agent, Crew, Process, Task

    bind_office_loop(asyncio.get_running_loop())
    resolved = resolve(model)
    # Raises before any work when the key is missing or the provider is blocked.
    llm = crew_llm(resolved, temperature=0.4)

    recall = ""
    if memory is not None and hasattr(memory, "recall"):
        try:
            recall = await memory.recall(prompt, k=4)
        except Exception:
            logger.warning("Skipping EpochDB recall", exc_info=True)
    if recall:
        prompt = (
            "Relevant memory from EpochDB (prior deliveries and board notes):\n"
            f"{recall}\n\n"
            f"{prompt}"
        )

    crew_tools = [_as_crew_tool(item) for item in (tools or []) if isinstance(item, Tool)]
    agent = Agent(
        role=(role or "Newsroom desk").strip(),
        goal=(
            f"Finish the assignment for {agent_name}. "
            "Use tools. Report file paths, URLs, and task ids. Do not introduce yourself."
        ),
        backstory=backstory or f"You are {agent_name}.",
        llm=llm,
        tools=crew_tools,
        allow_delegation=False,
        verbose=os.getenv("CREWAI_VERBOSE", "true").lower() not in {"0", "false", "no"},
        max_iter=max(1, int(max_iterations)),
        memory=False,
    )
    task = Task(
        description=prompt,
        expected_output=(
            "A short delivery in the office language: what was done, saved paths, "
            "published URLs, and any child task ids. No self-introduction."
        ),
        agent=agent,
    )
    crew = Crew(
        agents=[agent],
        tasks=[task],
        process=Process.sequential,
        verbose=agent.verbose,
        memory=False,
    )
    logger.info("CrewAI %s kickoff model=%s tools=%s", agent_name, resolved.label, [t.name for t in crew_tools])
    if hasattr(crew, "akickoff"):
        result = await crew.akickoff()
    else:
        result = await crew.kickoff_async()
    text = str(getattr(result, "raw", None) or result or "").strip()
    if not text:
        raise RuntimeError(f"CrewAI returned an empty delivery for {agent_name}")
    if memory is not None and hasattr(memory, "remember_outcome"):
        await memory.remember_outcome(agent_name, prompt, text)
    return text
