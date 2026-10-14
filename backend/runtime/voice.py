"""Gemini Live voice for one desk. The browser sends 16 kHz PCM and plays 24 kHz PCM back.

Interactive mode only. The session can pin a note, accept work on this desk,
hand work to a teammate, read the board, and drive this desk's browser.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
from typing import Any, Dict

from fastapi import WebSocket, WebSocketDisconnect

from office import (
    Office,
    browser_tools,
    format_team_roster,
    language_rule,
    office_language,
    office_language_code,
    roster_colleague_hint,
)
from runtime.providers import gemini_extras_available

logger = logging.getLogger("office.voice")

INPUT_SAMPLE_RATE = 16000
LIVE_MODEL = os.getenv("GEMINI_VOICE_MODEL", "gemini-3.8-live")


class LiveDeskSession:
    def __init__(self, office: Office, spec: Dict[str, Any]):
        self.office = office
        self.spec = spec
        self.client = None
        if gemini_extras_available():
            # Imported here so an office that does not use Gemini never loads its SDK.
            from google import genai

            self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY", "").strip())
        self.session = None
        self._ctx = None
        self._connected = False
        self._browser = {tool.name: tool for tool in browser_tools(spec["id"])}

    def _tools(self) -> list:
        from google.genai import types

        others = roster_colleague_hint(self.office.agents, exclude_id=self.spec.get("id") or "")
        return [
            types.Tool(
                function_declarations=[
                    types.FunctionDeclaration(
                        name="accept_task",
                        description="Put the user's request on YOUR desk when they ask you to do the work.",
                        parameters=types.Schema(
                            type=types.Type.OBJECT,
                            properties={
                                "task": types.Schema(type=types.Type.STRING, description="What you should do"),
                            },
                            required=["task"],
                        ),
                    ),
                    types.FunctionDeclaration(
                        name="assign_task",
                        description=(
                            "Assign a task to ONE colleague by exact name. "
                            f"Teammates: {others}."
                        ),
                        parameters=types.Schema(
                            type=types.Type.OBJECT,
                            properties={
                                "assignee_name": types.Schema(
                                    type=types.Type.STRING,
                                    description=f"Exact teammate name. Options: {others}.",
                                ),
                                "task": types.Schema(type=types.Type.STRING, description="What they should do"),
                                "wait_for": types.Schema(
                                    type=types.Type.STRING,
                                    description="Optional colleague name whose current task must finish first",
                                ),
                            },
                            required=["assignee_name", "task"],
                        ),
                    ),
                    types.FunctionDeclaration(
                        name="post_note",
                        description="Pin a short note on the shared board. This does not assign work.",
                        parameters=types.Schema(
                            type=types.Type.OBJECT,
                            properties={
                                "title": types.Schema(type=types.Type.STRING),
                                "content": types.Schema(type=types.Type.STRING),
                            },
                            required=["title", "content"],
                        ),
                    ),
                    types.FunctionDeclaration(
                        name="read_blackboard",
                        description="Read the latest notes, tasks, and results on the shared board.",
                        parameters=types.Schema(type=types.Type.OBJECT, properties={}),
                    ),
                    types.FunctionDeclaration(
                        name="navigate_browser",
                        description="Open a URL on this desk's monitor.",
                        parameters=types.Schema(
                            type=types.Type.OBJECT,
                            properties={"url": types.Schema(type=types.Type.STRING)},
                            required=["url"],
                        ),
                    ),
                    types.FunctionDeclaration(
                        name="click_element",
                        description="Click an element on the page currently on this desk monitor.",
                        parameters=types.Schema(
                            type=types.Type.OBJECT,
                            properties={
                                "css_selector": types.Schema(type=types.Type.STRING),
                                "text": types.Schema(type=types.Type.STRING),
                            },
                        ),
                    ),
                    types.FunctionDeclaration(
                        name="input_text",
                        description="Type into an element on the desk monitor page.",
                        parameters=types.Schema(
                            type=types.Type.OBJECT,
                            properties={
                                "selector": types.Schema(type=types.Type.STRING),
                                "text": types.Schema(type=types.Type.STRING),
                            },
                            required=["selector", "text"],
                        ),
                    ),
                    types.FunctionDeclaration(
                        name="scroll_page",
                        description="Scroll the page on the desk monitor.",
                        parameters=types.Schema(
                            type=types.Type.OBJECT,
                            properties={
                                "direction": types.Schema(type=types.Type.STRING, description="up or down"),
                            },
                        ),
                    ),
                    types.FunctionDeclaration(
                        name="extract_text",
                        description="Read visible text from the page on the desk monitor.",
                        parameters=types.Schema(
                            type=types.Type.OBJECT,
                            properties={"selector": types.Schema(type=types.Type.STRING)},
                        ),
                    ),
                ]
            )
        ]

    def _system_instruction(self) -> str:
        my_id = self.spec.get("id") or ""
        roster = format_team_roster(self.office.agents, exclude_id=my_id, max_prompt_chars=400)
        lang = office_language()
        return (
            f"You are {self.spec.get('name')}, role: {self.spec.get('role')}.\n"
            f"Your description: {(self.spec.get('systemPrompt') or '')[:500]}\n\n"
            f"YOUR TEAMMATES:\n{roster}\n\n"
            f"{language_rule()}\n"
            f"CRITICAL: Every spoken reply must be in {lang} only. "
            "Mirror the user's language; if they speak Portuguese, answer in Portuguese. "
            "Never switch to English unless the office language is English.\n"
            "Wait until the user clearly finishes their sentence before answering. "
            "Do not reply to half-finished phrases. "
            "Answer in one or two short spoken sentences. No markdown. "
            "Call read_blackboard only when you need the wall. "
            "When the user asks YOU to do the work, call accept_task. "
            "When they ask a colleague to do it, call assign_task with that colleague's exact name. "
            "Call post_note only to pin a note, not to assign work. "
            "Use the browser tools only when the user asks to open or browse a site."
        )

    def _config(self) -> Any:
        from google.genai import types

        voice = self.spec.get("voiceName") or "Puck"
        return types.LiveConnectConfig(
            response_modalities=[types.Modality.AUDIO],
            speech_config=types.SpeechConfig(
                language_code=office_language_code(),
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice)
                ),
            ),
            system_instruction=types.Content(parts=[types.Part(text=self._system_instruction())]),
            tools=self._tools(),
            output_audio_transcription=types.AudioTranscriptionConfig(),
            input_audio_transcription=types.AudioTranscriptionConfig(),
            realtime_input_config=types.RealtimeInputConfig(
                turn_coverage=types.TurnCoverage.TURN_INCLUDES_ONLY_ACTIVITY,
            ),
        )

    async def connect(self) -> None:
        if not self.client:
            raise RuntimeError(
                "Voice needs Gemini: set GEMINI_API_KEY, and list gemini in OFFICE_ALLOWED_PROVIDERS if that is set."
            )
        self._ctx = self.client.aio.live.connect(model=LIVE_MODEL, config=self._config())
        self.session = await self._ctx.__aenter__()
        self._connected = True
        logger.info("Live session for %s on %s", self.spec.get("name"), LIVE_MODEL)

    async def disconnect(self) -> None:
        if self._connected and self._ctx:
            try:
                await self._ctx.__aexit__(None, None, None)
            except Exception as exc:
                logger.warning("Live disconnect: %s", exc)
        self._connected = False
        self.session = None

    @property
    def is_connected(self) -> bool:
        return self._connected and self.session is not None

    async def send_audio(self, pcm_bytes: bytes) -> None:
        if not self.is_connected:
            return
        from google.genai import types

        await self.session.send_realtime_input(
            audio=types.Blob(data=pcm_bytes, mime_type=f"audio/pcm;rate={INPUT_SAMPLE_RATE}")
        )

    async def send_text(self, text: str) -> None:
        if not self.is_connected or not text:
            return
        await self.session.send_realtime_input(text=text)

    async def handle_tool(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        if name == "post_note":
            node_id = await self.office.post_note(
                str(args.get("title") or "Nota"),
                str(args.get("content") or ""),
                "You",
            )
            return {"status": "pinned", "node_id": node_id}
        if name == "accept_task":
            posted = await self.office.dispatch_task(
                self.spec["id"],
                str(args.get("task") or ""),
                author="You",
                allow_duplicate=True,
            )
            return {
                "status": posted.get("status") or "assigned",
                "assignee": self.spec.get("name"),
                "task_id": posted.get("id"),
            }
        if name == "assign_task":
            wanted = str(args.get("assignee_name") or "")
            target = self.office.find_agent(wanted)
            if target is None:
                roster = format_team_roster(self.office.agents, exclude_id=self.spec.get("id") or "")
                return {"error": f"No unique teammate matched '{wanted}'. Use an exact name from:\n{roster}"}
            depends_on = ""
            wait = str(args.get("wait_for") or "").strip()
            if wait:
                wait_l = wait.lower()
                dep_agent = None
                if wait_l not in ("me", "eu", "mim", "myself", "self"):
                    dep_agent = self.office.find_agent(wait)
                if dep_agent is None and wait_l not in ("me", "eu", "mim", "myself", "self"):
                    return {"error": f"wait_for '{wait}' not found among colleagues"}
                if dep_agent is not None:
                    depends_on = self.office.latest_task_key_for(dep_agent["id"]) or ""
                    if not depends_on:
                        return {
                            "error": (
                                f"No prior task for {dep_agent['name']} to wait on. "
                                "Assign their work first."
                            )
                        }
            posted = await self.office.dispatch_task(
                target["id"],
                str(args.get("task") or ""),
                author=self.spec.get("name") or "Agent",
                depends_on=depends_on,
                allow_duplicate=True,
            )
            return {
                "status": posted.get("status") or "assigned",
                "assignee": target["name"],
                "task_id": posted.get("id"),
                "job_id": posted.get("job_id"),
                "depends_on": depends_on or None,
            }
        if name == "read_blackboard":
            return {"blackboard": self.office.board_summary(limit=20)}
        tool = self._browser.get(name)
        if tool is not None:
            await self.office.set_status(self.spec["id"], "thinking", "using_computer")
            call_args = dict(args or {})
            if name == "click_element":
                call_args = {
                    "selector": str(
                        call_args.get("css_selector") or call_args.get("selector") or call_args.get("text") or ""
                    )
                }
            elif name == "scroll_page":
                call_args = {"direction": str(call_args.get("direction") or "down")}
            elif name == "extract_text":
                call_args = {"selector": str(call_args.get("selector") or "")}
            elif name == "navigate_browser":
                call_args = {"url": str(call_args.get("url") or "")}
            elif name == "input_text":
                call_args = {
                    "selector": str(call_args.get("selector") or ""),
                    "text": str(call_args.get("text") or ""),
                }
            result = await tool.execute(**call_args)
            text = result if isinstance(result, str) else str(result)
            return {"result": text[:1500]}
        return {"error": f"Unknown tool {name}"}

    def _parse(self, response) -> list:
        events = []
        server_content = getattr(response, "server_content", None)
        if not server_content:
            return events
        if getattr(server_content, "interrupted", False):
            events.append({"type": "interrupted"})
        model_turn = getattr(server_content, "model_turn", None)
        if model_turn:
            for part in model_turn.parts:
                inline = getattr(part, "inline_data", None)
                if inline and inline.data:
                    audio = inline.data
                    if isinstance(audio, str):
                        audio = base64.b64decode(audio)
                    events.append({"type": "audio", "data": audio})
        output = getattr(server_content, "output_transcription", None)
        if output and getattr(output, "text", None):
            events.append({"type": "output_transcription", "text": output.text})
        incoming = getattr(server_content, "input_transcription", None)
        if incoming and getattr(incoming, "text", None):
            events.append({
                "type": "input_transcription",
                "text": incoming.text,
                "finished": bool(getattr(incoming, "finished", False)),
            })
        if getattr(server_content, "turn_complete", False):
            events.append({"type": "turn_complete"})
        return events

    async def receive_responses(self):
        if not self.session:
            return
        from google.genai import types

        try:
            while self.is_connected:
                async for response in self.session.receive():
                    tool_call = getattr(response, "tool_call", None)
                    if tool_call and getattr(tool_call, "function_calls", None):
                        for call in tool_call.function_calls:
                            args = call.args or {}
                            yield {"type": "tool_call", "name": call.name, "args": args}
                            result = await self.handle_tool(call.name, args)
                            try:
                                await self.session.send_tool_response(
                                    function_responses=types.FunctionResponse(
                                        name=call.name,
                                        id=call.id,
                                        response=result,
                                    )
                                )
                            except Exception as exc:
                                logger.error("Tool response failed: %s", exc)
                            yield {"type": "tool_executed", "name": call.name, "args": args, "result": result}
                    for event in self._parse(response):
                        yield event
        except asyncio.CancelledError:
            return
        except Exception as exc:
            logger.error("Live receive error: %s", exc)
            yield {"type": "error", "message": str(exc)}


class OfficeVoiceSocket:
    def __init__(self, websocket: WebSocket, office: Office, agent_id: str):
        self.websocket = websocket
        self.office = office
        self.agent_id = agent_id

    async def run(self) -> None:
        await self.websocket.accept()
        spec = self.office.agents.get(self.agent_id)
        if not spec:
            await self.websocket.send_json({"event": "error", "message": "Agente não encontrado"})
            await self.websocket.close()
            return
        live = LiveDeskSession(self.office, spec)
        try:
            await live.connect()
        except Exception as exc:
            logger.exception("Live connect failed")
            await self.websocket.send_json({"event": "error", "message": str(exc)})
            await self.websocket.close()
            return

        await self.office.set_status(self.agent_id, "idle", "using_computer")
        await self.websocket.send_json({
            "event": "session.started",
            "agent_id": self.agent_id,
            "model": LIVE_MODEL,
            "message": f"Listening to {spec['name']}",
        })
        speaking = False

        async def output_loop():
            nonlocal speaking
            try:
                async for event in live.receive_responses():
                    kind = event["type"]
                    if kind == "audio":
                        if not speaking:
                            speaking = True
                            await self.office.set_status(self.agent_id, "speaking", "looking_window")
                        await self.websocket.send_bytes(event["data"])
                    elif kind == "interrupted":
                        speaking = False
                        await self.websocket.send_json({"event": "user.interrupted"})
                    elif kind == "output_transcription":
                        await self.websocket.send_json({
                            "event": "transcript",
                            "role": "agent",
                            "text": event["text"],
                            "finished": False,
                        })
                    elif kind == "input_transcription":
                        await self.websocket.send_json({
                            "event": "transcript",
                            "role": "user",
                            "text": event["text"],
                            "finished": bool(event.get("finished", False)),
                        })
                    elif kind == "turn_complete":
                        speaking = False
                        await self.office.set_status(self.agent_id, "idle", "using_computer")
                        await self.websocket.send_json({"event": "turn_complete", "role": "agent"})
                    elif kind in ("tool_call", "tool_executed"):
                        if kind == "tool_call":
                            speaking = False
                            await self.office.set_status(self.agent_id, "thinking", "using_computer")
                        await self.websocket.send_json({
                            "event": kind,
                            "name": event.get("name"),
                            "args": event.get("args"),
                            "result": event.get("result"),
                        })
                    elif kind == "error":
                        await self.websocket.send_json({"event": "error", "message": event.get("message")})
            except Exception as exc:
                logger.info("Voice output ended: %s", exc)

        output_task = asyncio.create_task(output_loop())
        try:
            while True:
                message = await self.websocket.receive()
                if message.get("bytes"):
                    await live.send_audio(message["bytes"])
                elif message.get("text"):
                    payload = json.loads(message["text"])
                    if payload.get("event") == "call.ended":
                        break
                    if payload.get("event") == "speech.final" and payload.get("transcript"):
                        await live.send_text(str(payload["transcript"]))
        except WebSocketDisconnect:
            logger.info("Voice client disconnected")
        finally:
            output_task.cancel()
            await live.disconnect()
            if self.agent_id in self.office.agents:
                await self.office.set_status(self.agent_id, "idle", "using_computer")
