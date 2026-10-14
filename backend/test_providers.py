"""Provider routing, anonymous mode, and the OpenAI-compatible paths.

Nothing here reaches the internet. CrewAI and the embedder talk to a fake
OpenAI-compatible server on 127.0.0.1 that records every request.
"""

import asyncio
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import textwrap
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import mock

import numpy as np

from runtime import providers
from runtime.embeddings import OpenAICompatibleEmbedder
from runtime.tools import Tool

BACKEND = Path(__file__).resolve().parent
FIREWORKS_MODEL = "accounts/fireworks/models/deepseek-v4-flash-0731"

_PROVIDER_VARS = (
    "OFFICE_TEXT_MODEL",
    "GEMINI_TEXT_MODEL",
    "GEMINI_API_KEY",
    "DEEPSEEK_API_KEY",
    "DEEPSEEK_BASE_URL",
    "FIREWORKS_API_KEY",
    "FIREWORKS_BASE_URL",
    "OFFICE_OPENAI_API_KEY",
    "OFFICE_OPENAI_BASE_URL",
    "OFFICE_ALLOWED_PROVIDERS",
    "OFFICE_EMBED_MODEL",
    "EPOCHDB_MODEL",
    "EPOCHDB_EMBEDDING_MODEL",
    "EPOCHDB_DIM",
)


def _has(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


def env(**values: str):
    """os.environ with every provider variable cleared, then ``values`` set."""
    clean = {key: value for key, value in os.environ.items() if key not in _PROVIDER_VARS}
    clean.update(values)
    return mock.patch.dict(os.environ, clean, clear=True)


def echo_tool(calls: list) -> Tool:
    async def echo_tool(text: str) -> str:
        calls.append(text)
        return f"eco: {text}"

    return Tool(echo_tool, name="echo_tool", description="Repeat the text back. Call it once.")


class FakeOpenAI:
    """OpenAI-compatible server on 127.0.0.1 that records each request.

    Chat: with tools it asks for one tool call, then answers "entregue". In
    CrewAI's text tool format it does the same with Action / Final Answer.
    Embeddings: deterministic vectors. Models named ``fixed-*`` reject
    ``dimensions`` the way Fireworks' fixed-size models do.
    """

    def __init__(self) -> None:
        self.requests: list[dict] = []
        self._turns: dict[str, int] = {}
        self._lock = threading.Lock()
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), self._handler())
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()

    def take(self) -> list[dict]:
        with self._lock:
            seen, self.requests = self.requests, []
        return seen

    def _handler(self):
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802
                length = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(length) or b"{}")
                with fake._lock:
                    fake.requests.append(
                        {"path": self.path, "body": body, "auth": self.headers.get("Authorization", "")}
                    )
                if self.path.endswith("/embeddings"):
                    status, payload = fake._embeddings(body)
                elif self.path.endswith("/chat/completions"):
                    status, payload = 200, fake._chat(body)
                else:
                    status, payload = 404, {"error": {"message": f"no route {self.path}"}}
                raw = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def log_message(self, *args) -> None:
                return

        return Handler

    def _chat(self, body: dict) -> dict:
        messages = body.get("messages") or []
        key = json.dumps(messages[:2], sort_keys=True, default=str)
        with self._lock:
            turn = self._turns.get(key, 0)
            self._turns[key] = turn + 1
        tools = body.get("tools") or []
        text_tools = not tools and "Action Input" in json.dumps(messages, default=str)
        message: dict = {"role": "assistant", "content": "entregue"}
        finish = "stop"
        if tools and not any(m.get("role") == "tool" for m in messages):
            message = {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": tools[0]["function"]["name"], "arguments": json.dumps({"text": "ping"})},
                }],
            }
            finish = "tool_calls"
        elif text_tools and turn == 0:
            message["content"] = 'Thought: vou usar a ferramenta\nAction: echo_tool\nAction Input: {"text": "ping"}'
        elif text_tools:
            message["content"] = "Thought: pronto\nFinal Answer: entregue"
        return {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 0,
            "model": body.get("model") or "",
            "choices": [{"index": 0, "message": message, "finish_reason": finish}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }

    @staticmethod
    def _embeddings(body: dict) -> tuple[int, dict]:
        model = str(body.get("model") or "")
        if model.startswith("fixed-") and "dimensions" in body:
            return 400, {"error": {"message": f"model: {model} is not resizable"}}
        size = 4 if model.startswith("fixed-") else int(body.get("dimensions") or 8)
        inputs = body.get("input") or []
        if isinstance(inputs, str):
            inputs = [inputs]
        data = [
            {"object": "embedding", "index": i, "embedding": [float(len(text) + i + j + 1) for j in range(size)]}
            for i, text in enumerate(inputs)
        ]
        return 200, {"object": "list", "model": model, "data": data, "usage": {"prompt_tokens": 1, "total_tokens": 1}}


class ResolveTests(unittest.TestCase):
    def test_bare_name_means_gemini(self):
        with env():
            resolved = providers.resolve("gemini-3.5-flash-lite")
        self.assertEqual(
            (resolved.provider, resolved.model, resolved.key_env),
            ("gemini", "gemini-3.5-flash-lite", "GEMINI_API_KEY"),
        )

    def test_gemini_resource_names_stay_gemini(self):
        with env():
            for raw in ("models/gemini-2.5-flash", "tunedModels/my-tune"):
                resolved = providers.resolve(raw)
                self.assertEqual((resolved.provider, resolved.model), ("gemini", raw), raw)

    def test_each_prefix_routes_to_its_provider(self):
        cases = {
            "google/gemini-x": ("gemini", "gemini-x"),
            "deepseek/deepseek-flash": ("deepseek", "deepseek-flash"),
            f"fireworks/{FIREWORKS_MODEL}": ("fireworks", FIREWORKS_MODEL),
            "OpenAI/local-model": ("openai", "local-model"),
        }
        with env(OFFICE_OPENAI_BASE_URL="http://127.0.0.1:8000/v1/"):
            for raw, expected in cases.items():
                resolved = providers.resolve(raw)
                self.assertEqual((resolved.provider, resolved.model), expected, raw)
            self.assertEqual(providers.resolve("fireworks/m").base_url, providers.FIREWORKS_BASE_URL)
            self.assertEqual(providers.resolve("openai/m").base_url, "http://127.0.0.1:8000/v1")

    def test_unknown_provider_and_missing_model_are_rejected(self):
        with env():
            with self.assertRaisesRegex(ValueError, "Unknown provider 'firework'"):
                providers.resolve("firework/x")
            with self.assertRaisesRegex(ValueError, "no model"):
                providers.resolve("fireworks/")

    def test_default_model_order(self):
        with env():
            self.assertEqual(providers.default_text_model(), "gemini-3.5-flash-lite")
        with env(GEMINI_TEXT_MODEL="gemini-older"):
            self.assertEqual(providers.default_text_model(), "gemini-older")
        with env(GEMINI_TEXT_MODEL="gemini-older", OFFICE_TEXT_MODEL="fireworks/m"):
            self.assertEqual(providers.default_text_model(), "fireworks/m")

    def test_desk_model_overrides_the_default_and_skips_the_voice_model(self):
        from office import desk_model

        with env(OFFICE_TEXT_MODEL="fireworks/m"):
            self.assertEqual(desk_model({}), "fireworks/m")
            self.assertEqual(desk_model({"model": "deepseek/deepseek-flash"}), "deepseek/deepseek-flash")
            self.assertEqual(desk_model({"model": "gemini-3.8-live"}), "fireworks/m")

    def test_require_names_the_variable_and_never_the_value(self):
        with env():
            with self.assertRaisesRegex(RuntimeError, "FIREWORKS_API_KEY is required"):
                providers.require(providers.resolve("fireworks/m"))
        with env(OFFICE_OPENAI_API_KEY="secret-value"):
            with self.assertRaises(RuntimeError) as caught:
                providers.require(providers.resolve("openai/m"))
            self.assertIn("OFFICE_OPENAI_BASE_URL", str(caught.exception))
            self.assertNotIn("secret-value", str(caught.exception))
        with env(GEMINI_API_KEY="MY_GEMINI_API_KEY"):
            with self.assertRaisesRegex(RuntimeError, "GEMINI_API_KEY is required"):
                providers.require(providers.resolve("gemini-x"))

    def test_allow_list_blocks_other_providers(self):
        with env(OFFICE_ALLOWED_PROVIDERS="Fireworks, google", GEMINI_API_KEY="g", DEEPSEEK_API_KEY="d"):
            self.assertEqual(providers.allowed_providers(), {"fireworks", "gemini"})
            self.assertEqual(providers.require(providers.resolve("gemini-x")), "g")
            with self.assertRaisesRegex(RuntimeError, "blocked"):
                providers.require(providers.resolve("deepseek/deepseek-flash"))


@unittest.skipUnless(_has("crewai"), "crewai is not installed")
class CrewWiringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fake = FakeOpenAI()

    @classmethod
    def tearDownClass(cls):
        cls.fake.close()

    def setUp(self):
        self.fake.take()

    def _run_desk(self, model: str, **values: str) -> tuple[str, list]:
        from runtime.crew_loop import run_desk_task

        calls: list = []
        with env(CREWAI_VERBOSE="false", **values):
            text = asyncio.run(run_desk_task(
                agent_name="Rafa",
                role="Diretor",
                backstory="Teste de provedor.",
                prompt=f"Chame echo_tool uma vez e entregue. ({model})",
                tools=[echo_tool(calls)],
                model=model,
                max_iterations=4,
            ))
        return text, calls

    def test_crew_llm_pins_chat_completions_for_openai_compatible_servers(self):
        with env(FIREWORKS_API_KEY="fw-test"):
            llm = providers.crew_llm(providers.resolve(f"fireworks/{FIREWORKS_MODEL}"))
        self.assertTrue(llm.custom_openai)
        self.assertEqual(llm.api, "completions")

    def test_deepseek_desk_turns_thinking_off_on_every_request(self):
        text, calls = self._run_desk(
            "deepseek/deepseek-flash",
            DEEPSEEK_API_KEY="ds-test",
            DEEPSEEK_BASE_URL=f"{self.fake.url}/v1",
        )
        chats = [r for r in self.fake.take() if r["path"].endswith("/chat/completions")]
        self.assertIn("entregue", text)
        self.assertEqual(calls, ["ping"])
        self.assertGreaterEqual(len(chats), 2)
        for request in chats:
            self.assertEqual(request["path"], "/v1/chat/completions")
            self.assertEqual(request["body"].get("model"), "deepseek-flash")
            self.assertEqual(request["body"].get("thinking"), {"type": "disabled"})
            self.assertEqual(request["auth"], "Bearer ds-test")

    def test_fireworks_desk_uses_chat_completions_at_its_base_url(self):
        text, calls = self._run_desk(
            f"fireworks/{FIREWORKS_MODEL}",
            FIREWORKS_API_KEY="fw-test",
            FIREWORKS_BASE_URL=f"{self.fake.url}/inference/v1",
        )
        seen = self.fake.take()
        chats = [r for r in seen if r["path"].endswith("/chat/completions")]
        self.assertIn("entregue", text)
        self.assertEqual(calls, ["ping"])
        self.assertGreaterEqual(len(chats), 2)
        self.assertEqual([r["path"] for r in seen if "/responses" in r["path"]], [])
        for request in chats:
            self.assertEqual(request["path"], "/inference/v1/chat/completions")
            self.assertEqual(request["body"].get("model"), FIREWORKS_MODEL)
            self.assertNotIn("thinking", request["body"])
            self.assertEqual(request["auth"], "Bearer fw-test")

    def test_missing_key_stops_the_desk_before_any_request(self):
        from runtime.crew_loop import run_desk_task

        with env(CREWAI_VERBOSE="false"):
            with self.assertRaisesRegex(RuntimeError, "FIREWORKS_API_KEY"):
                asyncio.run(run_desk_task(agent_name="Rafa", role="r", backstory="b", prompt="p", model="fireworks/m"))
        self.assertEqual(self.fake.take(), [])

    def test_complete_text_runs_on_the_desk_provider(self):
        from office import Office

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp, env(
            OFFICE_OPENAI_API_KEY="local",
            OFFICE_OPENAI_BASE_URL=f"{self.fake.url}/v1",
            COMPANY_WORKSPACE=os.path.join(tmp, "workspace"),
        ):
            office = Office(os.path.join(tmp, "data", "blackboard"), config={})
            text = asyncio.run(office.complete_text(
                {"name": "Rafa", "role": "Diretor", "model": "openai/test-model"},
                "Resuma o quadro.",
            ))
        chats = [r for r in self.fake.take() if r["path"].endswith("/chat/completions")]
        self.assertEqual(text, "entregue")
        self.assertEqual(chats[0]["path"], "/v1/chat/completions")
        self.assertEqual(chats[0]["body"]["model"], "test-model")
        self.assertEqual(chats[0]["body"]["messages"][0]["role"], "system")


class EmbedderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fake = FakeOpenAI()

    @classmethod
    def tearDownClass(cls):
        cls.fake.close()

    def setUp(self):
        self.fake.take()

    def _env(self, **values: str):
        return env(FIREWORKS_API_KEY="fw-test", FIREWORKS_BASE_URL=f"{self.fake.url}/inference/v1", **values)

    def test_sends_dimensions_and_returns_unit_vectors(self):
        with self._env():
            embedder = OpenAICompatibleEmbedder(providers.resolve("fireworks/fireworks/qwen3-embedding-0p6b"), dim=8)
            vector = embedder.encode("olá", normalize_embeddings=True)
            batch = embedder.encode_batch(["a", "bb"])
        request = self.fake.take()[0]
        self.assertEqual(request["path"], "/inference/v1/embeddings")
        self.assertEqual(request["body"]["model"], "fireworks/qwen3-embedding-0p6b")
        self.assertEqual(request["body"]["dimensions"], 8)
        self.assertEqual(request["auth"], "Bearer fw-test")
        self.assertEqual(vector.shape, (8,))
        self.assertAlmostEqual(float(np.linalg.norm(vector)), 1.0, places=5)
        self.assertEqual(batch.shape, (2, 8))

    def test_fixed_size_model_is_retried_without_dimensions(self):
        with self._env():
            embedder = OpenAICompatibleEmbedder(providers.resolve("fireworks/fixed-small"), dim=8)
            vector = embedder.encode("olá")
        first, second = self.fake.take()[:2]
        self.assertIn("dimensions", first["body"])
        self.assertNotIn("dimensions", second["body"])
        self.assertEqual(vector.shape, (8,))
        self.assertTrue(np.all(vector[4:] == 0))

    def test_blocked_provider_never_sends(self):
        with self._env(OFFICE_ALLOWED_PROVIDERS="gemini"):
            embedder = OpenAICompatibleEmbedder(providers.resolve("fireworks/m"), dim=8)
            with self.assertRaisesRegex(RuntimeError, "blocked"):
                embedder.encode("olá")
        self.assertEqual(self.fake.take(), [])


@unittest.skipUnless(_has("epochdb"), "epochdb is not installed")
class BoardEmbeddingTests(unittest.TestCase):
    def test_board_stores_and_recalls_through_the_remote_embedder(self):
        from runtime.blackboard import EpochBlackboard

        fake = FakeOpenAI()
        try:
            with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp, env(
                OFFICE_EMBED_MODEL="fireworks/fireworks/qwen3-embedding-0p6b",
                FIREWORKS_API_KEY="fw-test",
                FIREWORKS_BASE_URL=f"{fake.url}/inference/v1",
                EPOCHDB_DIM="8",
            ):
                board = EpochBlackboard(tmp)
                self.assertEqual(board.embedding_provider, "fireworks")
                asyncio.run(board.write_node(
                    "note:1",
                    "Note",
                    {"title": "Pauta", "text": "Agentes precisam de memória durável entre tarefas."},
                ))
                recalled = asyncio.run(board.recall("memória de agentes", k=2))
            paths = [r["path"] for r in fake.take()]
        finally:
            fake.close()
        self.assertGreaterEqual(paths.count("/inference/v1/embeddings"), 2)
        self.assertIn("memória durável", recalled)


class AnonymousModeTests(unittest.TestCase):
    def _office(self, tmp: str, agents: list):
        from office import Office

        return Office(os.path.join(tmp, "data", "blackboard"), config={"agents": agents})

    def test_desks_and_embeddings_outside_the_list_stop_the_start(self):
        agents = [
            {"id": "rafa", "name": "Rafa", "role": "Diretor"},
            {"id": "bete", "name": "Bete", "role": "Publicação", "model": "gemini-3.5-flash-lite"},
        ]
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp, env(
            OFFICE_ALLOWED_PROVIDERS="fireworks",
            OFFICE_TEXT_MODEL=f"fireworks/{FIREWORKS_MODEL}",
            EPOCHDB_MODEL="google:gemini-embedding-2",
            COMPANY_WORKSPACE=os.path.join(tmp, "workspace"),
        ):
            problems = "\n".join(self._office(tmp, agents).provider_problems())
        self.assertIn("desk 'bete'", problems)
        self.assertNotIn("desk 'rafa'", problems)
        self.assertIn("board embeddings uses gemini", problems)

    def test_unknown_provider_is_reported_without_anonymous_mode(self):
        agents = [{"id": "laura", "name": "Laura", "role": "Redatora", "model": "firework/x"}]
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp, env(
            COMPANY_WORKSPACE=os.path.join(tmp, "workspace"),
        ):
            problems = self._office(tmp, agents).provider_problems()
        self.assertEqual(len(problems), 1)
        self.assertIn("Unknown provider 'firework'", problems[0])

    def test_gemini_extras_follow_the_key_and_the_list(self):
        with env(GEMINI_API_KEY="g-test"):
            self.assertTrue(providers.gemini_extras_available())
        with env(GEMINI_API_KEY="g-test", OFFICE_ALLOWED_PROVIDERS="fireworks"):
            self.assertFalse(providers.gemini_extras_available())
        with env():
            self.assertFalse(providers.gemini_extras_available())

    def test_cover_tool_without_gemini_tells_bete_to_publish_text_only(self):
        from blog_images import blog_image_tools

        with tempfile.TemporaryDirectory() as tmp:
            tool = blog_image_tools(Path(tmp), None)[0]
            reply = asyncio.run(tool.execute(title="Agentes", description="redação"))
        self.assertIn("só com texto", reply)

    def test_privacy_switches_follow_anonymous_mode(self):
        names = ("CREWAI_DISABLE_TELEMETRY", "OTEL_SDK_DISABLED", "CREWAI_TRACING_ENABLED")
        with env():
            for name in names:
                os.environ.pop(name, None)
            providers.apply_privacy_defaults()
            self.assertEqual([os.environ.get(name) for name in names], [None, None, None])
        with env(OFFICE_ALLOWED_PROVIDERS="fireworks", CREWAI_TRACING_ENABLED="true"):
            providers.apply_privacy_defaults()
            self.assertEqual([os.environ.get(name) for name in names], ["true", "true", "false"])

    @unittest.skipUnless(_has("crewai") and _has("fastapi"), "crewai and fastapi are needed")
    def test_anonymous_office_reaches_only_the_allowed_provider(self):
        """Imports the real app in anonymous mode and runs one desk with every
        non-loopback DNS lookup blocked and recorded."""
        script = textwrap.dedent("""
            import asyncio, json, os, socket, sys

            sys.path.insert(0, sys.argv[1])
            outside = []
            real_getaddrinfo = socket.getaddrinfo

            def guarded_getaddrinfo(host, *args, **kwargs):
                name = host.decode() if isinstance(host, bytes) else str(host)
                if name not in ("127.0.0.1", "localhost", "::1"):
                    outside.append(name)
                    raise OSError("blocked by test: " + name)
                return real_getaddrinfo(host, *args, **kwargs)

            socket.getaddrinfo = guarded_getaddrinfo

            import main  # noqa: F401  the real import graph, in anonymous mode
            from runtime.crew_loop import run_desk_task
            from runtime.tools import Tool

            calls = []

            async def echo_tool(text: str) -> str:
                calls.append(text)
                return "eco: " + text

            text = asyncio.run(run_desk_task(
                agent_name="Rafa",
                role="Diretor",
                backstory="Teste anonimo.",
                prompt="Chame echo_tool uma vez e entregue. (anon)",
                tools=[Tool(echo_tool, name="echo_tool", description="Repeat the text back. Call it once.")],
                max_iterations=4,
            ))
            report = {
                "text": text,
                "calls": calls,
                "outside": outside,
                "google_genai": sorted(m for m in sys.modules if m == "google.genai" or m.startswith("google.genai.")),
                "telemetry": [os.environ.get(k) for k in (
                    "CREWAI_DISABLE_TELEMETRY", "OTEL_SDK_DISABLED", "CREWAI_TRACING_ENABLED"
                )],
            }
            with open(sys.argv[2], "w", encoding="utf-8") as handle:
                json.dump(report, handle)
        """)
        fake = FakeOpenAI()
        try:
            with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
                child_env = {
                    key: value
                    for key, value in os.environ.items()
                    if key not in _PROVIDER_VARS and not key.startswith(("CREWAI_", "OTEL_"))
                }
                child_env.update({
                    "OFFICE_ALLOWED_PROVIDERS": "fireworks",
                    "OFFICE_TEXT_MODEL": f"fireworks/{FIREWORKS_MODEL}",
                    "FIREWORKS_API_KEY": "fw-test",
                    "FIREWORKS_BASE_URL": f"{fake.url}/inference/v1",
                    "OFFICE_STORAGE_DIR": os.path.join(tmp, "data", "blackboard"),
                    "COMPANY_WORKSPACE": os.path.join(tmp, "workspace"),
                    "OFFICE_SCHEDULER_ENABLED": "false",
                    "CREWAI_VERBOSE": "false",
                    "HOME": tmp,
                })
                report_path = os.path.join(tmp, "report.json")
                result = subprocess.run(
                    [sys.executable, "-c", script, str(BACKEND), report_path],
                    env=child_env,
                    cwd=tmp,
                    capture_output=True,
                    text=True,
                    timeout=300,
                )
                self.assertEqual(result.returncode, 0, result.stderr[-4000:])
                with open(report_path, encoding="utf-8") as handle:
                    report = json.load(handle)
            chats = [r for r in fake.take() if r["path"].endswith("/chat/completions")]
        finally:
            fake.close()
        self.assertIn("entregue", report["text"])
        self.assertEqual(report["calls"], ["ping"])
        self.assertEqual(report["outside"], [], "something tried to reach the internet")
        self.assertEqual(report["google_genai"], [])
        self.assertEqual(report["telemetry"], ["true", "true", "false"])
        self.assertTrue(chats)


if __name__ == "__main__":
    unittest.main()
