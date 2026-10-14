# The Office

A local newsroom you can watch. Five CrewAI desks research, write, edit, and — if you configure a blog — publish on a schedule. The shared board lives in EpochDB. Nothing in this repo is wired to a public hostname.

![Isometric view of the newsroom: desks, agents, and the EpochDB board](screenshot.png)

Open it at **http://127.0.0.1:8780**. The server binds to localhost. Docker publishes that same address and does not expose the port on your LAN.

## Interactive mode

The server starts view-only: you can watch the floor, the board, and the desk monitors. Interactive mode adds talk, task assignment, notes, and archiving.

To enable interactive mode, set this in `.env` and restart:

```bash
OFFICE_READ_ONLY=false
```

`false` is the only value that opens it. `true`, empty, `0`, and `no` all stay view-only. The badge in the corner shows which mode is running.

With Docker, recreate the container so it reads the new value:

```bash
docker compose -f docker/docker-compose.yml --env-file .env up -d
```

Interactive controls:

- Click a person and use the card at the bottom left. **Falar** opens a Gemini Live call with that desk: you can speak or type, and they can take the request, hand it to a teammate, or pin a note. Voice needs Gemini; without it the button is hidden and the card only assigns tasks.
- In first-person or VR, look at someone. The microphone follows that person until you mute it.
- Open **Quadro** to assign a task, pin a note with no owner, or archive a card.

Voice audio goes from this browser to Gemini and back. It is not stored in the repo. View-only mode never opens the microphone.

The daily scheduler still runs in both modes. Interactive mode adds you as another source of tasks. It does not add a public chat API.

## Roster

Edit [`config/office.yml`](config/office.yml), then restart.

| Agent | Desk |
|-------|------|
| Rafa | Director — one sourced story per slot |
| Beto | Editor-in-chief — routes the job and closes it |
| Laura | Writer |
| Elisa | Copy editor |
| Bete | Publishing, including one cover image |

Default cadence is 3 articles, at 09:00, 13:00, and 17:00 in `America/Sao_Paulo`. Each slot opens one job.

## How a job moves

```
you or the scheduler
        │
        ▼
   Rafa (CrewAI) ──hand_off──▶ EpochDB blackboard
                                  Job · Task · Result
                                      │
                    Beto → Laura → Elisa → Bete
```

CrewAI **1.15.26** runs the desk that just woke up. Delegation inside CrewAI is off, so work moves only through `hand_off` on the board. When a desk finishes, EpochDB stores the delivery. The next desk recalls related atoms before it starts.

CrewAI's own LanceDB memory is disabled. Recall goes through EpochDB.

## Run it

```bash
cd the-office-crewai
cp .env.example .env
# set GEMINI_API_KEY, or another provider (see Providers)
# set OFFICE_READ_ONLY=false to enable interactive mode

docker compose -f docker/docker-compose.yml --env-file .env up --build
```

Then open http://127.0.0.1:8780.

Without Docker:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r backend/requirements.txt
playwright install chromium
cd frontend && npm install && npm run build && cd ..
python backend/main.py
```

Frontend dev server (proxies to 127.0.0.1:8780):

```bash
cd frontend && npm run dev
```

## Providers

Desks run on Gemini unless you pick something else. A model is written `<provider>/<model>`. A bare name, or a Gemini resource name like `models/gemini-2.5-flash`, means Gemini, so existing configs keep working.

| Provider | Example model | Key |
|----------|---------------|-----|
| `gemini` (default) | `gemini/gemini-3.5-flash-lite` | `GEMINI_API_KEY` |
| `deepseek` | `deepseek/deepseek-flash` | `DEEPSEEK_API_KEY` |
| `fireworks` | `fireworks/accounts/fireworks/models/deepseek-v4p1-flash` | `FIREWORKS_API_KEY` |
| `openai` | `openai/<model>` on any OpenAI-compatible server (vLLM, Ollama, a gateway) | `OFFICE_OPENAI_API_KEY` and `OFFICE_OPENAI_BASE_URL` |

`OFFICE_TEXT_MODEL` sets the model for every desk. A desk can send its own work elsewhere with `model:` in `config/office.yml`:

```yaml
  - id: laura
    name: Laura
    model: fireworks/accounts/fireworks/models/deepseek-v4p1-flash
```

- DeepSeek desks run with thinking off. In thinking mode with tools, DeepSeek requires `reasoning_content` to be sent back on every request, and CrewAI does not send it.
- Fireworks and `openai/` servers are called through Chat Completions only, never the Responses API, which can store conversations on the provider.
- Live voice (**Falar**) and cover images exist only on Gemini. Without Gemini, **Falar** is hidden, the first-person view says voice is unavailable, and Bete publishes text-only posts.
- Board recall follows `OFFICE_EMBED_MODEL`: `gemini/…`, `fireworks/…`, `openai/…`, `local/<sentence-transformers model>`, or `off`. `local/…` needs `pip install "epochdb[embeddings]"`, which brings in PyTorch and is not in the default install or image; without it, recall stays off and the log says why. Fireworks embedding ids start with `fireworks/`, so the value repeats it: `fireworks/fireworks/qwen3-embedding-8b`. Changing the embedding model or `EPOCHDB_DIM` needs a fresh `data/`.
- An unknown provider name stops the server at startup and names the desk.

## Anonymous mode

`OFFICE_ALLOWED_PROVIDERS` lists the only providers the office may call, for example `fireworks`. With it set:

- The server refuses to start if a desk or the board embeddings resolve to another provider, and says which.
- Covers and voice switch off unless `gemini` is in the list.
- CrewAI telemetry and tracing are turned off.

Fireworks does not store prompts or outputs for open models by default ([data handling](https://docs.fireworks.ai/guides/security_compliance/data_handling)). `local/…` embeddings run on this machine once `epochdb[embeddings]` is installed; the model downloads once. Publishing to Blogger talks to Google by design: leave the `BLOGGER_*` settings empty to keep the office off Google entirely.

Set `OFFICE_BROWSER_PROXY` (for example `socks5://127.0.0.1:9050`) so the desks' Chromium does not browse from your address. API calls follow the standard `HTTPS_PROXY`. The page itself still loads two fonts from Google Fonts.

A Fireworks-only `.env`:

```bash
OFFICE_ALLOWED_PROVIDERS=fireworks
FIREWORKS_API_KEY=...
OFFICE_TEXT_MODEL=fireworks/accounts/fireworks/models/deepseek-v4p1-flash
OFFICE_EMBED_MODEL=fireworks/fireworks/qwen3-embedding-8b
EPOCHDB_DIM=768
```

## Environment

Copy `.env.example` to `.env`. Restart after any change. Docker reads that file through `--env-file .env`. Inside the container, Compose forces `OFFICE_HOST=0.0.0.0` and still publishes the port only on `127.0.0.1`.

| Variable | Default | Purpose |
|----------|---------|---------|
| `GEMINI_API_KEY` | empty | Gemini key. Needed when desks, embeddings, covers, or voice use Gemini, which is the default. |
| `OFFICE_READ_ONLY` | `true` | `false` enables interactive mode. Any other value stays view-only. |
| `OFFICE_HOST` | `127.0.0.1` | Address for `python backend/main.py`. Docker overrides this inside the container. |
| `OFFICE_PORT` | `8780` | Listen port, and the host port Docker publishes on `127.0.0.1`. |
| `APP_URL` | `http://127.0.0.1:8780` | Local URL shown for this office. |
| `OFFICE_TEXT_MODEL` | `gemini-3.5-flash-lite` | Model for every desk, as `<provider>/<model>`. See [Providers](#providers). |
| `GEMINI_TEXT_MODEL` | empty | Older name for `OFFICE_TEXT_MODEL`, used when that is unset. |
| `DEEPSEEK_API_KEY` | empty | Key for `deepseek/…` models. `DEEPSEEK_BASE_URL` overrides the endpoint. |
| `FIREWORKS_API_KEY` | empty | Key for `fireworks/…` models. `FIREWORKS_BASE_URL` overrides the endpoint. |
| `OFFICE_OPENAI_BASE_URL` | empty | Base URL for `openai/…` models, for example `http://127.0.0.1:8000/v1`. |
| `OFFICE_OPENAI_API_KEY` | empty | Key for that server. Any value works for a server without auth. |
| `GEMINI_VOICE_MODEL` | `gemini-3.8-live` | Gemini Live model for **Falar**. Used only when interactive mode is on. |
| `GEMINI_IMAGE_MODEL` | `gemini-3.1-flash-image` | Model Bete uses for the one cover image per post. |
| `OFFICE_EMBED_MODEL` | Gemini when its key is set, else `off` | Board recall embeddings, as `<provider>/<model>`, `local/<model>`, or `off`. Wins over `EPOCHDB_MODEL`. |
| `EPOCHDB_MODEL` | `google:gemini-embedding-2` when an API key is set | Embedding model in EpochDB's own format. `EPOCHDB_EMBEDDING_MODEL` is an alias. |
| `EPOCHDB_DIM` | `768` | Embedding size. `768` matches `gemini-embedding-2` and is the default for `fireworks/` and `openai/` embeddings. |
| `OFFICE_ALLOWED_PROVIDERS` | empty | Anonymous mode: comma-separated providers the office may call. See [Anonymous mode](#anonymous-mode). |
| `OFFICE_BROWSER_PROXY` | empty | `http://host:port` or `socks5://host:port` for the desks' Chromium. |
| `OFFICE_LANGUAGE` | `pt-BR` | Language the desks write and speak. `en` or `en-US` selects English. When this is unset, `language` in `config/office.yml` is used. |
| `TZ` | `America/Sao_Paulo` | Clock for the process. Publish-slot times follow `timezone` in `config/office.yml`. |
| `OFFICE_SCHEDULER_ENABLED` | `true` | `false`, `0`, `no`, or `off` pauses new daily slots. Tasks already on the board still resume. |
| `OFFICE_KICKOFF_ON_START` | `false` | `true`, `1`, or `yes` opens one slot as soon as the server boots. |
| `CREWAI_VERBOSE` | `true` | `false`, `0`, or `no` hides the desk tool log. |
| `BLOGGER_BLOG_URL` | empty | Blog to publish to. Leave empty to keep publishing off. |
| `BLOGGER_CLIENT_ID` | empty | Google OAuth client for the Blogger API. |
| `BLOGGER_CLIENT_SECRET` | empty | OAuth client secret. Keep this in `.env` only. |
| `BLOGGER_TOKEN_FILE` | `./secrets/blogger_credentials.json` | Where `blogger_auth.py` writes the token. Docker stores the writable copy under `data/`. |

## Publishing

Publishing is off until you opt in. There is no blog URL in the repo.

1. Create a Google OAuth desktop client with the Blogger API enabled.
2. Put `BLOGGER_CLIENT_ID`, `BLOGGER_CLIENT_SECRET`, and `BLOGGER_BLOG_URL` in `.env` (not in git).
3. Run `python backend/blogger_auth.py`. It listens on http://127.0.0.1:8080/ and writes a token JSON.
4. Move that file to `secrets/blogger_credentials.json`. Docker mounts `secrets/` read-only and copies the token into `data/` so a refresh can be saved.

`post_to_blogger` refuses a near-duplicate title. Do not enable force.

## What stays on your machine

These paths are gitignored. Do not commit them.

| Path | Contents |
|------|----------|
| `.env` | API keys and OAuth client |
| `secrets/` | Blogger token JSON. Only `.gitkeep` is tracked. |
| `data/` | EpochDB WAL, knowledge graph, job state, refreshed token |
| `workspace/**/*.md` | Articles, pautas, and publication reports the desks write |
| `workspace/imagens/` | Cover files |

A restart reloads the board from `data/blackboard/blackboard/global/wal.jsonl`. Deleting `data/` starts a blank newsroom.

## Layout

```
config/office.yml       roster, slots, prompts
backend/main.py         localhost API, view-only or interactive
backend/office.py       desks, jobs, handoffs
backend/scheduler.py    one article per slot
backend/runtime/        CrewAI kickoff, EpochDB, browser, Blogger
frontend/               isometric / FPS / VR
docker/                 local compose file, bound to 127.0.0.1
```

## Stack

| Piece | Role |
|-------|------|
| CrewAI 1.15.26 | Tool loop for the desk that is awake |
| Models | Desks use Gemini `gemini-3.5-flash-lite` by default, or DeepSeek, Fireworks, or any OpenAI-compatible server. Live voice (`gemini-3.8-live`) and covers need Gemini. |
| EpochDB ≥ 1.11.2 | Board atoms and recall. Embeddings follow `OFFICE_EMBED_MODEL`; the default is `google:gemini-embedding-2` at 768 dimensions when `GEMINI_API_KEY` is set |
| FastAPI | Health, office snapshot, jobs, SSE, monitor WebSocket, and the interactive writes |
| Playwright | One headless Chromium page per desk, mirrored onto that monitor |

## License

[MIT](LICENSE). Dependencies stay under their own licenses and are not vendored into this repo.

The ambient loop in `frontend/src/assets/bg.wav` is by Fupicat, who gave permission to use it and does not require credit.
