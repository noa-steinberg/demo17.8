"""Entry point for the web local-LLM chat app (noastein-ex07, version 3).

Usage
-----
    python app.py                 # start the web server, print the URL
    python app.py --port 9000     # serve on a chosen port
    python app.py --smoke         # headless one-shot through the engine (CI)

Architecture (Web edition): a **web GUI** (``web/index.html`` + ``style.css``
+ ``app.js``) served by a **minimal stdlib Python backend** (``src/server.py``).
The backend holds the API key and makes the genuine ``Authorization: Bearer``
call to Ollama — the browser never sees the key (browser → backend → Ollama).

Configuration comes from environment variables / a ``.env`` file (see
``.env.example``). With ``OLLAMA_MOCK=1`` — or whenever no Ollama server is
running — the backend uses a deterministic offline mock so it always serves.
"""
from __future__ import annotations

import argparse
import sys

from src.backend import build_client
from src.config import Config
from src.mock_backend import MockClient
from src.server import chat_once, create_server


def _describe(config: Config) -> str:
    client = build_client(config)
    kind = "offline mock" if isinstance(client, MockClient) else "live API"
    return f"{config.model} · {kind}"


def smoke(config: Config | None = None) -> int:
    """Send one message through the server's engine without a browser.

    Exercises the exact same ``chat_once`` path the ``/api/chat`` handler uses,
    so a green ``--smoke`` proves the request pipeline end-to-end (minus HTTP).
    """
    config = config or Config.from_env()
    client = build_client(config)
    kind = "mock" if isinstance(client, MockClient) else "live"
    messages = [{"role": "user", "content": "Hello in one sentence, who are you?"}]
    payload = chat_once(config, client, messages)
    print(f"[backend={kind} model={config.model}] tokens~{payload['tokens']} "
          f"injection_flagged={payload['injection_flagged']}")
    print("Assistant:", payload["content"])
    return 0 if payload["content"] else 1


def serve(config: Config, port: int) -> int:
    """Start the HTTP server and block until Ctrl-C."""
    server = create_server(config, port)
    actual_port = server.server_address[1]
    url = f"http://localhost:{actual_port}"
    print(f"noastein-ex07 web chat ({_describe(config)})")
    print(f"  serving on {url}   (Ctrl-C to stop)")
    print(f"  demo view: {url}/?demo=1")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down.")
    finally:
        server.server_close()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Local-LLM web chat (Ollama, stdlib backend).")
    parser.add_argument("--smoke", action="store_true", help="headless one-shot test (no browser)")
    parser.add_argument("--port", type=int, default=8000, help="port to serve on (default 8000)")
    args = parser.parse_args(argv)

    config = Config.from_env()
    if args.smoke:
        return smoke(config)
    return serve(config, args.port)


if __name__ == "__main__":
    sys.exit(main())
