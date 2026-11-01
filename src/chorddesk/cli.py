"""Minimal interactive command-line interface for ChordDesk."""

from __future__ import annotations

import os
import sys
from collections.abc import Callable, Mapping
from typing import Protocol, TextIO

from openai import OpenAIError

from chorddesk.bootstrap import build_session
from chorddesk.openai_agent import AgentError, ToolProtocolError


WELCOME = "ChordDesk pronto. Digite sua mensagem (/reset, /exit ou /quit)."
RESET_CONFIRMATION = "Conversa reiniciada."
MESSAGE_FAILURE = "Não foi possível responder agora. Tente novamente."


class Session(Protocol):
    def respond(self, user_message: str) -> str: ...

    def reset(self) -> None: ...


def run_interactive(
    session: Session,
    *,
    input_stream: TextIO = sys.stdin,
    output_stream: TextIO = sys.stdout,
) -> None:
    """Run one session until a local exit command, EOF, or interruption."""
    print(WELCOME, file=output_stream)
    while True:
        try:
            line = input_stream.readline()
        except KeyboardInterrupt:
            print(file=output_stream)
            return

        if line == "":
            return

        message = line.strip()
        if not message:
            continue
        if message in {"/exit", "/quit"}:
            return
        if message == "/reset":
            session.reset()
            print(RESET_CONFIRMATION, file=output_stream)
            continue

        try:
            answer = session.respond(message)
        except KeyboardInterrupt:
            print(file=output_stream)
            return
        except (AgentError, ToolProtocolError, OpenAIError):
            print(MESSAGE_FAILURE, file=output_stream)
            continue
        print(answer, file=output_stream)


def main(
    *,
    session_factory: Callable[[], Session] | None = None,
    environ: Mapping[str, str] = os.environ,
    input_stream: TextIO = sys.stdin,
    output_stream: TextIO = sys.stdout,
    error_stream: TextIO = sys.stderr,
) -> int:
    """Start the production CLI, with injectable boundaries for offline tests."""
    if session_factory is None:
        if not environ.get("OPENAI_API_KEY", "").strip():
            print(
                "Erro: defina OPENAI_API_KEY antes de iniciar o ChordDesk.",
                file=error_stream,
            )
            return 2
        session_factory = build_session

    try:
        session = session_factory()
    except (FileNotFoundError, ValueError) as exc:
        print(f"Erro ao iniciar o ChordDesk: {exc}", file=error_stream)
        return 2

    run_interactive(
        session,
        input_stream=input_stream,
        output_stream=output_stream,
    )
    return 0
