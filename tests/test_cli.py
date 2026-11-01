from __future__ import annotations

import io
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from chorddesk.cli import MESSAGE_FAILURE, RESET_CONFIRMATION, main, run_interactive
from chorddesk.openai_agent import AgentError, ToolProtocolError


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class FakeSession:
    def __init__(self, replies: list[str] | None = None) -> None:
        self.replies = list(replies or [])
        self.messages: list[str] = []
        self.reset_calls = 0

    def respond(self, user_message: str) -> str:
        self.messages.append(user_message)
        return self.replies.pop(0)

    def reset(self) -> None:
        self.reset_calls += 1


class InterruptingInput:
    def readline(self) -> str:
        raise KeyboardInterrupt


class CliLoopTests(unittest.TestCase):
    def run_cli(self, text: str, session: FakeSession) -> str:
        output = io.StringIO()
        run_interactive(
            session,
            input_stream=io.StringIO(text),
            output_stream=output,
        )
        return output.getvalue()

    def test_normal_message_is_sent_and_reply_is_printed(self) -> None:
        session = FakeSession(["Olá!"])

        output = self.run_cli("Oi\n/exit\n", session)

        self.assertEqual(session.messages, ["Oi"])
        self.assertIn("Olá!", output)

    def test_two_messages_reuse_the_same_session(self) -> None:
        session = FakeSession(["Primeira", "Segunda"])

        output = self.run_cli("um\ndois\n/quit\n", session)

        self.assertEqual(session.messages, ["um", "dois"])
        self.assertIn("Primeira", output)
        self.assertIn("Segunda", output)

    def test_reset_resets_without_responding(self) -> None:
        session = FakeSession()

        output = self.run_cli("/reset\n/exit\n", session)

        self.assertEqual(session.reset_calls, 1)
        self.assertEqual(session.messages, [])
        self.assertIn(RESET_CONFIRMATION, output)

    def test_exit_and_quit_exit_cleanly(self) -> None:
        for command in ("/exit", "/quit"):
            with self.subTest(command=command):
                session = FakeSession()
                self.run_cli(f"{command}\nignored\n", session)
                self.assertEqual(session.messages, [])

    def test_blank_input_and_eof_are_ignored(self) -> None:
        session = FakeSession()

        self.run_cli("\n   \n", session)

        self.assertEqual(session.messages, [])

    def test_keyboard_interrupt_exits_without_traceback(self) -> None:
        output = io.StringIO()

        run_interactive(
            FakeSession(),
            input_stream=InterruptingInput(),  # type: ignore[arg-type]
            output_stream=output,
        )

        self.assertNotIn("Traceback", output.getvalue())

    def test_agent_failure_is_safe_and_next_message_can_retry(self) -> None:
        session = Mock()
        session.respond.side_effect = [AgentError("private details"), "Tudo certo."]
        output = io.StringIO()

        run_interactive(
            session,
            input_stream=io.StringIO("falha\ntente novamente\n/exit\n"),
            output_stream=output,
        )

        rendered = output.getvalue()
        self.assertIn(MESSAGE_FAILURE, rendered)
        self.assertIn("Tudo certo.", rendered)
        self.assertNotIn("private details", rendered)
        self.assertEqual(session.respond.call_count, 2)

    def test_tool_protocol_failure_is_safe_and_next_message_can_retry(self) -> None:
        session = Mock()
        session.respond.side_effect = [
            ToolProtocolError("malformed private tool arguments"),
            "Tudo certo.",
        ]
        output = io.StringIO()

        run_interactive(
            session,
            input_stream=io.StringIO("falha\ntente novamente\n/exit\n"),
            output_stream=output,
        )

        rendered = output.getvalue()
        self.assertIn(MESSAGE_FAILURE, rendered)
        self.assertIn("Tudo certo.", rendered)
        self.assertNotIn("malformed private tool arguments", rendered)
        self.assertEqual(session.respond.call_count, 2)

    def test_output_does_not_print_session_internals(self) -> None:
        session = FakeSession(["Resposta segura"])
        session.history = "INTERNAL_TOOL_DATA"

        output = self.run_cli("pergunta\n", session)

        self.assertNotIn("INTERNAL_TOOL_DATA", output)


class CliStartupTests(unittest.TestCase):
    def test_missing_api_key_fails_before_bootstrap(self) -> None:
        errors = io.StringIO()
        with patch("chorddesk.cli.build_session") as bootstrap:
            status = main(environ={}, error_stream=errors)

        self.assertNotEqual(status, 0)
        self.assertIn("OPENAI_API_KEY", errors.getvalue())
        bootstrap.assert_not_called()

    def test_injected_session_needs_no_api_key(self) -> None:
        session = FakeSession()

        status = main(
            session_factory=lambda: session,
            environ={},
            input_stream=io.StringIO("/exit\n"),
            output_stream=io.StringIO(),
            error_stream=io.StringIO(),
        )

        self.assertEqual(status, 0)

    def test_missing_data_error_is_concise(self) -> None:
        errors = io.StringIO()

        status = main(
            session_factory=lambda: (_ for _ in ()).throw(
                FileNotFoundError("required file missing")
            ),
            environ={},
            input_stream=io.StringIO(),
            output_stream=io.StringIO(),
            error_stream=errors,
        )

        self.assertNotEqual(status, 0)
        self.assertIn("required file missing", errors.getvalue())
        self.assertNotIn("Traceback", errors.getvalue())

    def test_console_script_points_to_cli(self) -> None:
        pyproject = (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
        self.assertIn('chorddesk = "chorddesk.cli:main"', pyproject)


if __name__ == "__main__":
    unittest.main()
