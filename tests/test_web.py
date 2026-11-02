from __future__ import annotations

import unittest

from fastapi.testclient import TestClient
from openai import OpenAIError

from chorddesk.openai_agent import AgentError, ToolProtocolError
from chorddesk.web import (
    COOKIE_NAME,
    DEFAULT_MAX_MESSAGE_LENGTH,
    MESSAGE_TOO_LONG_DETAIL,
    RATE_LIMIT_DETAIL,
    SAFE_FAILURE_MESSAGE,
    create_app,
)


class FakeClock:
    def __init__(self) -> None:
        self.value = 1_000.0

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


class FakeSession:
    def __init__(self) -> None:
        self.messages: list[str] = []
        self.reset_calls = 0
        self.failure: Exception | None = None

    def respond(self, user_message: str) -> str:
        self.messages.append(user_message)
        if self.failure is not None:
            raise self.failure
        return f"Resposta {len(self.messages)}: {user_message}"

    def reset(self) -> None:
        self.reset_calls += 1
        self.messages.clear()


class FakeSessionFactory:
    def __init__(self) -> None:
        self.sessions: list[FakeSession] = []

    def __call__(self) -> FakeSession:
        session = FakeSession()
        self.sessions.append(session)
        return session


class WebTests(unittest.TestCase):
    def setUp(self) -> None:
        self.factory = FakeSessionFactory()
        self.client = TestClient(
            create_app(session_factory=self.factory, environ={}),
            raise_server_exceptions=False,
        )

    def test_index_returns_responsive_chat_interface(self) -> None:
        response = self.client.get("/")

        self.assertEqual(response.status_code, 200)
        self.assertIn("ChordDesk", response.text)
        self.assertIn("Empório da Música", response.text)
        self.assertIn('name="viewport"', response.text)
        self.assertIn('id="messages"', response.text)
        self.assertIn('id="input"', response.text)
        self.assertIn('id="reset"', response.text)
        self.assertNotIn("innerHTML", response.text)
        self.assertEqual(self.factory.sessions, [])

    def test_health_does_not_construct_a_session(self) -> None:
        response = self.client.get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})
        self.assertEqual(self.factory.sessions, [])

    def test_blank_chat_request_is_rejected_without_a_session(self) -> None:
        response = self.client.post("/api/chat", json={"message": "  \n "})

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json(), {"detail": "A mensagem não pode estar vazia."})
        self.assertEqual(self.factory.sessions, [])

    def test_successful_chat_uses_injected_session_factory(self) -> None:
        response = self.client.post("/api/chat", json={"message": "  Olá  "})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"message": "Resposta 1: Olá"})
        self.assertEqual(self.factory.sessions[0].messages, ["Olá"])
        self.assertIn(COOKIE_NAME, self.client.cookies)
        self.assertEqual(response.headers["cache-control"], "no-store")
        cookie = response.headers["set-cookie"]
        self.assertIn("HttpOnly", cookie)
        self.assertIn("SameSite=lax", cookie)
        self.assertNotIn("Olá", cookie)

    def test_independent_browsers_have_independent_sessions(self) -> None:
        first = TestClient(self.client.app)
        second = TestClient(self.client.app)

        first_response = first.post("/api/chat", json={"message": "primeiro"})
        second_response = second.post("/api/chat", json={"message": "segundo"})

        self.assertEqual(first_response.json()["message"], "Resposta 1: primeiro")
        self.assertEqual(second_response.json()["message"], "Resposta 1: segundo")
        self.assertEqual(len(self.factory.sessions), 2)
        self.assertIsNot(self.factory.sessions[0], self.factory.sessions[1])
        self.assertNotEqual(first.cookies[COOKIE_NAME], second.cookies[COOKIE_NAME])

    def test_same_browser_preserves_session_across_messages(self) -> None:
        first = self.client.post("/api/chat", json={"message": "uma"})
        second = self.client.post("/api/chat", json={"message": "duas"})

        self.assertEqual(first.json()["message"], "Resposta 1: uma")
        self.assertEqual(second.json()["message"], "Resposta 2: duas")
        self.assertEqual(len(self.factory.sessions), 1)
        self.assertEqual(self.factory.sessions[0].messages, ["uma", "duas"])

    def test_reset_affects_only_requesting_browser(self) -> None:
        first = TestClient(self.client.app)
        second = TestClient(self.client.app)
        first.post("/api/chat", json={"message": "A"})
        second.post("/api/chat", json={"message": "B"})

        response = first.post("/api/reset")
        first_after = first.post("/api/chat", json={"message": "A2"})
        second_after = second.post("/api/chat", json={"message": "B2"})

        self.assertEqual(response.json(), {"ok": True})
        self.assertEqual(first_after.json()["message"], "Resposta 1: A2")
        self.assertEqual(second_after.json()["message"], "Resposta 2: B2")
        self.assertEqual(self.factory.sessions[0].reset_calls, 1)
        self.assertEqual(self.factory.sessions[1].reset_calls, 0)

    def test_expected_failures_return_generic_safe_response(self) -> None:
        failures = (
            AgentError("private agent details"),
            ToolProtocolError("private tool details"),
            OpenAIError("private API details"),
        )
        for failure in failures:
            with self.subTest(failure=type(failure).__name__):
                factory = FakeSessionFactory()
                client = TestClient(create_app(session_factory=factory, environ={}))
                client.post("/api/reset")
                factory.sessions[0].failure = failure

                response = client.post("/api/chat", json={"message": "teste"})

                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json(), {"message": SAFE_FAILURE_MESSAGE})
                self.assertNotIn(str(failure), response.text)

    def test_expected_factory_failure_returns_generic_safe_response(self) -> None:
        def failing_factory() -> FakeSession:
            raise OpenAIError("private API configuration details")

        client = TestClient(create_app(session_factory=failing_factory, environ={}))

        response = client.post("/api/chat", json={"message": "teste"})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"message": SAFE_FAILURE_MESSAGE})
        self.assertNotIn("private API configuration details", response.text)


class AbuseProtectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.clock = FakeClock()
        self.factory = FakeSessionFactory()

    def make_client(
        self,
        *,
        per_session_limit: int = 8,
        global_limit: int = 40,
    ) -> TestClient:
        app = create_app(
            session_factory=self.factory,
            per_session_chat_limit=per_session_limit,
            global_chat_limit=global_limit,
            clock=self.clock,
            environ={},
        )
        return TestClient(app)

    def test_exactly_2000_characters_is_accepted(self) -> None:
        client = self.make_client()
        message = "a" * DEFAULT_MAX_MESSAGE_LENGTH

        response = client.post("/api/chat", json={"message": message})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.factory.sessions[0].messages, [message])

    def test_2001_characters_is_rejected_without_calling_session(self) -> None:
        client = self.make_client()

        response = client.post(
            "/api/chat",
            json={"message": "a" * (DEFAULT_MAX_MESSAGE_LENGTH + 1)},
        )

        self.assertEqual(response.status_code, 413)
        self.assertEqual(response.json(), {"detail": MESSAGE_TOO_LONG_DETAIL})
        self.assertEqual(self.factory.sessions, [])

    def test_first_eight_requests_are_accepted_and_ninth_is_limited(self) -> None:
        client = self.make_client()

        for number in range(8):
            response = client.post("/api/chat", json={"message": str(number)})
            self.assertEqual(response.status_code, 200)

        limited = client.post("/api/chat", json={"message": "nona"})

        self.assertEqual(limited.status_code, 429)
        self.assertEqual(limited.json(), {"detail": RATE_LIMIT_DETAIL})
        self.assertEqual(limited.headers["retry-after"], "60")
        self.assertEqual(len(self.factory.sessions[0].messages), 8)

    def test_advancing_clock_past_window_permits_requests_again(self) -> None:
        client = self.make_client(per_session_limit=1)
        self.assertEqual(
            client.post("/api/chat", json={"message": "uma"}).status_code,
            200,
        )
        self.assertEqual(
            client.post("/api/chat", json={"message": "duas"}).status_code,
            429,
        )

        self.clock.advance(61)
        response = client.post("/api/chat", json={"message": "três"})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.factory.sessions[0].messages, ["uma", "três"])

    def test_browser_sessions_have_independent_quotas(self) -> None:
        shared_app = self.make_client(per_session_limit=1).app
        first = TestClient(shared_app)
        second = TestClient(shared_app)

        self.assertEqual(
            first.post("/api/chat", json={"message": "A"}).status_code,
            200,
        )
        self.assertEqual(
            first.post("/api/chat", json={"message": "A2"}).status_code,
            429,
        )
        self.assertEqual(
            second.post("/api/chat", json={"message": "B"}).status_code,
            200,
        )
        self.assertEqual(
            [session.messages for session in self.factory.sessions],
            [["A"], ["B"]],
        )

    def test_process_quota_applies_across_browser_sessions(self) -> None:
        shared_app = self.make_client(per_session_limit=10, global_limit=3).app
        clients = [TestClient(shared_app) for _ in range(4)]

        for number, client in enumerate(clients[:3]):
            response = client.post("/api/chat", json={"message": str(number)})
            self.assertEqual(response.status_code, 200)

        limited = clients[3].post("/api/chat", json={"message": "bloqueada"})

        self.assertEqual(limited.status_code, 429)
        self.assertEqual(limited.headers["retry-after"], "60")
        self.assertEqual(
            sum(len(session.messages) for session in self.factory.sessions),
            3,
        )

    def test_invalid_messages_do_not_consume_quota(self) -> None:
        client = self.make_client(per_session_limit=1, global_limit=1)

        blank = client.post("/api/chat", json={"message": "   "})
        too_long = client.post(
            "/api/chat",
            json={"message": "a" * (DEFAULT_MAX_MESSAGE_LENGTH + 1)},
        )
        accepted = client.post("/api/chat", json={"message": "válida"})
        limited = client.post("/api/chat", json={"message": "segunda"})

        self.assertEqual(blank.status_code, 400)
        self.assertEqual(too_long.status_code, 413)
        self.assertEqual(accepted.status_code, 200)
        self.assertEqual(limited.status_code, 429)
        self.assertEqual(self.factory.sessions[0].messages, ["válida"])

    def test_health_and_reset_do_not_consume_chat_quota(self) -> None:
        client = self.make_client(per_session_limit=1, global_limit=1)

        self.assertEqual(client.get("/").status_code, 200)
        self.assertEqual(client.get("/health").status_code, 200)
        self.assertEqual(client.post("/api/reset").status_code, 200)
        accepted = client.post("/api/chat", json={"message": "válida"})
        limited = client.post("/api/chat", json={"message": "segunda"})

        self.assertEqual(accepted.status_code, 200)
        self.assertEqual(limited.status_code, 429)
        self.assertEqual(self.factory.sessions[0].messages, ["válida"])

    def test_accepted_failed_request_consumes_quota(self) -> None:
        client = self.make_client(per_session_limit=1, global_limit=1)
        client.post("/api/reset")
        self.factory.sessions[0].failure = OpenAIError("private API details")

        failed = client.post("/api/chat", json={"message": "primeira"})
        limited = client.post("/api/chat", json={"message": "segunda"})

        self.assertEqual(failed.status_code, 200)
        self.assertEqual(failed.json(), {"message": SAFE_FAILURE_MESSAGE})
        self.assertEqual(limited.status_code, 429)
        self.assertEqual(self.factory.sessions[0].messages, ["primeira"])


if __name__ == "__main__":
    unittest.main()
