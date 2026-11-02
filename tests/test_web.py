from __future__ import annotations

import unittest

from fastapi.testclient import TestClient
from openai import OpenAIError

from chorddesk.openai_agent import AgentError, ToolProtocolError
from chorddesk.web import COOKIE_NAME, SAFE_FAILURE_MESSAGE, create_app


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


if __name__ == "__main__":
    unittest.main()
