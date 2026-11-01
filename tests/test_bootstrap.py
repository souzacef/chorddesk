from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from chorddesk.bootstrap import MissingDataFileError, build_session
from chorddesk.customer_service import OrderStatusService, PolicyService
from chorddesk.openai_agent import ChordDeskAgent, ConversationSession, ToolDispatcher
from chorddesk.storefront import StorefrontService


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"


class BootstrapTests(unittest.TestCase):
    def test_default_data_directory_resolves_repository_data(self) -> None:
        session = build_session(client=Mock())

        self.assertIsInstance(session, ConversationSession)

    def test_builds_complete_existing_dependency_graph_without_api_request(self) -> None:
        client = Mock()

        session = build_session(DATA_DIR, client=client)

        self.assertIsInstance(session, ConversationSession)
        self.assertIsInstance(session.agent, ChordDeskAgent)
        self.assertIs(session.agent.client, client)
        dispatcher = session.agent.dispatcher
        self.assertIsInstance(dispatcher, ToolDispatcher)
        self.assertIsInstance(dispatcher.storefront, StorefrontService)
        self.assertIsInstance(dispatcher.policies, PolicyService)
        self.assertIsInstance(dispatcher.orders, OrderStatusService)
        client.responses.create.assert_not_called()

    def test_accepts_explicit_data_directory(self) -> None:
        session = build_session(Path(DATA_DIR), client=Mock())

        self.assertIsInstance(session, ConversationSession)

    def test_missing_required_file_fails_clearly(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(
                MissingDataFileError, "required data file.*categories.csv"
            ):
                build_session(Path(directory), client=Mock())


if __name__ == "__main__":
    unittest.main()
