"""Application composition root for ChordDesk's repository data."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from openai import OpenAI

from chorddesk.catalog import Catalog
from chorddesk.customer_orders import CustomerOrderData, OrderOwnershipVerifier
from chorddesk.customer_service import OrderStatusService, PolicyService
from chorddesk.openai_agent import ChordDeskAgent, ConversationSession, ToolDispatcher
from chorddesk.policy import parse_policy_pdf
from chorddesk.policy_search import PolicyRetriever
from chorddesk.promotions import Promotions
from chorddesk.storefront import StorefrontService


CATEGORIES_FILE = "desafio_tecnico_ai_eng - categories.csv"
PRODUCTS_FILE = "desafio_tecnico_ai_eng - products.csv"
PROMOTIONS_FILE = "desafio_tecnico_ai_eng - promotions.csv"
CUSTOMERS_FILE = "desafio_tecnico_ai_eng - customers.csv"
ORDERS_FILE = "desafio_tecnico_ai_eng - orders.csv"
ORDER_ITEMS_FILE = "desafio_tecnico_ai_eng - order_items.csv"
POLICY_FILE = "políticas_da_loja.pdf"

REQUIRED_DATA_FILES = (
    CATEGORIES_FILE,
    PRODUCTS_FILE,
    PROMOTIONS_FILE,
    CUSTOMERS_FILE,
    ORDERS_FILE,
    ORDER_ITEMS_FILE,
    POLICY_FILE,
)


class MissingDataFileError(FileNotFoundError):
    """Raised when the application data directory is incomplete."""


def repository_data_dir() -> Path:
    """Return the data directory next to this repository's ``src`` directory."""
    return Path(__file__).resolve().parents[2] / "data"


def build_session(
    data_dir: Path | None = None,
    *,
    client: Any | None = None,
) -> ConversationSession:
    """Build the complete application graph without issuing an API request."""
    source_dir = Path(data_dir) if data_dir is not None else repository_data_dir()
    paths = _required_paths(source_dir)

    catalog = Catalog.from_csv(paths[CATEGORIES_FILE], paths[PRODUCTS_FILE])
    promotions = Promotions.from_csv(paths[PROMOTIONS_FILE])
    customer_orders = CustomerOrderData.from_csv(
        paths[CUSTOMERS_FILE], paths[ORDERS_FILE], paths[ORDER_ITEMS_FILE]
    )
    policy_retriever = PolicyRetriever(parse_policy_pdf(paths[POLICY_FILE]))

    storefront = StorefrontService(catalog, promotions)
    verifier = OrderOwnershipVerifier(customer_orders)
    policies = PolicyService(policy_retriever)
    orders = OrderStatusService(verifier)
    dispatcher = ToolDispatcher(storefront, policies, orders)
    agent = ChordDeskAgent(
        client=client if client is not None else OpenAI(),
        dispatcher=dispatcher,
    )
    return ConversationSession(agent)


def _required_paths(data_dir: Path) -> dict[str, Path]:
    if not data_dir.is_dir():
        raise MissingDataFileError(f"data directory does not exist: {data_dir}")

    paths = {name: data_dir / name for name in REQUIRED_DATA_FILES}
    missing = [path for path in paths.values() if not path.is_file()]
    if missing:
        names = ", ".join(path.name for path in missing)
        raise MissingDataFileError(f"required data file(s) missing: {names}")
    return paths
