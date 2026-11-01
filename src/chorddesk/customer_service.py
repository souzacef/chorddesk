"""Customer-safe application services for policies and order status."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import Decimal

from chorddesk.customer_orders import OrderOwnershipVerifier, OrderStatus, VerifiedOrder
from chorddesk.policy_search import PolicyRetriever


@dataclass(frozen=True, slots=True)
class PolicyResult:
    """Grounded policy content safe to expose to a customer-facing caller."""

    section_number: str
    title: str
    content: str
    page_start: int
    page_end: int
    relevance_score: int


@dataclass(frozen=True, slots=True)
class OrderItemResult:
    """Minimal customer-safe facts for an item in a verified order."""

    product_id: int
    quantity: int


@dataclass(frozen=True, slots=True)
class OrderStatusResult:
    """Customer-safe order facts released only after ownership verification."""

    order_id: int
    status: OrderStatus
    order_date: date
    total_brl: Decimal
    payment_method: str
    tracking_code: str | None
    estimated_delivery: date | None
    notes: str | None
    items: tuple[OrderItemResult, ...]


class PolicyService:
    """Application boundary over an already-constructed policy index."""

    __slots__ = ("_retriever",)

    def __init__(self, retriever: PolicyRetriever) -> None:
        self._retriever = retriever

    def search_store_policies(
        self, query: str, *, limit: int = 3
    ) -> tuple[PolicyResult, ...]:
        """Return unaltered policy sections in retriever-defined rank order."""
        return tuple(
            PolicyResult(
                section_number=result.section.number,
                title=result.section.title,
                content=result.section.content,
                page_start=result.section.page_start,
                page_end=result.section.page_end,
                relevance_score=result.score,
            )
            for result in self._retriever.search(query, limit=limit)
        )


class OrderStatusService:
    """Application boundary for status of deterministically verified orders."""

    __slots__ = ("_verifier",)

    def __init__(self, verifier: OrderOwnershipVerifier) -> None:
        self._verifier = verifier

    def get_order_status(
        self,
        order_id: int,
        *,
        phone: str | None = None,
        email: str | None = None,
    ) -> OrderStatusResult | None:
        """Return safe order facts, or one generic result when not authorized."""
        if (phone is None) == (email is None):
            raise ValueError("exactly one of phone or email must be supplied")

        verified = (
            self._verifier.verify_by_phone(order_id, phone)
            if phone is not None
            else self._verifier.verify_by_email(order_id, email)
        )
        if verified is None:
            return None
        return self._result(verified)

    @staticmethod
    def _result(verified: VerifiedOrder) -> OrderStatusResult:
        order = verified.order
        return OrderStatusResult(
            order_id=order.order_id,
            status=order.status,
            order_date=order.order_date,
            total_brl=order.total_brl,
            payment_method=order.payment_method,
            tracking_code=order.tracking_code,
            estimated_delivery=order.estimated_delivery,
            notes=order.notes,
            items=tuple(
                OrderItemResult(product_id=item.product_id, quantity=item.quantity)
                for item in verified.items
            ),
        )
