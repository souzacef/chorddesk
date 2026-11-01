from __future__ import annotations

import unittest
from dataclasses import FrozenInstanceError, fields
from datetime import date
from decimal import Decimal
from pathlib import Path

from chorddesk.customer_orders import (
    Customer,
    CustomerOrderData,
    Order,
    OrderItem,
    OrderOwnershipVerifier,
    OrderStatus,
)
from chorddesk.customer_service import (
    OrderStatusResult,
    OrderStatusService,
    PolicyService,
)
from chorddesk.policy import PolicySection, parse_policy_pdf
from chorddesk.policy_search import PolicyRetriever


PROJECT_ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = PROJECT_ROOT / "data" / "políticas_da_loja.pdf"


class PolicyServiceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.sections = parse_policy_pdf(POLICY_PATH)
        cls.service = PolicyService(PolicyRetriever(cls.sections))

    def test_store_hours_section_is_near_top(self) -> None:
        results = self.service.search_store_policies("qual o horário da loja?")
        self.assertIn("2", (result.section_number for result in results))

    def test_online_return_section_is_near_top(self) -> None:
        results = self.service.search_store_policies(
            "posso devolver uma compra online depois de 5 dias?"
        )
        self.assertIn("4.1", (result.section_number for result in results))

    def test_policy_text_and_page_metadata_are_preserved(self) -> None:
        source = next(section for section in self.sections if section.number == "4.1")
        result = next(
            result
            for result in self.service.search_store_policies(
                "devolver compra online", limit=3
            )
            if result.section_number == "4.1"
        )
        self.assertEqual(result.content, source.content)
        self.assertEqual(result.title, source.title)
        self.assertEqual(
            (result.page_start, result.page_end),
            (source.page_start, source.page_end),
        )

    def test_ordering_limit_and_scores_match_retriever(self) -> None:
        sections = (
            PolicySection("9", "Alpha Beta", "", 4, 4),
            PolicySection("2", "Alpha", "", 5, 6),
            PolicySection("1", "Other", "alpha", 7, 7),
        )
        retriever = PolicyRetriever(sections)
        expected = retriever.search("alpha beta", limit=2)
        actual = PolicyService(retriever).search_store_policies(
            "alpha beta", limit=2
        )
        self.assertEqual(
            tuple((item.section_number, item.relevance_score) for item in actual),
            tuple((item.section.number, item.score) for item in expected),
        )

    def test_result_is_immutable(self) -> None:
        result = self.service.search_store_policies("horário", limit=1)[0]
        with self.assertRaises(FrozenInstanceError):
            result.title = "Changed"  # type: ignore[misc]


class OrderStatusServiceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        data = CustomerOrderData(
            customers=(
                Customer(1, "Alpha", "phone-a", "shared@test", "City"),
                Customer(2, "Beta", "phone-b", "shared@test", "City"),
                Customer(3, "Gamma", "phone-c", "unique@test", "City"),
            ),
            orders=(
                Order(
                    101,
                    1,
                    date(2026, 1, 1),
                    OrderStatus.SHIPPED,
                    Decimal("100.00"),
                    "pix",
                    "TRACK-101",
                    date(2026, 1, 8),
                    "Leave at reception",
                ),
                Order(
                    103,
                    3,
                    date(2026, 1, 2),
                    OrderStatus.PENDING,
                    Decimal("50.25"),
                    "debit",
                    None,
                    None,
                    None,
                ),
            ),
            order_items=(
                OrderItem(101, 1, 900),
                OrderItem(101, 2, 901),
                OrderItem(103, 3, 902),
            ),
        )
        cls.service = OrderStatusService(OrderOwnershipVerifier(data))

    def test_successful_phone_verification_returns_safe_facts_and_items(self) -> None:
        result = self.service.get_order_status(101, phone="phone-a")
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.order_id, 101)
        self.assertEqual(result.status, OrderStatus.SHIPPED)
        self.assertEqual(result.order_date, date(2026, 1, 1))
        self.assertEqual(result.payment_method, "pix")
        self.assertEqual(result.notes, "Leave at reception")
        self.assertEqual(
            tuple((item.product_id, item.quantity) for item in result.items),
            ((900, 1), (901, 2)),
        )

    def test_successful_unique_email_verification(self) -> None:
        result = self.service.get_order_status(103, email="  UNIQUE@TEST ")
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.order_id, 103)
        self.assertEqual(
            tuple((item.product_id, item.quantity) for item in result.items),
            ((902, 3),),
        )

    def test_all_verification_failures_have_one_generic_outcome(self) -> None:
        failures = (
            self.service.get_order_status(999, phone="phone-a"),
            self.service.get_order_status(101, phone="phone-c"),
            self.service.get_order_status(101, phone="unknown"),
            self.service.get_order_status(101, email="unknown@test"),
            self.service.get_order_status(101, email="shared@test"),
        )
        self.assertEqual(failures, (None,) * len(failures))

    def test_neither_identifier_is_invalid(self) -> None:
        with self.assertRaisesRegex(ValueError, "exactly one"):
            self.service.get_order_status(101)

    def test_both_identifiers_are_invalid(self) -> None:
        with self.assertRaisesRegex(ValueError, "exactly one"):
            self.service.get_order_status(101, phone="phone-a", email="shared@test")

    def test_result_has_no_customer_pii_fields(self) -> None:
        names = {field.name for field in fields(OrderStatusResult)}
        self.assertTrue(
            names.isdisjoint(
                {"customer", "customer_id", "name", "phone", "email", "city"}
            )
        )

    def test_optional_delivery_values_are_preserved(self) -> None:
        present = self.service.get_order_status(101, phone="phone-a")
        absent = self.service.get_order_status(103, email="unique@test")
        assert present is not None and absent is not None
        self.assertEqual(present.tracking_code, "TRACK-101")
        self.assertEqual(present.estimated_delivery, date(2026, 1, 8))
        self.assertIsNone(absent.tracking_code)
        self.assertIsNone(absent.estimated_delivery)
        self.assertIsNone(absent.notes)

    def test_decimal_total_is_preserved(self) -> None:
        result = self.service.get_order_status(103, email="unique@test")
        assert result is not None
        self.assertEqual(result.total_brl, Decimal("50.25"))
        self.assertIsInstance(result.total_brl, Decimal)

    def test_items_are_not_released_when_verification_fails(self) -> None:
        self.assertIsNone(self.service.get_order_status(101, phone="phone-c"))

    def test_results_and_items_are_immutable(self) -> None:
        result = self.service.get_order_status(101, phone="phone-a")
        assert result is not None
        with self.assertRaises(FrozenInstanceError):
            result.status = OrderStatus.DELIVERED  # type: ignore[misc]
        with self.assertRaises(FrozenInstanceError):
            result.items[0].quantity = 99  # type: ignore[misc]


if __name__ == "__main__":
    unittest.main()
