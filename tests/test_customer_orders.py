from __future__ import annotations

import tempfile
import unittest
from datetime import date
from decimal import Decimal
from pathlib import Path

from chorddesk.customer_orders import (
    Customer,
    CustomerOrderData,
    CustomerOrderDataError,
    Order,
    OrderItem,
    OrderOwnershipVerifier,
    OrderStatus,
    ResolutionStatus,
    load_customers,
    load_order_items,
    load_orders,
)


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
CUSTOMERS = DATA / "desafio_tecnico_ai_eng - customers.csv"
ORDERS = DATA / "desafio_tecnico_ai_eng - orders.csv"
ITEMS = DATA / "desafio_tecnico_ai_eng - order_items.csv"


class LoadingTests(unittest.TestCase):
    def test_loads_all_supplied_rows(self) -> None:
        self.assertEqual(len(load_customers(CUSTOMERS)), 50)
        self.assertEqual(len(load_orders(ORDERS)), 20)
        self.assertEqual(len(load_order_items(ITEMS)), 22)

    def test_typed_and_optional_order_fields(self) -> None:
        shipped, confirmed = load_orders(ORDERS)[7:9]
        self.assertEqual(shipped.order_date, date(2026, 2, 15))
        self.assertEqual(shipped.status, OrderStatus.SHIPPED)
        self.assertEqual(shipped.total_brl, Decimal("349.9"))
        self.assertEqual(shipped.estimated_delivery, date(2026, 2, 28))
        self.assertIsNotNone(shipped.tracking_code)
        self.assertIsNone(confirmed.tracking_code)
        self.assertIsNone(confirmed.estimated_delivery)
        self.assertIsNone(confirmed.notes)

    def test_invalid_quantity_has_file_row_and_field_context(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "order_items.csv"
            path.write_text("order_id,quantity,product_id\n1,0,93\n", encoding="utf-8")
            with self.assertRaisesRegex(
                CustomerOrderDataError,
                r"order_items\.csv: row 2, field 'quantity'.*positive integer",
            ):
                load_order_items(path)

    def test_duplicate_order_product_rows_are_preserved_in_source_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "order_items.csv"
            path.write_text(
                "order_id,quantity,product_id\n"
                "1,1,93\n"
                "1,2,93\n",
                encoding="utf-8",
            )

            items = load_order_items(path)

        self.assertEqual(
            tuple((item.order_id, item.quantity, item.product_id) for item in items),
            ((1, 1, 93), (1, 2, 93)),
        )


class DataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.data = CustomerOrderData.from_csv(CUSTOMERS, ORDERS, ITEMS)

    def test_lookups_and_relationships(self) -> None:
        order = self.data.get_order(10)
        self.assertIsNotNone(order)
        assert order is not None
        self.assertEqual(order.customer_id, 11)
        self.assertIsNone(self.data.get_order(999))
        self.assertEqual(
            tuple(item.order_id for item in self.data.orders_for_customer(3)),
            (1, 19),
        )
        self.assertEqual(
            tuple(item.product_id for item in self.data.items_for_order(15)),
            (82, 106),
        )

    def test_email_resolution_is_case_insensitive(self) -> None:
        customer = self.data.get_customer(1)
        self.assertIsNotNone(customer)
        assert customer is not None

        result = self.data.resolve_email(f"  {customer.email.upper()} ")
        self.assertEqual(result.status, ResolutionStatus.UNIQUE)


class VerificationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        data = CustomerOrderData(
            customers=(
                Customer(1, "Alpha", "phone-alpha", "shared@example.test", "Test City"),
                Customer(2, "Beta", "phone-beta", "shared@example.test", "Test City"),
                Customer(3, "Gamma", "phone-gamma", "unique@example.test", "Test City"),
            ),
            orders=(
                Order(
                    101,
                    1,
                    date(2026, 1, 1),
                    OrderStatus.CONFIRMED,
                    Decimal("100.00"),
                    "pix",
                    None,
                    None,
                    None,
                ),
                Order(
                    103,
                    3,
                    date(2026, 1, 2),
                    OrderStatus.PENDING,
                    Decimal("50.00"),
                    "debit",
                    None,
                    None,
                    None,
                ),
            ),
            order_items=(OrderItem(101, 1, 900), OrderItem(103, 2, 901)),
        )
        cls.verifier = OrderOwnershipVerifier(data)

    def test_matching_phone_returns_details(self) -> None:
        result = self.verifier.verify_by_phone(101, "phone-alpha")
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.order.order_id, 101)
        self.assertEqual(tuple(item.product_id for item in result.items), (900,))

    def test_matching_unique_email_returns_details(self) -> None:
        self.assertIsNotNone(
            self.verifier.verify_by_email(103, "UNIQUE@EXAMPLE.TEST")
        )

    def test_wrong_customer_fails(self) -> None:
        self.assertIsNone(self.verifier.verify_by_phone(101, "phone-gamma"))

    def test_unknown_order_and_identifiers_fail(self) -> None:
        self.assertIsNone(self.verifier.verify_by_phone(999, "phone-alpha"))
        self.assertIsNone(self.verifier.verify_by_phone(101, "unknown"))
        self.assertIsNone(self.verifier.verify_by_email(101, "unknown@example.test"))

    def test_ambiguous_email_cannot_authorize_order(self) -> None:
        self.assertIsNone(
            self.verifier.verify_by_email(101, "shared@example.test")
        )


class SuppliedDuplicateEmailIntegrationTests(unittest.TestCase):
    def test_supplied_duplicate_email_cannot_authorize_order_10(self) -> None:
        data = CustomerOrderData.from_csv(CUSTOMERS, ORDERS, ITEMS)
        order = data.get_order(10)
        self.assertIsNotNone(order)
        assert order is not None
        customer = data.get_customer(order.customer_id)
        self.assertIsNotNone(customer)
        assert customer is not None

        resolution = data.resolve_email(customer.email)

        self.assertEqual(resolution.status, ResolutionStatus.AMBIGUOUS)
        self.assertIsNone(
            OrderOwnershipVerifier(data).verify_by_email(10, customer.email)
        )


if __name__ == "__main__":
    unittest.main()
