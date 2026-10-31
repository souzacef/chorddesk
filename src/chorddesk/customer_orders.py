"""Deterministic customer/order data and ownership verification."""

from __future__ import annotations

import csv
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal, InvalidOperation
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, TypeVar


class CustomerOrderDataError(ValueError):
    """Raised when customer or order source data is invalid."""


class OrderStatus(StrEnum):
    DELIVERED = "delivered"
    CONFIRMED = "confirmed"
    PENDING = "pending"
    CANCELLED = "cancelled"
    SHIPPED = "shipped"


class ResolutionStatus(StrEnum):
    NOT_FOUND = "not_found"
    UNIQUE = "unique"
    AMBIGUOUS = "ambiguous"


@dataclass(frozen=True, slots=True)
class Customer:
    customer_id: int
    name: str
    phone: str
    email: str
    city: str


@dataclass(frozen=True, slots=True)
class Order:
    order_id: int
    customer_id: int
    order_date: date
    status: OrderStatus
    total_brl: Decimal
    payment_method: str
    tracking_code: str | None
    estimated_delivery: date | None
    notes: str | None

    def __post_init__(self) -> None:
        if not self.total_brl.is_finite() or self.total_brl < 0:
            raise ValueError("total_brl must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class OrderItem:
    order_id: int
    quantity: int
    product_id: int

    def __post_init__(self) -> None:
        if self.quantity <= 0:
            raise ValueError("quantity must be a positive integer")


@dataclass(frozen=True, slots=True)
class CustomerResolution:
    status: ResolutionStatus
    customer: Customer | None = None

    def __post_init__(self) -> None:
        if (self.customer is not None) != (self.status == ResolutionStatus.UNIQUE):
            raise ValueError("only a unique resolution may contain a customer")


@dataclass(frozen=True, slots=True)
class VerifiedOrder:
    """Details released only after deterministic ownership verification."""

    order: Order
    items: tuple[OrderItem, ...]


@dataclass(frozen=True, slots=True)
class CustomerOrderData:
    customers: tuple[Customer, ...]
    orders: tuple[Order, ...]
    order_items: tuple[OrderItem, ...]
    _customers_by_id: Mapping[int, Customer] = field(init=False, repr=False)
    _customers_by_phone: Mapping[str, tuple[Customer, ...]] = field(
        init=False, repr=False
    )
    _customers_by_email: Mapping[str, tuple[Customer, ...]] = field(
        init=False, repr=False
    )
    _orders_by_id: Mapping[int, Order] = field(init=False, repr=False)
    _orders_by_customer: Mapping[int, tuple[Order, ...]] = field(
        init=False, repr=False
    )
    _items_by_order: Mapping[int, tuple[OrderItem, ...]] = field(
        init=False, repr=False
    )

    def __post_init__(self) -> None:
        customers = {item.customer_id: item for item in self.customers}
        orders = {item.order_id: item for item in self.orders}
        if len(customers) != len(self.customers):
            raise CustomerOrderDataError("duplicate customer_id in customer data")
        if len(orders) != len(self.orders):
            raise CustomerOrderDataError("duplicate order_id in order data")
        unknown_customers = {
            item.customer_id
            for item in self.orders
            if item.customer_id not in customers
        }
        unknown_orders = {
            item.order_id for item in self.order_items if item.order_id not in orders
        }
        if unknown_customers:
            values = ", ".join(map(str, sorted(unknown_customers)))
            raise CustomerOrderDataError(
                f"orders reference unknown customer_id(s): {values}"
            )
        if unknown_orders:
            values = ", ".join(map(str, sorted(unknown_orders)))
            raise CustomerOrderDataError(
                f"order items reference unknown order_id(s): {values}"
            )

        object.__setattr__(self, "_customers_by_id", MappingProxyType(customers))
        object.__setattr__(
            self,
            "_customers_by_phone",
            MappingProxyType(_group(self.customers, lambda item: item.phone)),
        )
        object.__setattr__(
            self,
            "_customers_by_email",
            MappingProxyType(
                _group(self.customers, lambda item: _normalize_email(item.email))
            ),
        )
        object.__setattr__(self, "_orders_by_id", MappingProxyType(orders))
        object.__setattr__(
            self,
            "_orders_by_customer",
            MappingProxyType(_group(self.orders, lambda item: item.customer_id)),
        )
        object.__setattr__(
            self,
            "_items_by_order",
            MappingProxyType(_group(self.order_items, lambda item: item.order_id)),
        )

    @classmethod
    def from_csv(
        cls, customers_path: Path, orders_path: Path, items_path: Path
    ) -> CustomerOrderData:
        return cls(
            load_customers(customers_path),
            load_orders(orders_path),
            load_order_items(items_path),
        )

    def get_customer(self, customer_id: int) -> Customer | None:
        return self._customers_by_id.get(customer_id)

    def resolve_phone(self, phone: str) -> CustomerResolution:
        return _resolve(self._customers_by_phone.get(phone.strip(), ()))

    def resolve_email(self, email: str) -> CustomerResolution:
        return _resolve(self._customers_by_email.get(_normalize_email(email), ()))

    def get_order(self, order_id: int) -> Order | None:
        """Look up an order for trusted internal data-layer use."""
        return self._orders_by_id.get(order_id)

    def orders_for_customer(self, customer_id: int) -> tuple[Order, ...]:
        return self._orders_by_customer.get(customer_id, ())

    def items_for_order(self, order_id: int) -> tuple[OrderItem, ...]:
        return self._items_by_order.get(order_id, ())


@dataclass(frozen=True, slots=True)
class OrderOwnershipVerifier:
    """Safe boundary for customer-facing access to order details."""

    data: CustomerOrderData

    def verify_by_phone(self, order_id: int, phone: str) -> VerifiedOrder | None:
        order = self.data.get_order(order_id)
        if order is None:
            return None
        return self._verify(order, self.data.resolve_phone(phone))

    def verify_by_email(self, order_id: int, email: str) -> VerifiedOrder | None:
        order = self.data.get_order(order_id)
        if order is None:
            return None
        return self._verify(order, self.data.resolve_email(email))

    def _verify(
        self, order: Order, resolution: CustomerResolution
    ) -> VerifiedOrder | None:
        if resolution.status != ResolutionStatus.UNIQUE:
            return None
        customer = resolution.customer
        if customer is None or customer.customer_id != order.customer_id:
            return None
        return VerifiedOrder(order, self.data.items_for_order(order.order_id))


_CUSTOMER_FIELDS = frozenset({"customer_id", "name", "phone", "email", "city"})
_ORDER_FIELDS = frozenset(
    {
        "order_id",
        "customer_id",
        "order_date",
        "status",
        "total_brl",
        "payment_method",
        "tracking_code",
        "estimated_delivery",
        "notes",
    }
)
_ITEM_FIELDS = frozenset({"order_id", "quantity", "product_id"})


def load_customers(path: Path) -> tuple[Customer, ...]:
    result = tuple(
        Customer(
            _integer(row, "customer_id", path, number),
            _required(row, "name", path, number),
            _required(row, "phone", path, number),
            _required(row, "email", path, number),
            _required(row, "city", path, number),
        )
        for number, row in _rows(path, _CUSTOMER_FIELDS)
    )
    _unique((item.customer_id for item in result), "customer_id", path)
    return result


def load_orders(path: Path) -> tuple[Order, ...]:
    result = tuple(
        _order(row, path, number) for number, row in _rows(path, _ORDER_FIELDS)
    )
    _unique((item.order_id for item in result), "order_id", path)
    return result


def load_order_items(path: Path) -> tuple[OrderItem, ...]:
    return tuple(
        _item(row, path, number) for number, row in _rows(path, _ITEM_FIELDS)
    )


def _order(row: dict[str, str], path: Path, number: int) -> Order:
    raw_status = _required(row, "status", path, number)
    try:
        status = OrderStatus(raw_status)
    except ValueError as exc:
        raise _error(path, number, "status", "unknown order status") from exc
    total = _decimal(row, "total_brl", path, number)
    if total < 0:
        raise _error(path, number, "total_brl", "cannot be negative")
    return Order(
        order_id=_integer(row, "order_id", path, number),
        customer_id=_integer(row, "customer_id", path, number),
        order_date=_date(row, "order_date", path, number),
        status=status,
        total_brl=total,
        payment_method=_required(row, "payment_method", path, number),
        tracking_code=_optional(row, "tracking_code"),
        estimated_delivery=_optional_date(
            row, "estimated_delivery", path, number
        ),
        notes=_optional(row, "notes"),
    )


def _item(row: dict[str, str], path: Path, number: int) -> OrderItem:
    order_id = _integer(row, "order_id", path, number)
    quantity = _integer(row, "quantity", path, number)
    product_id = _integer(row, "product_id", path, number)
    if quantity <= 0:
        raise _error(path, number, "quantity", "must be a positive integer")
    return OrderItem(order_id, quantity, product_id)


def _rows(
    path: Path, fields: frozenset[str]
) -> tuple[tuple[int, dict[str, str]], ...]:
    try:
        with path.open(encoding="utf-8", newline="") as source:
            reader = csv.DictReader(source)
            if reader.fieldnames is None:
                raise CustomerOrderDataError(f"{path}: missing CSV header")
            missing = fields.difference(reader.fieldnames)
            if missing:
                names = ", ".join(sorted(missing))
                raise CustomerOrderDataError(
                    f"{path}: missing required column(s): {names}"
                )
            return tuple(enumerate(reader, start=2))
    except UnicodeDecodeError as exc:
        raise CustomerOrderDataError(f"{path}: file is not valid UTF-8") from exc


def _required(row: dict[str, str], field_name: str, path: Path, number: int) -> str:
    value = row.get(field_name)
    if value is None or not value.strip():
        raise _error(path, number, field_name, "value is required")
    return value.strip()


def _optional(row: dict[str, str], field_name: str) -> str | None:
    value = row.get(field_name)
    return value.strip() or None if value is not None else None


def _integer(row: dict[str, str], field_name: str, path: Path, number: int) -> int:
    try:
        return int(_required(row, field_name, path, number))
    except ValueError as exc:
        raise _error(path, number, field_name, "expected an integer") from exc


def _decimal(row: dict[str, str], field_name: str, path: Path, number: int) -> Decimal:
    try:
        value = Decimal(_required(row, field_name, path, number))
    except InvalidOperation as exc:
        raise _error(path, number, field_name, "expected a decimal") from exc
    if not value.is_finite():
        raise _error(path, number, field_name, "expected a finite decimal")
    return value


def _date(row: dict[str, str], field_name: str, path: Path, number: int) -> date:
    try:
        return date.fromisoformat(_required(row, field_name, path, number))
    except ValueError as exc:
        raise _error(path, number, field_name, "expected an ISO date") from exc


def _optional_date(
    row: dict[str, str], field_name: str, path: Path, number: int
) -> date | None:
    value = _optional(row, field_name)
    if value is None:
        return None
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise _error(path, number, field_name, "expected an ISO date") from exc


def _unique(ids: Iterable[int], field_name: str, path: Path) -> None:
    seen: set[int] = set()
    for identifier in ids:
        if identifier in seen:
            raise CustomerOrderDataError(
                f"{path}: duplicate {field_name}: {identifier}"
            )
        seen.add(identifier)


def _normalize_email(email: str) -> str:
    return email.strip().casefold()


def _resolve(customers: tuple[Customer, ...]) -> CustomerResolution:
    if not customers:
        return CustomerResolution(ResolutionStatus.NOT_FOUND)
    if len(customers) > 1:
        return CustomerResolution(ResolutionStatus.AMBIGUOUS)
    return CustomerResolution(ResolutionStatus.UNIQUE, customers[0])


_Item = TypeVar("_Item")
_Key = TypeVar("_Key")


def _group(
    items: tuple[_Item, ...], key: Callable[[_Item], _Key]
) -> dict[_Key, tuple[_Item, ...]]:
    groups: dict[_Key, list[_Item]] = {}
    for item in items:
        groups.setdefault(key(item), []).append(item)
    return {value: tuple(matches) for value, matches in groups.items()}


def _error(
    path: Path, number: int, field_name: str, message: str
) -> CustomerOrderDataError:
    return CustomerOrderDataError(
        f"{path}: row {number}, field {field_name!r}: {message}"
    )
