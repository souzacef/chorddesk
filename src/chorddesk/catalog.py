"""Deterministic access to ChordDesk's category and product catalog."""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal, InvalidOperation
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import Mapping


class CatalogDataError(ValueError):
    """Raised when catalog source data is structurally invalid."""


class ProductStatus(StrEnum):
    ACTIVE = "active"
    DISCONTINUED = "discontinued"
    COMING_SOON = "coming_soon"


@dataclass(frozen=True, slots=True)
class Category:
    category_id: int
    name: str
    description: str


@dataclass(frozen=True, slots=True)
class Product:
    product_id: int
    price_brl: Decimal
    name: str
    category_id: int
    description: str
    stock_quantity: int
    status: ProductStatus
    specs: Mapping[str, str]
    created_at: date

    def __post_init__(self) -> None:
        if self.price_brl < 0:
            raise ValueError("price_brl cannot be negative")
        if self.stock_quantity < 0:
            raise ValueError("stock_quantity cannot be negative")
        object.__setattr__(self, "specs", MappingProxyType(dict(self.specs)))

    @property
    def is_available_for_sale(self) -> bool:
        """Whether this product may currently be sold."""
        return self.status == ProductStatus.ACTIVE and self.stock_quantity > 0


@dataclass(frozen=True, slots=True)
class Catalog:
    categories: tuple[Category, ...]
    products: tuple[Product, ...]
    _products_by_id: Mapping[int, Product] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        category_ids = {category.category_id for category in self.categories}
        if len(category_ids) != len(self.categories):
            raise CatalogDataError("duplicate category_id in catalog")

        products_by_id = {product.product_id: product for product in self.products}
        if len(products_by_id) != len(self.products):
            raise CatalogDataError("duplicate product_id in catalog")

        unknown_category_ids = {
            product.category_id
            for product in self.products
            if product.category_id not in category_ids
        }
        if unknown_category_ids:
            unknown = ", ".join(map(str, sorted(unknown_category_ids)))
            raise CatalogDataError(f"products reference unknown category_id(s): {unknown}")

        object.__setattr__(
            self, "_products_by_id", MappingProxyType(products_by_id)
        )

    @classmethod
    def from_csv(cls, categories_path: Path, products_path: Path) -> Catalog:
        return cls(
            categories=load_categories(categories_path),
            products=load_products(products_path),
        )

    def get_product(self, product_id: int) -> Product | None:
        return self._products_by_id.get(product_id)

    def search_products(
        self,
        *,
        query: str | None = None,
        category_id: int | None = None,
        min_price: Decimal | None = None,
        max_price: Decimal | None = None,
        available_only: bool = False,
    ) -> tuple[Product, ...]:
        if min_price is not None and max_price is not None and min_price > max_price:
            raise ValueError("min_price cannot be greater than max_price")

        normalized_query = query.strip().casefold() if query else None
        return tuple(
            product
            for product in self.products
            if (
                normalized_query is None
                or normalized_query in product.name.casefold()
                or normalized_query in product.description.casefold()
            )
            and (category_id is None or product.category_id == category_id)
            and (min_price is None or product.price_brl >= min_price)
            and (max_price is None or product.price_brl <= max_price)
            and (not available_only or product.is_available_for_sale)
        )


_CATEGORY_FIELDS = frozenset({"category_id", "name", "description"})
_PRODUCT_FIELDS = frozenset(
    {
        "product_id",
        "price_brl",
        "name",
        "category_id",
        "description",
        "stock_quantity",
        "status",
        "specs",
        "created_at",
    }
)


def load_categories(path: Path) -> tuple[Category, ...]:
    rows = _read_rows(path, _CATEGORY_FIELDS)
    categories = tuple(
        Category(
            category_id=_parse_int(row, "category_id", path, row_number),
            name=_required_text(row, "name", path, row_number),
            description=_required_text(row, "description", path, row_number),
        )
        for row_number, row in rows
    )
    _ensure_unique_ids(
        (category.category_id for category in categories), "category_id", path
    )
    return categories


def load_products(path: Path) -> tuple[Product, ...]:
    rows = _read_rows(path, _PRODUCT_FIELDS)
    products = tuple(_parse_product(row, path, row_number) for row_number, row in rows)
    _ensure_unique_ids((product.product_id for product in products), "product_id", path)
    return products


def _read_rows(
    path: Path, required_fields: frozenset[str]
) -> tuple[tuple[int, dict[str, str]], ...]:
    try:
        with path.open(encoding="utf-8", newline="") as source:
            reader = csv.DictReader(source)
            if reader.fieldnames is None:
                raise CatalogDataError(f"{path}: missing CSV header")
            missing_fields = required_fields.difference(reader.fieldnames)
            if missing_fields:
                missing = ", ".join(sorted(missing_fields))
                raise CatalogDataError(f"{path}: missing required column(s): {missing}")
            return tuple((number, row) for number, row in enumerate(reader, start=2))
    except UnicodeDecodeError as exc:
        raise CatalogDataError(f"{path}: file is not valid UTF-8") from exc


def _parse_product(row: dict[str, str], path: Path, row_number: int) -> Product:
    price = _parse_decimal(row, "price_brl", path, row_number)
    stock_quantity = _parse_int(row, "stock_quantity", path, row_number)
    try:
        status = ProductStatus(_required_text(row, "status", path, row_number))
    except ValueError as exc:
        raise _field_error(path, row_number, "status", "unknown product status") from exc

    raw_specs = _required_text(row, "specs", path, row_number)
    try:
        specs = json.loads(raw_specs)
    except json.JSONDecodeError as exc:
        raise _field_error(path, row_number, "specs", "invalid JSON") from exc
    if not isinstance(specs, dict) or not all(
        isinstance(key, str) and isinstance(value, str)
        for key, value in specs.items()
    ):
        raise _field_error(path, row_number, "specs", "expected a JSON object of strings")

    raw_created_at = _required_text(row, "created_at", path, row_number)
    try:
        created_at = date.fromisoformat(raw_created_at)
    except ValueError as exc:
        raise _field_error(path, row_number, "created_at", "expected an ISO date") from exc

    try:
        return Product(
            product_id=_parse_int(row, "product_id", path, row_number),
            price_brl=price,
            name=_required_text(row, "name", path, row_number),
            category_id=_parse_int(row, "category_id", path, row_number),
            description=_required_text(row, "description", path, row_number),
            stock_quantity=stock_quantity,
            status=status,
            specs=specs,
            created_at=created_at,
        )
    except ValueError as exc:
        raise CatalogDataError(f"{path}: row {row_number}: {exc}") from exc


def _required_text(
    row: dict[str, str], field_name: str, path: Path, row_number: int
) -> str:
    value = row.get(field_name)
    if value is None or not value.strip():
        raise _field_error(path, row_number, field_name, "value is required")
    return value.strip()


def _parse_int(
    row: dict[str, str], field_name: str, path: Path, row_number: int
) -> int:
    value = _required_text(row, field_name, path, row_number)
    try:
        return int(value)
    except ValueError as exc:
        raise _field_error(path, row_number, field_name, "expected an integer") from exc


def _parse_decimal(
    row: dict[str, str], field_name: str, path: Path, row_number: int
) -> Decimal:
    value = _required_text(row, field_name, path, row_number)
    try:
        number = Decimal(value)
    except InvalidOperation as exc:
        raise _field_error(path, row_number, field_name, "expected a decimal") from exc
    if not number.is_finite():
        raise _field_error(path, row_number, field_name, "expected a finite decimal")
    return number


def _ensure_unique_ids(ids: Iterable[int], field_name: str, path: Path) -> None:
    seen: set[int] = set()
    for identifier in ids:
        if identifier in seen:
            raise CatalogDataError(f"{path}: duplicate {field_name}: {identifier}")
        seen.add(identifier)


def _field_error(
    path: Path, row_number: int, field_name: str, message: str
) -> CatalogDataError:
    return CatalogDataError(f"{path}: row {row_number}, field {field_name!r}: {message}")
