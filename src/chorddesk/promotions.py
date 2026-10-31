"""Deterministic access to ChordDesk's product promotions."""

from __future__ import annotations

import csv
from collections.abc import Iterable
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from pathlib import Path
from types import MappingProxyType
from typing import Mapping


class PromotionDataError(ValueError):
    """Raised when promotion source data is structurally invalid."""


@dataclass(frozen=True, slots=True)
class Promotion:
    promotion_id: int
    product_id: int
    discount_percent: Decimal
    description: str
    is_active: bool

    def __post_init__(self) -> None:
        if not self.discount_percent.is_finite():
            raise ValueError("discount_percent must be finite")
        if not Decimal("0") < self.discount_percent <= Decimal("100"):
            raise ValueError("discount_percent must be greater than 0 and at most 100")

    def discounted_price(self, base_price: Decimal) -> Decimal:
        """Apply this promotion's percentage to a supplied base price."""
        if not base_price.is_finite():
            raise ValueError("base_price must be finite")
        if base_price < 0:
            raise ValueError("base_price cannot be negative")
        return base_price * (Decimal("100") - self.discount_percent) / Decimal("100")


@dataclass(frozen=True, slots=True)
class Promotions:
    promotions: tuple[Promotion, ...]
    _promotions_by_id: Mapping[int, Promotion] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        promotions_by_id = {
            promotion.promotion_id: promotion for promotion in self.promotions
        }
        if len(promotions_by_id) != len(self.promotions):
            raise PromotionDataError("duplicate promotion_id in promotions")
        object.__setattr__(
            self, "_promotions_by_id", MappingProxyType(promotions_by_id)
        )

    @classmethod
    def from_csv(cls, path: Path) -> Promotions:
        return cls(promotions=load_promotions(path))

    def get_promotion(self, promotion_id: int) -> Promotion | None:
        return self._promotions_by_id.get(promotion_id)

    def for_product(self, product_id: int) -> tuple[Promotion, ...]:
        return tuple(
            promotion
            for promotion in self.promotions
            if promotion.product_id == product_id
        )

    def active_promotions(self) -> tuple[Promotion, ...]:
        return tuple(promotion for promotion in self.promotions if promotion.is_active)


_PROMOTION_FIELDS = frozenset(
    {
        "promotion_id",
        "product_id",
        "discount_percent",
        "description",
        "is_active",
    }
)


def load_promotions(path: Path) -> tuple[Promotion, ...]:
    rows = _read_rows(path)
    promotions = tuple(
        _parse_promotion(row, path, row_number) for row_number, row in rows
    )
    _ensure_unique_ids(
        (promotion.promotion_id for promotion in promotions), path
    )
    return promotions


def _read_rows(path: Path) -> tuple[tuple[int, dict[str, str]], ...]:
    try:
        with path.open(encoding="utf-8", newline="") as source:
            reader = csv.DictReader(source)
            if reader.fieldnames is None:
                raise PromotionDataError(f"{path}: missing CSV header")
            missing_fields = _PROMOTION_FIELDS.difference(reader.fieldnames)
            if missing_fields:
                missing = ", ".join(sorted(missing_fields))
                raise PromotionDataError(
                    f"{path}: missing required column(s): {missing}"
                )
            return tuple((number, row) for number, row in enumerate(reader, start=2))
    except UnicodeDecodeError as exc:
        raise PromotionDataError(f"{path}: file is not valid UTF-8") from exc


def _parse_promotion(
    row: dict[str, str], path: Path, row_number: int
) -> Promotion:
    return Promotion(
        promotion_id=_parse_int(row, "promotion_id", path, row_number),
        product_id=_parse_int(row, "product_id", path, row_number),
        discount_percent=_parse_discount(row, path, row_number),
        description=_required_text(row, "description", path, row_number),
        is_active=_parse_bool(row, path, row_number),
    )


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


def _parse_discount(
    row: dict[str, str], path: Path, row_number: int
) -> Decimal:
    value = _required_text(row, "discount_percent", path, row_number)
    try:
        discount = Decimal(value)
    except InvalidOperation as exc:
        raise _field_error(
            path, row_number, "discount_percent", "expected a decimal"
        ) from exc
    if not discount.is_finite():
        raise _field_error(
            path, row_number, "discount_percent", "expected a finite decimal"
        )
    if not Decimal("0") < discount <= Decimal("100"):
        raise _field_error(
            path,
            row_number,
            "discount_percent",
            "must be greater than 0 and at most 100",
        )
    return discount


def _parse_bool(row: dict[str, str], path: Path, row_number: int) -> bool:
    value = _required_text(row, "is_active", path, row_number)
    if value == "1":
        return True
    if value == "0":
        return False
    raise _field_error(path, row_number, "is_active", "expected '0' or '1'")


def _ensure_unique_ids(ids: Iterable[int], path: Path) -> None:
    seen: set[int] = set()
    for promotion_id in ids:
        if promotion_id in seen:
            raise PromotionDataError(
                f"{path}: duplicate promotion_id: {promotion_id}"
            )
        seen.add(promotion_id)


def _field_error(
    path: Path, row_number: int, field_name: str, message: str
) -> PromotionDataError:
    return PromotionDataError(
        f"{path}: row {row_number}, field {field_name!r}: {message}"
    )
