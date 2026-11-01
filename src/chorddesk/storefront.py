"""Customer-safe application services for products and promotions."""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from decimal import Decimal
from types import MappingProxyType
from typing import Mapping

from chorddesk.catalog import Catalog, Category, Product
from chorddesk.promotions import Promotion, Promotions


class StorefrontDataError(ValueError):
    """Raised when injected deterministic data cannot be joined safely."""


class UnknownCategoryError(LookupError):
    """Raised when a customer-facing category name is not in the catalog."""


@dataclass(frozen=True, slots=True)
class ProductResult:
    """Customer-safe product facts returned by product search."""

    product_id: int
    name: str
    category_name: str
    price_brl: Decimal
    is_available: bool
    stock_quantity: int
    description: str
    specs: Mapping[str, str]

    def __post_init__(self) -> None:
        object.__setattr__(self, "specs", MappingProxyType(dict(self.specs)))


@dataclass(frozen=True, slots=True)
class ActivePromotionResult:
    """An active promotion joined to its customer-facing product facts."""

    promotion_id: int
    description: str
    discount_percent: Decimal
    product_id: int
    product_name: str
    category_name: str
    original_price_brl: Decimal
    discounted_price_brl: Decimal
    product_is_available: bool
    stock_quantity: int


class StorefrontService:
    """Application boundary over already-loaded catalog and promotion data."""

    def __init__(self, catalog: Catalog, promotions: Promotions) -> None:
        self._catalog = catalog
        self._promotions = promotions
        self._categories_by_id = {
            category.category_id: category for category in catalog.categories
        }
        self._categories_by_name = self._index_category_names(catalog.categories)

    def search_products(
        self,
        *,
        query: str | None = None,
        category: str | None = None,
        min_price: Decimal | None = None,
        max_price: Decimal | None = None,
        available_only: bool = True,
    ) -> tuple[ProductResult, ...]:
        """Search products using a human-facing category name."""
        category_id = (
            self._resolve_category(category).category_id
            if category is not None
            else None
        )
        products = self._catalog.search_products(
            query=query,
            category_id=category_id,
            min_price=min_price,
            max_price=max_price,
            available_only=available_only,
        )
        return tuple(self._product_result(product) for product in products)

    def get_active_promotions(
        self, *, available_only: bool = False
    ) -> tuple[ActivePromotionResult, ...]:
        """Return active promotions, optionally limited to saleable products."""
        results: list[ActivePromotionResult] = []
        for promotion in self._promotions.active_promotions():
            product = self._catalog.get_product(promotion.product_id)
            if product is None:
                raise StorefrontDataError(
                    "active promotion "
                    f"{promotion.promotion_id} references unknown product_id "
                    f"{promotion.product_id}"
                )
            if available_only and not product.is_available_for_sale:
                continue
            results.append(self._promotion_result(promotion, product))
        return tuple(results)

    def _resolve_category(self, name: str) -> Category:
        normalized_name = _normalize_category_name(name)
        category = self._categories_by_name.get(normalized_name)
        if category is None:
            raise UnknownCategoryError(f"unknown category: {name!r}")
        return category

    def _product_result(self, product: Product) -> ProductResult:
        category = self._category_for_product(product)
        return ProductResult(
            product_id=product.product_id,
            name=product.name,
            category_name=category.name,
            price_brl=product.price_brl,
            is_available=product.is_available_for_sale,
            stock_quantity=product.stock_quantity,
            description=product.description,
            specs=product.specs,
        )

    def _promotion_result(
        self, promotion: Promotion, product: Product
    ) -> ActivePromotionResult:
        category = self._category_for_product(product)
        return ActivePromotionResult(
            promotion_id=promotion.promotion_id,
            description=promotion.description,
            discount_percent=promotion.discount_percent,
            product_id=product.product_id,
            product_name=product.name,
            category_name=category.name,
            original_price_brl=product.price_brl,
            discounted_price_brl=promotion.discounted_price(product.price_brl),
            product_is_available=product.is_available_for_sale,
            stock_quantity=product.stock_quantity,
        )

    def _category_for_product(self, product: Product) -> Category:
        category = self._categories_by_id.get(product.category_id)
        if category is None:
            raise StorefrontDataError(
                f"product {product.product_id} references unknown category_id "
                f"{product.category_id}"
            )
        return category

    @staticmethod
    def _index_category_names(
        categories: tuple[Category, ...],
    ) -> dict[str, Category]:
        indexed: dict[str, Category] = {}
        for category in categories:
            normalized_name = _normalize_category_name(category.name)
            existing = indexed.get(normalized_name)
            if existing is not None:
                raise StorefrontDataError(
                    "ambiguous normalized category name: "
                    f"{existing.name!r} and {category.name!r}"
                )
            indexed[normalized_name] = category
        return indexed


def _normalize_category_name(name: str) -> str:
    decomposed = unicodedata.normalize("NFKD", name.strip().casefold())
    return "".join(
        character
        for character in decomposed
        if not unicodedata.combining(character)
    )
