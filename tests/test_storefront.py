from __future__ import annotations

import unittest
from dataclasses import FrozenInstanceError
from datetime import date
from decimal import Decimal
from pathlib import Path

from chorddesk.catalog import Catalog, Category, Product, ProductStatus
from chorddesk.promotions import Promotion, Promotions
from chorddesk.storefront import (
    StorefrontDataError,
    StorefrontService,
    UnknownCategoryError,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
CATEGORIES_PATH = DATA_DIR / "desafio_tecnico_ai_eng - categories.csv"
PRODUCTS_PATH = DATA_DIR / "desafio_tecnico_ai_eng - products.csv"
PROMOTIONS_PATH = DATA_DIR / "desafio_tecnico_ai_eng - promotions.csv"


def make_product(
    product_id: int,
    *,
    name: str,
    price: str,
    category_id: int = 1,
    stock_quantity: int = 1,
    status: ProductStatus = ProductStatus.ACTIVE,
    description: str = "Instrumento musical",
) -> Product:
    return Product(
        product_id=product_id,
        price_brl=Decimal(price),
        name=name,
        category_id=category_id,
        description=description,
        stock_quantity=stock_quantity,
        status=status,
        specs={"color": "Natural"},
        created_at=date(2024, 1, 1),
    )


def make_promotion(
    promotion_id: int,
    *,
    product_id: int,
    discount_percent: str = "10",
    is_active: bool = True,
) -> Promotion:
    return Promotion(
        promotion_id=promotion_id,
        product_id=product_id,
        discount_percent=Decimal(discount_percent),
        description="Oferta especial",
        is_active=is_active,
    )


class StorefrontProductSearchTests(unittest.TestCase):
    def setUp(self) -> None:
        self.available_guitar = make_product(
            1,
            name="Violão Clássico",
            price="999.90",
            category_id=5,
            description="Cordas de nylon",
        )
        self.expensive_guitar = make_product(
            2,
            name="Violão Profissional",
            price="1500.00",
            category_id=5,
            description="Cordas de aço",
        )
        self.unavailable_guitar = make_product(
            3,
            name="Violão Futuro",
            price="1200.00",
            category_id=5,
            status=ProductStatus.COMING_SOON,
            stock_quantity=4,
        )
        self.keyboard = make_product(
            4,
            name="Teclado Digital",
            price="1100.00",
            category_id=4,
        )
        catalog = Catalog(
            categories=(
                Category(5, "Violões", "Instrumentos de corda"),
                Category(4, "Teclados e Pianos", "Instrumentos de teclas"),
            ),
            products=(
                self.available_guitar,
                self.expensive_guitar,
                self.unavailable_guitar,
                self.keyboard,
            ),
        )
        self.service = StorefrontService(catalog, Promotions(()))

    def test_resolves_exact_category_name(self) -> None:
        results = self.service.search_products(category="Violões")
        self.assertEqual(tuple(result.product_id for result in results), (1, 2))

    def test_resolves_category_name_without_accents(self) -> None:
        results = self.service.search_products(category="violoes")
        self.assertEqual(tuple(result.product_id for result in results), (1, 2))

    def test_resolves_trimmed_case_insensitive_category_name(self) -> None:
        results = self.service.search_products(category="  VIOLÕES  ")
        self.assertEqual(tuple(result.product_id for result in results), (1, 2))

    def test_unknown_category_fails_clearly(self) -> None:
        with self.assertRaisesRegex(UnknownCategoryError, "Bandolins"):
            self.service.search_products(category="Bandolins")

    def test_combines_query_category_and_price_in_source_order(self) -> None:
        results = self.service.search_products(
            query="violão",
            category="Violões",
            min_price=Decimal("1000.00"),
            max_price=Decimal("1500.00"),
        )
        self.assertEqual(tuple(result.product_id for result in results), (2,))

    def test_search_defaults_to_available_products(self) -> None:
        results = self.service.search_products(category="Violões")
        self.assertNotIn(3, (result.product_id for result in results))

    def test_unavailable_products_can_be_included(self) -> None:
        results = self.service.search_products(
            category="Violões", available_only=False
        )
        self.assertEqual(tuple(result.product_id for result in results), (1, 2, 3))
        self.assertFalse(results[-1].is_available)

    def test_result_has_category_name_and_exact_decimal_price(self) -> None:
        result = self.service.search_products(category="Violões")[0]
        self.assertEqual(result.category_name, "Violões")
        self.assertEqual(result.price_brl, Decimal("999.90"))
        self.assertIsInstance(result.price_brl, Decimal)

    def test_results_and_specs_are_immutable(self) -> None:
        result = self.service.search_products(category="Violões")[0]
        with self.assertRaises(FrozenInstanceError):
            result.name = "Changed"  # type: ignore[misc]
        with self.assertRaises(TypeError):
            result.specs["color"] = "Changed"  # type: ignore[index]

    def test_normalized_category_name_collision_fails_construction(self) -> None:
        catalog = Catalog(
            categories=(
                Category(1, "Violões", "First"),
                Category(2, "Violoes", "Second"),
            ),
            products=(),
        )
        with self.assertRaisesRegex(StorefrontDataError, "ambiguous"):
            StorefrontService(catalog, Promotions(()))


class StorefrontPromotionTests(unittest.TestCase):
    def test_active_promotion_reports_unavailable_product_independently(self) -> None:
        unavailable = make_product(
            10,
            name="Sem estoque",
            price="199.90",
            stock_quantity=0,
        )
        service = StorefrontService(
            Catalog(
                categories=(Category(1, "Guitarras", "Cordas"),),
                products=(unavailable,),
            ),
            Promotions((make_promotion(1, product_id=10, discount_percent="15"),)),
        )

        result = service.get_active_promotions()[0]

        self.assertFalse(result.product_is_available)
        self.assertEqual(result.original_price_brl, Decimal("199.90"))
        self.assertEqual(result.discounted_price_brl, Decimal("169.915"))
        self.assertIsInstance(result.original_price_brl, Decimal)
        self.assertIsInstance(result.discounted_price_brl, Decimal)
        self.assertEqual(service.get_active_promotions(available_only=True), ())

    def test_active_flag_does_not_override_product_availability(self) -> None:
        coming_soon = make_product(
            10,
            name="Lançamento",
            price="100.00",
            stock_quantity=5,
            status=ProductStatus.COMING_SOON,
        )
        service = StorefrontService(
            Catalog(
                categories=(Category(1, "Guitarras", "Cordas"),),
                products=(coming_soon,),
            ),
            Promotions((make_promotion(1, product_id=10),)),
        )
        self.assertFalse(service.get_active_promotions()[0].product_is_available)

    def test_unknown_promoted_product_fails_clearly(self) -> None:
        service = StorefrontService(
            Catalog(categories=(), products=()),
            Promotions((make_promotion(1, product_id=999),)),
        )
        with self.assertRaisesRegex(StorefrontDataError, "product_id 999"):
            service.get_active_promotions()


class SuppliedDataStorefrontTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.service = StorefrontService(
            Catalog.from_csv(CATEGORIES_PATH, PRODUCTS_PATH),
            Promotions.from_csv(PROMOTIONS_PATH),
        )

    def test_all_four_active_promotions_join_correct_products_in_source_order(self) -> None:
        results = self.service.get_active_promotions()
        self.assertEqual(
            tuple((result.promotion_id, result.product_id) for result in results),
            ((12, 127), (14, 121), (19, 90), (24, 94)),
        )

    def test_supplied_promotions_include_joined_customer_facts(self) -> None:
        results = self.service.get_active_promotions()
        by_id = {result.promotion_id: result for result in results}
        self.assertEqual(by_id[12].product_name, "Kalani KAL-700T Tenor Natural")
        self.assertEqual(by_id[12].category_name, "Ukuleles")
        self.assertEqual(by_id[12].original_price_brl, Decimal("359"))
        self.assertEqual(by_id[12].discounted_price_brl, Decimal("323.1"))
        self.assertTrue(all(result.product_is_available for result in results))


if __name__ == "__main__":
    unittest.main()
