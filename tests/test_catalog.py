from __future__ import annotations

import tempfile
import unittest
from datetime import date
from decimal import Decimal
from pathlib import Path

from chorddesk.catalog import (
    Catalog,
    CatalogDataError,
    Category,
    Product,
    ProductStatus,
    load_categories,
    load_products,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
CATEGORIES_PATH = DATA_DIR / "desafio_tecnico_ai_eng - categories.csv"
PRODUCTS_PATH = DATA_DIR / "desafio_tecnico_ai_eng - products.csv"


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


class ProductAvailabilityTests(unittest.TestCase):
    def test_active_product_with_positive_stock_is_available(self) -> None:
        product = make_product(1, name="Available", price="10.00")
        self.assertTrue(product.is_available_for_sale)

    def test_active_product_with_zero_stock_is_unavailable(self) -> None:
        product = make_product(
            1, name="No stock", price="10.00", stock_quantity=0
        )
        self.assertFalse(product.is_available_for_sale)

    def test_coming_soon_product_with_stock_is_unavailable(self) -> None:
        product = make_product(
            1,
            name="Coming soon",
            price="10.00",
            stock_quantity=2,
            status=ProductStatus.COMING_SOON,
        )
        self.assertFalse(product.is_available_for_sale)

    def test_discontinued_product_with_stock_is_unavailable(self) -> None:
        product = make_product(
            1,
            name="Discontinued",
            price="10.00",
            stock_quantity=13,
            status=ProductStatus.DISCONTINUED,
        )
        self.assertFalse(product.is_available_for_sale)

    def test_supplied_edge_cases_are_unavailable(self) -> None:
        catalog = Catalog.from_csv(CATEGORIES_PATH, PRODUCTS_PATH)
        for product_id in (96, 130, 136):
            with self.subTest(product_id=product_id):
                product = catalog.get_product(product_id)
                self.assertIsNotNone(product)
                assert product is not None
                self.assertFalse(product.is_available_for_sale)


class CatalogTests(unittest.TestCase):
    def setUp(self) -> None:
        self.guitar = make_product(
            1,
            name="Classic Guitar",
            price="999.90",
            description="Nylon string instrument",
        )
        self.piano = make_product(
            2,
            name="Stage Piano",
            price="1000.00",
            category_id=2,
            description="Digital keyboard",
        )
        self.unavailable_guitar = make_product(
            3,
            name="Future Guitar",
            price="1000.01",
            stock_quantity=4,
            status=ProductStatus.COMING_SOON,
        )
        self.catalog = Catalog(
            categories=(
                Category(1, "Guitars", "String instruments"),
                Category(2, "Keys", "Keyboard instruments"),
            ),
            products=(self.guitar, self.piano, self.unavailable_guitar),
        )

    def test_product_lookup(self) -> None:
        self.assertIs(self.catalog.get_product(2), self.piano)
        self.assertIsNone(self.catalog.get_product(999))

    def test_search_combines_text_category_and_availability_filters(self) -> None:
        result = self.catalog.search_products(
            query="GUITAR", category_id=1, available_only=True
        )
        self.assertEqual(result, (self.guitar,))

    def test_text_search_includes_description(self) -> None:
        self.assertEqual(
            self.catalog.search_products(query="keyboard"), (self.piano,)
        )

    def test_price_filtering_uses_exact_inclusive_decimal_comparison(self) -> None:
        result = self.catalog.search_products(
            min_price=Decimal("1000.00"), max_price=Decimal("1000.00")
        )
        self.assertEqual(result, (self.piano,))

    def test_rejects_inverted_price_range(self) -> None:
        with self.assertRaisesRegex(ValueError, "min_price"):
            self.catalog.search_products(
                min_price=Decimal("2.00"), max_price=Decimal("1.00")
            )


class LoadingTests(unittest.TestCase):
    def test_loads_supplied_categories(self) -> None:
        categories = load_categories(CATEGORIES_PATH)
        self.assertEqual(len(categories), 9)
        self.assertEqual(categories[0].category_id, 1)
        self.assertEqual(categories[0].name, "Guitarras")

    def test_loads_all_supplied_products(self) -> None:
        self.assertEqual(len(load_products(PRODUCTS_PATH)), 65)

    def test_loads_supplied_product_types(self) -> None:
        product = load_products(PRODUCTS_PATH)[0]
        self.assertIsInstance(product.product_id, int)
        self.assertIsInstance(product.price_brl, Decimal)
        self.assertIsInstance(product.created_at, date)
        self.assertIsInstance(product.status, ProductStatus)

    def test_invalid_required_data_fails_with_field_context(self) -> None:
        csv_text = (
            "product_id,price_brl,name,category_id,description,stock_quantity,"
            "status,specs,created_at\n"
            '1,not-money,Guitar,1,Description,2,active,"{}",2024-01-01\n'
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "products.csv"
            path.write_text(csv_text, encoding="utf-8")
            with self.assertRaisesRegex(CatalogDataError, "price_brl"):
                load_products(path)


if __name__ == "__main__":
    unittest.main()
