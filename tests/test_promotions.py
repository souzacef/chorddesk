from __future__ import annotations

import tempfile
import unittest
from decimal import Decimal
from pathlib import Path

from chorddesk.promotions import (
    Promotion,
    PromotionDataError,
    Promotions,
    load_promotions,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROMOTIONS_PATH = (
    PROJECT_ROOT / "data" / "desafio_tecnico_ai_eng - promotions.csv"
)
CSV_HEADER = (
    "promotion_id,product_id,discount_percent,description,is_active\n"
)


def make_promotion(
    promotion_id: int = 1,
    *,
    product_id: int = 100,
    discount_percent: str = "10",
    is_active: bool = True,
) -> Promotion:
    return Promotion(
        promotion_id=promotion_id,
        product_id=product_id,
        discount_percent=Decimal(discount_percent),
        description="Special offer",
        is_active=is_active,
    )


def write_promotions(directory: str, rows: str) -> Path:
    path = Path(directory) / "promotions.csv"
    path.write_text(CSV_HEADER + rows, encoding="utf-8")
    return path


class PromotionTests(unittest.TestCase):
    def test_calculates_exact_decimal_discounted_price(self) -> None:
        promotion = make_promotion(discount_percent="15")
        self.assertEqual(
            promotion.discounted_price(Decimal("199.90")), Decimal("169.915")
        )


class PromotionsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.promotions = Promotions.from_csv(PROMOTIONS_PATH)

    def test_lookup_by_promotion_id(self) -> None:
        promotion = self.promotions.get_promotion(12)
        self.assertIsNotNone(promotion)
        assert promotion is not None
        self.assertEqual(promotion.product_id, 127)
        self.assertEqual(promotion.discount_percent, Decimal("10"))
        self.assertIsNone(self.promotions.get_promotion(999))

    def test_finds_promotions_for_product_in_source_order(self) -> None:
        promotions = self.promotions.for_product(96)
        self.assertEqual(
            tuple(promotion.promotion_id for promotion in promotions), (8, 22)
        )

    def test_filters_active_promotions(self) -> None:
        active = self.promotions.active_promotions()
        self.assertEqual(
            tuple(promotion.promotion_id for promotion in active), (12, 14, 19, 24)
        )


class PromotionLoadingTests(unittest.TestCase):
    def test_loads_all_supplied_promotions(self) -> None:
        self.assertEqual(len(load_promotions(PROMOTIONS_PATH)), 25)

    def test_supplied_data_has_four_active_promotions(self) -> None:
        promotions = load_promotions(PROMOTIONS_PATH)
        self.assertEqual(sum(promotion.is_active for promotion in promotions), 4)

    def test_rejects_invalid_discount_percentage(self) -> None:
        for discount in ("0", "100.01", "NaN"):
            with self.subTest(discount=discount), tempfile.TemporaryDirectory() as directory:
                path = write_promotions(
                    directory, f"1,100,{discount},Special offer,1\n"
                )
                with self.assertRaisesRegex(
                    PromotionDataError, "discount_percent"
                ):
                    load_promotions(path)

    def test_rejects_invalid_active_flag(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = write_promotions(directory, "1,100,10,Special offer,true\n")
            with self.assertRaisesRegex(PromotionDataError, "is_active"):
                load_promotions(path)

    def test_rejects_duplicate_promotion_ids(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = write_promotions(
                directory,
                "1,100,10,First offer,1\n"
                "1,101,15,Second offer,0\n",
            )
            with self.assertRaisesRegex(PromotionDataError, "duplicate promotion_id"):
                load_promotions(path)


if __name__ == "__main__":
    unittest.main()
