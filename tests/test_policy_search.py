from __future__ import annotations

import unittest
from pathlib import Path

from chorddesk.policy import PolicySection, parse_policy_pdf
from chorddesk.policy_search import PolicyRetriever, search_policy


PROJECT_ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = PROJECT_ROOT / "data" / "políticas_da_loja.pdf"


class SuppliedPolicySearchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.sections = parse_policy_pdf(POLICY_PATH)
        cls.retriever = PolicyRetriever(cls.sections)

    def assert_section_near_top(
        self,
        expected_number: str,
        query: str,
        *,
        top: int = 3,
    ) -> None:
        numbers = [
            result.section.number
            for result in self.retriever.search(query, limit=top)
        ]
        self.assertIn(expected_number, numbers, (query, numbers))

    def test_store_hours(self) -> None:
        self.assert_section_near_top("2", "qual o horário da loja?")

    def test_online_return_right_of_withdrawal(self) -> None:
        self.assert_section_near_top(
            "4.1",
            "posso devolver uma compra online depois de 5 dias?",
        )

    def test_pix_discount_returns_both_useful_sections(self) -> None:
        numbers = [
            result.section.number
            for result in self.retriever.search("tem desconto no pix?", limit=3)
        ]
        self.assertIn("3", numbers)
        self.assertIn("6.2", numbers)

    def test_pix_promotion_stacking(self) -> None:
        self.assert_section_near_top(
            "6.2",
            "o desconto do pix acumula com promoção?",
        )

    def test_unsupported_accessories(self) -> None:
        self.assert_section_near_top("1", "vocês vendem cordas e palhetas?")

    def test_order_tracking(self) -> None:
        self.assert_section_near_top("5.3", "como rastrear meu pedido?")

    def test_warranty_sections(self) -> None:
        numbers = [
            result.section.number
            for result in self.retriever.search("qual a garantia?", limit=4)
        ]
        self.assertEqual(set(numbers), {"8", "8.1", "8.2", "8.3"})

    def test_accent_insensitive_search(self) -> None:
        accented = self.retriever.search("horário", limit=3)
        unaccented = self.retriever.search("horario", limit=3)
        self.assertEqual(accented, unaccented)

    def test_case_insensitive_search(self) -> None:
        lower = self.retriever.search("garantia", limit=4)
        upper = self.retriever.search("GARANTIA", limit=4)
        self.assertEqual(lower, upper)

    def test_rastrear_alias_matches_rastreamento(self) -> None:
        result = self.retriever.search("rastrear", limit=1)[0]
        self.assertEqual(result.section.number, "5.3")
        self.assertIn("rastreamento", result.heading_matches)

    def test_devolver_alias_matches_devolucao_and_arrependimento(self) -> None:
        result = self.retriever.search("devolver online", limit=1)[0]
        self.assertEqual(result.section.number, "4.1")
        self.assertIn("devolucao", result.heading_matches)


class PolicySearchBehaviorTests(unittest.TestCase):
    def test_equal_scores_preserve_source_order(self) -> None:
        sections = (
            PolicySection("9", "Alpha", "", 1, 1),
            PolicySection("2", "Alpha", "", 1, 1),
            PolicySection("1", "Alpha", "", 1, 1),
        )
        numbers = [
            result.section.number
            for result in PolicyRetriever(sections).search("alpha")
        ]
        self.assertEqual(numbers, ["9", "2", "1"])

    def test_limit_returns_deterministic_top_k(self) -> None:
        sections = (
            PolicySection("1", "Alpha Beta", "", 1, 1),
            PolicySection("2", "Alpha", "", 1, 1),
            PolicySection("3", "Other", "alpha", 1, 1),
        )
        retriever = PolicyRetriever(sections)
        first = retriever.search("alpha beta", limit=2)
        second = retriever.search("alpha beta", limit=2)
        self.assertEqual(first, second)
        self.assertEqual([result.section.number for result in first], ["1", "2"])

    def test_blank_query_fails_clearly(self) -> None:
        with self.assertRaisesRegex(ValueError, "query must not be blank"):
            PolicyRetriever(()).search("  \n ")

    def test_non_positive_limit_fails_clearly(self) -> None:
        with self.assertRaisesRegex(ValueError, "limit must be greater than zero"):
            PolicyRetriever(()).search("garantia", limit=0)

    def test_zero_match_returns_empty_tuple(self) -> None:
        section = PolicySection("1", "Garantia", "Defeitos", 1, 1)
        self.assertEqual(PolicyRetriever((section,)).search("xilofonismo"), ())

    def test_convenience_function_searches_supplied_sections(self) -> None:
        section = PolicySection("1", "Garantia", "Defeitos", 1, 1)
        results = search_policy((section,), "garantias")
        self.assertEqual(results[0].section, section)


if __name__ == "__main__":
    unittest.main()
