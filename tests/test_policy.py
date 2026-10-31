from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from chorddesk.policy import PolicyParseError, parse_policy_pdf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = PROJECT_ROOT / "data" / "políticas_da_loja.pdf"


class PolicyParsingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.sections = parse_policy_pdf(POLICY_PATH)
        cls.by_number = {section.number: section for section in cls.sections}

    def test_supplied_pdf_parses_all_top_level_sections(self) -> None:
        top_level_numbers = [
            section.number for section in self.sections if "." not in section.number
        ]
        self.assertEqual(top_level_numbers, [str(number) for number in range(1, 11)])

    def test_important_subsections_are_detected(self) -> None:
        for number in ("4.1", "6.2", "7.3", "8.3"):
            with self.subTest(number=number):
                self.assertIn(number, self.by_number)

    def test_section_titles_are_parsed_exactly(self) -> None:
        self.assertEqual(
            self.by_number["4.1"].title,
            "Direito de Arrependimento (Compras Online)",
        )
        self.assertEqual(
            self.by_number["10"].title,
            "Disposições Finais",
        )

    def test_sections_with_source_body_have_non_empty_content(self) -> None:
        for number in ("1", "3.1", "4.1", "5.2", "6.2", "9", "10"):
            with self.subTest(number=number):
                self.assertTrue(self.by_number[number].content.strip())

    def test_section_order_is_deterministic(self) -> None:
        self.assertEqual(
            [section.number for section in self.sections[:8]],
            ["1", "1.1", "1.2", "2", "3", "3.1", "4", "4.1"],
        )
        self.assertEqual(
            [section.number for section in parse_policy_pdf(POLICY_PATH)],
            [section.number for section in self.sections],
        )

    def test_page_furniture_is_removed_from_policy_content(self) -> None:
        all_content = "\n".join(section.content for section in self.sections)
        self.assertNotIn(
            "Empório da Música Manual de Políticas e Procedimentos",
            all_content,
        )
        self.assertNotRegex(all_content, r"(?m)^\s*Página\s+\d+\s*$")

    def test_cross_page_sections_are_preserved_as_single_sections(self) -> None:
        shipping = self.by_number["5.2"]
        self.assertEqual((shipping.page_start, shipping.page_end), (4, 5))
        self.assertIn("Utilizamos os serviços dos Correios", shipping.content)

        service_flow = self.by_number["7.2"]
        self.assertEqual((service_flow.page_start, service_flow.page_end), (6, 7))
        self.assertIn("3.   Consulta:", service_flow.content)

    def test_representative_page_metadata(self) -> None:
        self.assertEqual(
            (self.by_number["1"].page_start, self.by_number["1"].page_end),
            (2, 2),
        )
        self.assertEqual(
            (self.by_number["9"].page_start, self.by_number["9"].page_end),
            (7, 8),
        )


class PolicyParsingFailureTests(unittest.TestCase):
    def test_pdf_without_numbered_sections_fails_clearly(self) -> None:
        page = MagicMock()
        page.extract_text.return_value = "Documento sem seções numeradas."
        reader = MagicMock()
        reader.pages = [page]

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "policy.pdf"
            path.touch()
            with patch("chorddesk.policy.PdfReader", return_value=reader):
                with self.assertRaisesRegex(
                    PolicyParseError, "no numbered policy sections"
                ):
                    parse_policy_pdf(path)

        page.extract_text.assert_called_once_with(extraction_mode="layout")

    def test_missing_pdf_fails_with_path_context(self) -> None:
        missing = Path("does-not-exist.pdf")
        with self.assertRaisesRegex(PolicyParseError, "does-not-exist.pdf"):
            parse_policy_pdf(missing)


if __name__ == "__main__":
    unittest.main()
