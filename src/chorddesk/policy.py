"""Deterministic parsing of ChordDesk's numbered policy document."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader
from pypdf.errors import PdfReadError


class PolicyParseError(ValueError):
    """Raised when a policy PDF cannot be parsed into numbered sections."""


@dataclass(frozen=True, slots=True)
class PolicySection:
    number: str
    title: str
    content: str
    page_start: int
    page_end: int


_PAGE_HEADER = "Empório da Música Manual de Políticas e Procedimentos"
_PAGE_FOOTER_RE = re.compile(r"^Página\s+\d+$")
_HEADING_RE = re.compile(
    r"^(?P<number>\d+(?:\.\d+)*)(?P<top_level_period>\.)? "
    r"(?P<title>\S.*)$"
)


def parse_policy_pdf(path: Path) -> tuple[PolicySection, ...]:
    """Parse numbered policy sections from a text-based PDF in source order."""
    try:
        reader = PdfReader(path)
    except (OSError, PdfReadError) as exc:
        raise PolicyParseError(f"unable to read policy PDF {path}: {exc}") from exc

    pages: list[tuple[int, tuple[str, ...]]] = []
    for page_number, page in enumerate(reader.pages, start=1):
        try:
            extracted = page.extract_text(extraction_mode="layout")
        except Exception as exc:
            raise PolicyParseError(
                f"unable to extract text from {path}, page {page_number}: {exc}"
            ) from exc
        if not extracted or not extracted.strip():
            raise PolicyParseError(
                f"no extractable text in policy PDF {path}, page {page_number}"
            )
        pages.append((page_number, _clean_page(extracted)))

    sections = _parse_numbered_sections(pages)
    if not sections:
        raise PolicyParseError(f"no numbered policy sections found in {path}")
    return sections


def _clean_page(text: str) -> tuple[str, ...]:
    lines = []
    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()
        if stripped == _PAGE_HEADER or _PAGE_FOOTER_RE.fullmatch(stripped):
            continue
        lines.append(line)

    while lines and not lines[0]:
        lines.pop(0)
    while lines and not lines[-1]:
        lines.pop()
    return tuple(lines)


def _parse_numbered_sections(
    pages: list[tuple[int, tuple[str, ...]]],
) -> tuple[PolicySection, ...]:
    sections: list[PolicySection] = []
    current_number: str | None = None
    current_title = ""
    current_page_start = 0
    current_page_end = 0
    content_lines: list[str] = []

    def finish_section() -> None:
        nonlocal content_lines
        if current_number is None:
            return
        content = "\n".join(_trim_blank_lines(content_lines))
        sections.append(
            PolicySection(
                number=current_number,
                title=current_title,
                content=content,
                page_start=current_page_start,
                page_end=current_page_end,
            )
        )
        content_lines = []

    for page_number, lines in pages:
        if current_number is not None and content_lines:
            content_lines.append("")
        for line in lines:
            heading = _match_heading(line)
            if heading is not None:
                finish_section()
                current_number, current_title = heading
                current_page_start = page_number
                current_page_end = page_number
                continue
            if current_number is not None:
                content_lines.append(line)
                if line.strip():
                    current_page_end = page_number

    finish_section()
    return tuple(sections)


def _match_heading(line: str) -> tuple[str, str] | None:
    if not line or line[0].isspace():
        return None
    match = _HEADING_RE.fullmatch(line)
    if match is None:
        return None

    number = match.group("number")
    has_top_level_period = match.group("top_level_period") is not None
    is_top_level = "." not in number
    if has_top_level_period != is_top_level:
        return None
    return number, match.group("title")


def _trim_blank_lines(lines: list[str]) -> list[str]:
    start = 0
    end = len(lines)
    while start < end and not lines[start]:
        start += 1
    while end > start and not lines[end - 1]:
        end -= 1
    return lines[start:end]
