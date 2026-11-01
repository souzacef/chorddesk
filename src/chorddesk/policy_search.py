"""Deterministic lexical retrieval over parsed policy sections."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Iterable

from chorddesk.policy import PolicySection


_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    {
        "a",
        "ao",
        "aos",
        "as",
        "como",
        "da",
        "das",
        "de",
        "depois",
        "do",
        "dos",
        "e",
        "em",
        "eu",
        "meu",
        "minha",
        "na",
        "nas",
        "no",
        "nos",
        "o",
        "os",
        "ou",
        "para",
        "por",
        "posso",
        "qual",
        "que",
        "se",
        "sem",
        "sob",
        "sobre",
        "tem",
        "uma",
        "um",
        "voces",
    }
)

_ALIAS_GROUPS = {
    "devolucao": {"arrependimento", "devolver", "devolucao", "devolucoes"},
    "rastreamento": {"rastrear", "rastreio", "rastreamento", "rastreamentos"},
    "promocao": {"promocao", "promocoes", "promocional", "promocionais"},
    "cumulativo": {
        "acumula",
        "acumular",
        "acumulativa",
        "acumulativas",
        "acumulativo",
        "acumulativos",
        "cumulatividade",
        "cumulativa",
        "cumulativas",
        "cumulativo",
        "cumulativos",
    },
    "vender": {
        "comercializa",
        "comercializamos",
        "comercializar",
        "venda",
        "vendas",
        "vendem",
        "vender",
    },
    "horario": {
        "abre",
        "aberto",
        "abertos",
        "fecha",
        "fechada",
        "fechado",
        "fechados",
        "funcionamento",
        "horario",
        "horarios",
    },
    "garantia": {"garantia", "garantias"},
}
_ALIASES = {
    alias: canonical
    for canonical, aliases in _ALIAS_GROUPS.items()
    for alias in aliases
}

_HEADING_MATCH_WEIGHT = 4
_BODY_MATCH_WEIGHT = 1


@dataclass(frozen=True, slots=True)
class PolicySearchResult:
    """A policy section and its explainable deterministic relevance data."""

    section: PolicySection
    score: int
    heading_matches: tuple[str, ...]
    body_matches: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _IndexedSection:
    section: PolicySection
    heading_tokens: frozenset[str]
    body_tokens: frozenset[str]


@dataclass(frozen=True, slots=True, init=False)
class PolicyRetriever:
    """Immutable reusable lexical index for already-parsed policy sections."""

    _index: tuple[_IndexedSection, ...]

    def __init__(self, sections: Iterable[PolicySection]) -> None:
        index = tuple(
            _IndexedSection(
                section=section,
                heading_tokens=_normalize_tokens(section.title),
                body_tokens=_normalize_tokens(section.content),
            )
            for section in sections
        )
        object.__setattr__(self, "_index", index)

    def search(
        self,
        query: str,
        *,
        limit: int = 3,
    ) -> tuple[PolicySearchResult, ...]:
        """Return the highest-scoring policy sections in source order on ties."""
        if not query.strip():
            raise ValueError("policy search query must not be blank")
        if limit <= 0:
            raise ValueError("policy search limit must be greater than zero")

        query_tokens = _normalize_tokens(query)
        if not query_tokens:
            return ()

        results: list[PolicySearchResult] = []
        for indexed in self._index:
            heading_matches = query_tokens & indexed.heading_tokens
            body_matches = query_tokens & indexed.body_tokens
            score = (
                _HEADING_MATCH_WEIGHT * len(heading_matches)
                + _BODY_MATCH_WEIGHT * len(body_matches)
            )
            if score == 0:
                continue
            results.append(
                PolicySearchResult(
                    section=indexed.section,
                    score=score,
                    heading_matches=tuple(sorted(heading_matches)),
                    body_matches=tuple(sorted(body_matches)),
                )
            )

        results.sort(key=lambda result: result.score, reverse=True)
        return tuple(results[:limit])


def search_policy(
    sections: Iterable[PolicySection],
    query: str,
    *,
    limit: int = 3,
) -> tuple[PolicySearchResult, ...]:
    """Search parsed policy sections without retaining a reusable index."""
    return PolicyRetriever(sections).search(query, limit=limit)


def _normalize_tokens(text: str) -> frozenset[str]:
    decomposed = unicodedata.normalize("NFKD", text.casefold())
    unaccented = "".join(
        character
        for character in decomposed
        if not unicodedata.combining(character)
    )
    return frozenset(
        _ALIASES.get(token, token)
        for token in _TOKEN_RE.findall(unaccented)
        if token not in _STOPWORDS
    )
