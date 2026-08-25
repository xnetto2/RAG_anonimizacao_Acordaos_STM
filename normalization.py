"""Normalização canônica das entidades jurídicas do grafo."""
from __future__ import annotations

import re
import unicodedata


CODE_ALIASES = {
    "CPM": ("CPM", "CODIGO PENAL MILITAR"),
    "CPPM": ("CPPM", "CODIGO DE PROCESSO PENAL MILITAR"),
    "CF": ("CF", "CF/88", "CONSTITUICAO FEDERAL"),
    "CP": ("CP", "CODIGO PENAL"),
    "CPP": ("CPP", "CODIGO DE PROCESSO PENAL"),
}


def ascii_upper(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value or "")
    return " ".join("".join(c for c in decomposed if not unicodedata.combining(c)).upper().split())


def normalize_process(value: str) -> str:
    compact = re.sub(r"\s+", "", value or "")
    match = re.search(r"\d{7}-\d{2}\.\d{4}\.\d\.\d{2}\.\d{4}", compact)
    return f"PROCESSO:{match.group(0)}" if match else f"PROCESSO:{ascii_upper(value)}"


def identify_code(value: str) -> str | None:
    normalized = ascii_upper(value)
    # Nomes extensos primeiro para impedir que CP seja encontrado dentro de CPPM.
    for code, aliases in CODE_ALIASES.items():
        if any(re.search(rf"(?<![A-Z]){re.escape(alias)}(?![A-Z])", normalized) for alias in sorted(aliases, key=len, reverse=True)):
            return code
    return None


def normalize_norm(value: str) -> str:
    normalized = ascii_upper(value)
    code = identify_code(value) or "NORMA_NAO_IDENTIFICADA"
    article = re.search(r"\bART(?:IGO)?\.?\s*(\d+[A-Z]?)", normalized)
    parts = [code, f"ARTIGO_{article.group(1)}" if article else normalized]
    paragraph = re.search(r"(?:§|PARAGRAFO)\s*(\d+|UNICO)", normalized)
    inciso = re.search(r"\b(?:INCISO\s*)?([IVXLCDM]+)\b", normalized)
    alinea = re.search(r"\bALINEA\s*['\"]?([A-Z])", normalized)
    if paragraph:
        parts.append(f"PARAGRAFO_{paragraph.group(1)}")
    if inciso:
        parts.append(f"INCISO_{inciso.group(1)}")
    if alinea:
        parts.append(f"ALINEA_{alinea.group(1)}")
    return ":".join(parts)


COURT_ALIASES = {
    "STM": ("STM", "SUPERIOR TRIBUNAL MILITAR", "CORTE CASTRENSE", "CORTE MARCIAL"),
    "STJ": ("STJ", "SUPERIOR TRIBUNAL DE JUSTICA"),
    "STF": ("STF", "SUPREMO TRIBUNAL FEDERAL", "CORTE SUPREMA", "SUPREMA CORTE"),
}

# Catálogo confirmado por inspeção dos excertos deste corpus. Ele funciona
# como desempate quando a sigla não foi capturada ou sofreu erro de OCR.
SUMULA_DEFAULT_COURTS = {
    3: "STM", 5: "STM", 8: "STM", 9: "STM", 12: "STM", 14: "STM",
    17: "STM", 18: "STM",
    7: "STJ", 16: "STJ", 111: "STJ", 182: "STJ", 231: "STJ",
    513: "STJ", 545: "STJ",
    146: "STF", 279: "STF", 282: "STF", 339: "STF", 711: "STF",
}


def identify_sumula_court(value: str, context: str = "") -> str:
    """Identifica tribunal ligado sintaticamente à menção; usa catálogo no desempate."""
    normalized_value = ascii_upper(value)
    for court, aliases in COURT_ALIASES.items():
        if any(re.search(rf"(?<![A-Z]){re.escape(alias)}(?![A-Z])", normalized_value) for alias in aliases):
            return court
    normalized_context = ascii_upper(context or value)
    occurrences = [m.start() for m in re.finditer(re.escape(normalized_value), normalized_context)]
    center = len(normalized_context) // 2
    anchor = min(occurrences, key=lambda pos: abs(pos - center)) if occurrences else center

    before = normalized_context[max(0, anchor - 80):anchor]
    after = normalized_context[anchor + len(normalized_value):anchor + len(normalized_value) + 100]
    for court, aliases in COURT_ALIASES.items():
        for alias in aliases:
            escaped = re.escape(alias)
            # Formas como "Súmula 231 do STJ" e "Súmula 339/STF".
            if re.match(rf"^\s*(?:N[º°.]?\s*)?(?:(?:DO|DA|DESTE|DESTA)\s+|/\s*|\(\s*){escaped}(?![A-Z])", after):
                return court
            # Formas como "entendimento do STJ, Súmula 231" ou
            # "sumulada pelo STM, in verbis: Súmula 17".
            if re.search(rf"(?<![A-Z]){escaped}(?![A-Z]).{{0,55}}$", before):
                return court
    number = re.search(r"\b(\d+)\b", normalized_value)
    return SUMULA_DEFAULT_COURTS.get(int(number.group(1)), "STM") if number else "STM"


def normalize_sumula(value: str, context: str = "") -> str:
    normalized = ascii_upper(value)
    number = re.search(r"\b(\d+)\b", normalized)
    court = identify_sumula_court(value, context)
    # Remove zeros à esquerda para unir, por exemplo, Súmula 03 e Súmula 3.
    normalized_number = str(int(number.group(1))) if number else normalized
    return f"{court}:SUMULA_{normalized_number}"


def canonical_entity(kind: str, label: str, context: str = "") -> str:
    if kind == "SUMULA":
        return normalize_sumula(label, context)
    return {
        "PROCESSO": normalize_process,
        "ACORDAO": normalize_process,
        "NORMA": normalize_norm,
    }.get(kind, lambda value: f"{kind}:{ascii_upper(value)}")(label)
