"""Anonimização auditável dos acórdãos em PDF.

Os arquivos oficiais nunca são sobrescritos. A rotina cria cópias com redações
reais (remoção do conteúdo subjacente), elimina metadados e registra apenas a
impressão digital dos valores removidos no relatório público de auditoria.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import pymupdf as fitz


# Identificadores cuja remoção independe de contexto.
IDENTIFIER_PATTERNS = {
    "CPF": re.compile(r"(?<!\d)\d{3}\.?\d{3}\.?\d{3}-?\d{2}(?!\d)"),
    "EMAIL": re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I),
    "TELEFONE": re.compile(r"(?<!\d)(?:\+?55\s*)?(?:\(?\d{2}\)?\s*)?9?\d{4}[-.\s]?\d{4}(?!\d)"),
    "CEP": re.compile(r"(?<!\d)\d{5}-?\d{3}(?!\d)"),
    "IDENTIDADE": re.compile(
        r"\b(?:RG|identidade|matr[íi]cula(?:\s+militar)?|SARAM)\s*(?:n[ºo°.]*)?\s*[:.-]?\s*[A-Z0-9.-]{5,}\b",
        re.I,
    ),
}

# Somente papéis privados. Relator, revisor, magistrados, MP e defesa não entram.
PRIVATE_ROLES = (
    r"acusad[oa]|r[ée]u|apelante|apelad[oa]|desertor|denunciad[oa]|sentenciad[oa]|"
    r"testemunha|ofendid[oa]|v[íi]tima|genitor(?:a)?|m[ãa]e|pai|irm[ãa]o|companheir[oa]|c[ôo]njuge"
)
NAME_WORD = r"(?:[A-ZÁÀÂÃÉÈÊÍÌÎÓÒÔÕÚÙÛÇ][A-ZÁÀÂÃÉÈÊÍÌÎÓÒÔÕÚÙÛÇa-záàâãéèêíìîóòôõúùûç'’-]+|[A-ZÁÉÍÓÚÇ]\.)"
ROLE_NAME_PATTERN = re.compile(
    rf"\b(?i:{PRIVATE_ROLES})\b\s*(?i:(?:devidamente\s+qualificad[oa]|de\s+nome|denominad[oa]|chamad[oa])?\s*(?:n[ºo°.]*)?\s*[:,-]?\s*)"
    rf"(?P<name>{NAME_WORD}(?:\s+(?:d[aeo]s?|e)\s+|\s+){NAME_WORD}(?:(?:\s+(?:d[aeo]s?|e)\s+|\s+){NAME_WORD}){{0,4}})",
)

# Cabeçalhos dos acórdãos identificam a parte pelo rótulo processual. Não se
# incluem polos que frequentemente são ocupados pelo MPM/União (p.ex.
# EMBARGADO), para evitar a anonimização de instituições públicas.
PRIVATE_PARTY_LABELS = (
    r"ACUSAD[OA]|R[ÉE]U|EMBARGANTE|APELANTE|RECORRENTE|PACIENTE|"
    r"IMPETRANTE|DENUNCIAD[OA]|SENTENCIAD[OA]"
)
PARTY_LABEL_PATTERN = re.compile(
    rf"(?im)^\s*(?P<label>{PRIVATE_PARTY_LABELS})\s*:\s*"
    rf"(?P<name>{NAME_WORD}(?:[ \t]+(?:d[aeo]s?|e)[ \t]+|[ \t]+){NAME_WORD}"
    rf"(?:(?:[ \t]+(?:d[aeo]s?|e)[ \t]+|[ \t]+){NAME_WORD}){{0,4}})"
)
PUBLIC_INSTITUTION_PREFIXES = (
    "MINISTERIO PUBLICO", "DEFENSORIA PUBLICA", "UNIAO", "JUSTICA PUBLICA",
    "SUPERIOR TRIBUNAL", "EXERCITO BRASILEIRO", "MARINHA DO BRASIL",
    "FORCA AEREA BRASILEIRA",
)


def _canonical(value: str) -> str:
    return " ".join(value.casefold().split())


def _fingerprint(value: str, document_uuid: str) -> str:
    return hashlib.sha256((document_uuid + "\0" + _canonical(value)).encode("utf-8")).hexdigest()


def find_sensitive_values(page_texts: list[str]) -> list[dict]:
    """Detecta candidatos. Nomes exigem marcador contextual de papel privado."""
    found: dict[tuple[str, str], dict] = {}
    for page_no, text in enumerate(page_texts, 1):
        for category, pattern in IDENTIFIER_PATTERNS.items():
            for match in pattern.finditer(text):
                value = match.group(0).strip()
                key = (category, _canonical(value))
                item = found.setdefault(key, {"category": category, "value": value,
                    "pages_detected": set(), "method": "REGEX", "confidence": 0.99})
                item["pages_detected"].add(page_no)
        for match in ROLE_NAME_PATTERN.finditer(text):
            value = match.group("name").strip(" ,.;:-")
            if norm_without_accents(value).startswith(PUBLIC_INSTITUTION_PREFIXES):
                continue
            # Barreira contra fragmentos de fundamentação capturados como nomes.
            if len(value.split()) < 2 or len(value) > 100:
                continue
            key = ("PESSOA", _canonical(value))
            item = found.setdefault(key, {"category": "PESSOA", "value": value,
                "pages_detected": set(), "method": "REGEX_CONTEXTO_PAPEL", "confidence": 0.82})
            item["pages_detected"].add(page_no)
        for match in PARTY_LABEL_PATTERN.finditer(text):
            value = match.group("name").strip(" ,.;:-")
            if norm_without_accents(value).startswith(PUBLIC_INSTITUTION_PREFIXES):
                continue
            key = ("PESSOA", _canonical(value))
            item = found.setdefault(key, {"category": "PESSOA", "value": value,
                "pages_detected": set(), "method": "REGEX_CABECALHO_PARTE", "confidence": 0.97})
            item["pages_detected"].add(page_no)
    for item in found.values():
        item["pages_detected"] = sorted(item["pages_detected"])
        item["search_values"] = [item["value"]]
        if item["category"] == "PESSOA":
            first_name = item["value"].split()[0]
            # Referências abreviadas podem reidentificar a parte. A barreira de
            # sete caracteres evita apagar prenomes comuns como João ou José.
            if len(first_name) >= 7:
                item["search_values"].append(first_name)
    return list(found.values())


def norm_without_accents(value: str) -> str:
    import unicodedata
    decomposed = unicodedata.normalize("NFKD", value)
    return " ".join("".join(c for c in decomposed if not unicodedata.combining(c)).upper().split())


def anonymize_pdf(source: Path, target: Path, document_uuid: str) -> dict:
    """Cria PDF redigido; não altera ``source`` e não guarda os valores em claro."""
    doc = fitz.open(source)
    page_texts = [page.get_text("text", sort=True) for page in doc]
    candidates = find_sensitive_values(page_texts)
    counters: defaultdict[str, int] = defaultdict(int)
    aliases: dict[tuple[str, str], str] = {}
    audit_items = []
    covered: defaultdict[int, list] = defaultdict(list)

    for item in sorted(candidates, key=lambda x: (x["category"], _canonical(x["value"]))):
        key = (item["category"], _canonical(item["value"]))
        counters[item["category"]] += 1
        aliases[key] = f"[{item['category']}_{counters[item['category']]:03d}]"

    # Nomes completos têm precedência sobre variantes contextuais menores. Isso
    # evita que uma anotação parcial ocupe a área e impeça a redação integral.
    ordered_candidates = sorted(
        candidates,
        key=lambda x: (x["category"] != "PESSOA", -len(x["value"]), -x["confidence"]),
    )
    for item in ordered_candidates:
        value = item["value"]
        replacement = aliases[(item["category"], _canonical(value))]
        redacted_pages = []
        occurrences = 0
        # Uma entidade descoberta em qualquer página é removida do documento todo.
        for page_no, page in enumerate(doc, 1):
            page_had_redaction = False
            for search_value in sorted(item.get("search_values", [value]), key=len, reverse=True):
                rects = page.search_for(search_value, quads=False)
                for rect in rects:
                    # Evita anotações sobrepostas entre nome completo e prenome.
                    if any(rect.intersects(previous) for previous in covered[page_no]):
                        continue
                    covered[page_no].append(rect)
                    page_had_redaction = True
                    occurrences += 1
                    page.add_redact_annot(rect, text=replacement, fontname="helv", fontsize=7,
                                          fill=(1, 1, 1), text_color=(0, 0, 0), cross_out=False)
            if page_had_redaction:
                redacted_pages.append(page_no)
        audit_items.append({
            "category": item["category"], "replacement": replacement,
            "value_fingerprint": _fingerprint(value, document_uuid),
            "pages_detected": item["pages_detected"], "pages_redacted": redacted_pages,
            "occurrences_redacted": occurrences, "method": item["method"],
            "confidence": item["confidence"], "review_status": "NAO_REVISADO",
        })

    for page in doc:
        page.apply_redactions(images=fitz.PDF_REDACT_IMAGE_PIXELS)
    doc.set_metadata({})
    try:
        doc.del_xml_metadata()
    except Exception:
        pass
    target.parent.mkdir(parents=True, exist_ok=True)
    doc.save(target, garbage=4, clean=True, deflate=True)
    doc.close()

    # Verificação: nenhum valor detectado pode permanecer no texto extraível.
    check = fitz.open(target)
    remaining = "\n".join(page.get_text("text") for page in check)
    check.close()
    leaks = [_fingerprint(i["value"], document_uuid) for i in candidates
             if any(v.casefold() in remaining.casefold() for v in i.get("search_values", [i["value"]]))]
    return {
        "source": str(source), "target": str(target), "document_uuid": document_uuid,
        "processed_at": datetime.now().isoformat(), "candidate_count": len(candidates),
        "redaction_count": sum(i["occurrences_redacted"] for i in audit_items),
        "unredacted_candidate_fingerprints": leaks, "items": audit_items,
    }


def anonymize_corpus(root: Path, con, force: bool = False, limit: int | None = None) -> None:
    output_dir = root / "data" / "anonymized" / "pdfs"
    reports_dir = root / "data" / "anonymized" / "reports"
    output_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    docs = con.execute("SELECT * FROM documents ORDER BY id" + (" LIMIT ?" if limit else ""), ((limit,) if limit else ())).fetchall()
    ok = failed = 0
    for row in docs:
        source = root / row["pdf_path"]
        target = output_dir / source.name
        source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
        existing = con.execute("SELECT * FROM anonymized_documents WHERE document_id=?", (row["id"],)).fetchone()
        if existing and existing["source_sha256"] == source_hash and target.exists() and not force:
            ok += 1
            continue
        try:
            report = anonymize_pdf(source, target, row["uuid"])
            report_path = reports_dir / f"{source.stem}.json"
            report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            target_hash = hashlib.sha256(target.read_bytes()).hexdigest()
            status = "OK" if not report["unredacted_candidate_fingerprints"] else "REVISAR_VAZAMENTO"
            con.execute("""INSERT INTO anonymized_documents(document_id,source_sha256,anonymized_pdf_path,
                anonymized_sha256,report_path,status,processed_at) VALUES(?,?,?,?,?,?,?)
                ON CONFLICT(document_id) DO UPDATE SET source_sha256=excluded.source_sha256,
                anonymized_pdf_path=excluded.anonymized_pdf_path,anonymized_sha256=excluded.anonymized_sha256,
                report_path=excluded.report_path,status=excluded.status,processed_at=excluded.processed_at""",
                (row["id"], source_hash, str(target.relative_to(root)), target_hash,
                 str(report_path.relative_to(root)), status, datetime.now().isoformat()))
            con.commit(); ok += 1
            print(f"anonimizado: {row['process_number']} ({report['redaction_count']} redações)")
        except Exception as exc:
            con.execute("""INSERT INTO anonymized_documents(document_id,source_sha256,status,processed_at)
                VALUES(?,?,?,?) ON CONFLICT(document_id) DO UPDATE SET status=excluded.status,
                processed_at=excluded.processed_at""",
                (row["id"], source_hash, f"ERRO:{type(exc).__name__}:{exc}", datetime.now().isoformat()))
            con.commit(); failed += 1
            print(f"erro ao anonimizar {row['process_number']}: {exc}")
    print(f"anonimização concluída: {ok} documentos; {failed} falhas")


def anonymize_unregistered_pdfs(root: Path, con, force: bool = False) -> None:
    """Processa PDFs presentes em ``raw/pdfs`` que não constem no banco.

    Isso evita que um download concluído imediatamente antes de uma interrupção
    fique fora do lote. O hash do PDF funciona como identificador técnico.
    """
    input_dir = root / "data" / "raw" / "pdfs"
    output_dir = root / "data" / "anonymized" / "pdfs"
    reports_dir = root / "data" / "anonymized" / "reports"
    output_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    processed = failed = 0
    registered_names = {
        Path(row[0]).name for row in con.execute("SELECT pdf_path FROM documents").fetchall()
    }
    for source in sorted(input_dir.glob("*.pdf")):
        if source.name in registered_names:
            continue
        target = output_dir / source.name
        if target.exists() and not force:
            continue
        try:
            source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
            report = anonymize_pdf(source, target, source_hash)
            report["database_status"] = "PDF_NAO_REGISTRADO_NO_BANCO"
            report_path = reports_dir / f"{source.stem}.json"
            report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            processed += 1
            print(f"anonimizado sem registro no banco: {source.name} ({report['redaction_count']} redações)")
        except Exception as exc:
            failed += 1
            print(f"erro no PDF sem registro {source.name}: {type(exc).__name__}: {exc}")
    if processed or failed:
        print(f"PDFs adicionais: {processed} processados; {failed} falhas")


def main() -> None:
    parser = argparse.ArgumentParser(description="Anonimiza os PDFs oficiais sem sobrescrever os originais")
    parser.add_argument("--force", action="store_true", help="regenera cópias já existentes")
    parser.add_argument("--limit", type=int, help="limita os documentos registrados (uso em testes)")
    args = parser.parse_args()
    from rag_stm import ROOT, db_conn
    con = db_conn()
    anonymize_corpus(ROOT, con, force=args.force, limit=args.limit)
    if args.limit is None:
        anonymize_unregistered_pdfs(ROOT, con, force=args.force)


if __name__ == "__main__":
    main()
