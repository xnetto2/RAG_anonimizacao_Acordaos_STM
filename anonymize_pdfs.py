"""Anonimização auditável dos acórdãos em PDF.

Os arquivos oficiais nunca são sobrescritos. A rotina cria cópias com redações
reais (remoção do conteúdo subjacente), elimina metadados e registra apenas a
impressão digital dos valores removidos no relatório público de auditoria.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import fitz


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
    rf"\b(?i:{PRIVATE_ROLES})\b\s*(?i:(?:de\s+nome|denominad[oa]|chamad[oa])?\s*(?:n[ºo°.]*)?\s*[:,-]?\s*)"
    rf"(?P<name>{NAME_WORD}(?:\s+(?:d[aeo]s?|e)\s+|\s+){NAME_WORD}(?:(?:\s+(?:d[aeo]s?|e)\s+|\s+){NAME_WORD}){{0,4}})",
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
            # Barreira contra fragmentos de fundamentação capturados como nomes.
            if len(value.split()) < 2 or len(value) > 100:
                continue
            key = ("PESSOA", _canonical(value))
            item = found.setdefault(key, {"category": "PESSOA", "value": value,
                "pages_detected": set(), "method": "REGEX_CONTEXTO_PAPEL", "confidence": 0.82})
            item["pages_detected"].add(page_no)
    for item in found.values():
        item["pages_detected"] = sorted(item["pages_detected"])
    return list(found.values())


def anonymize_pdf(source: Path, target: Path, document_uuid: str) -> dict:
    """Cria PDF redigido; não altera ``source`` e não guarda os valores em claro."""
    doc = fitz.open(source)
    page_texts = [page.get_text("text", sort=True) for page in doc]
    candidates = find_sensitive_values(page_texts)
    counters: defaultdict[str, int] = defaultdict(int)
    aliases: dict[tuple[str, str], str] = {}
    audit_items = []

    for item in sorted(candidates, key=lambda x: (x["category"], _canonical(x["value"]))):
        key = (item["category"], _canonical(item["value"]))
        counters[item["category"]] += 1
        aliases[key] = f"[{item['category']}_{counters[item['category']]:03d}]"

    for item in candidates:
        value = item["value"]
        replacement = aliases[(item["category"], _canonical(value))]
        redacted_pages = []
        occurrences = 0
        # Uma entidade descoberta em qualquer página é removida do documento todo.
        for page_no, page in enumerate(doc, 1):
            rects = page.search_for(value, quads=False)
            if not rects:
                continue
            redacted_pages.append(page_no)
            occurrences += len(rects)
            for rect in rects:
                page.add_redact_annot(rect, text=replacement, fontname="helv", fontsize=7,
                                      fill=(1, 1, 1), text_color=(0, 0, 0), cross_out=False)
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
    leaks = [_fingerprint(i["value"], document_uuid) for i in candidates if i["value"] in remaining]
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
