from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import sqlite3
import sys
import time
import unicodedata
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urljoin, urlparse, urlunparse

import fitz
import joblib
import networkx as nx
import numpy as np
import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import Normalizer

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RAW = DATA / "raw"
PDFS = RAW / "pdfs"
PAGES = RAW / "search_pages"
DB = DATA / "stm_rag.sqlite3"
URL_FILE = ROOT / "jurisprudência_desercao_exculpante.url"
START_DATE = datetime(2021, 1, 1)
END_DATE = datetime(2025, 12, 31)
MAX_TOKENS = 500
OVERLAP_TOKENS = 75
USER_AGENT = "STM-RAG-academico/1.0 (pesquisa; contato no short paper)"


def ensure_dirs() -> None:
    for p in (DATA, RAW, PDFS, PAGES):
        p.mkdir(parents=True, exist_ok=True)


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s or "")
    return " ".join("".join(c for c in s if not unicodedata.combining(c)).upper().split())


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def parse_date(s: str | None) -> datetime | None:
    if not s:
        return None
    m = re.search(r"(\d{2})/(\d{2})/(\d{4})", s)
    return datetime.strptime(m.group(0), "%d/%m/%Y") if m else None


def db_conn() -> sqlite3.Connection:
    ensure_dirs()
    con = sqlite3.connect(DB)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys=ON")
    con.execute("PRAGMA journal_mode=WAL")
    con.executescript(
        """
        CREATE TABLE IF NOT EXISTS documents(
          id INTEGER PRIMARY KEY, uuid TEXT UNIQUE NOT NULL, process_number TEXT,
          case_class TEXT, rapporteur TEXT, reviewer TEXT, subjects TEXT,
          filing_date TEXT, judgment_date TEXT, publication_date TEXT,
          summary TEXT, source_search_url TEXT, official_pdf_url TEXT,
          pdf_path TEXT, pdf_sha256 TEXT, page_count INTEGER, extraction_status TEXT,
          collected_at TEXT NOT NULL, metadata_json TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS collection_log(
          id INTEGER PRIMARY KEY, uuid TEXT, process_number TEXT, decision TEXT NOT NULL,
          reason TEXT NOT NULL, source_page INTEGER, logged_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS anonymized_documents(
          document_id INTEGER PRIMARY KEY REFERENCES documents(id),
          source_sha256 TEXT NOT NULL, anonymized_pdf_path TEXT,
          anonymized_sha256 TEXT, report_path TEXT, status TEXT NOT NULL,
          processed_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS chunks(
          id INTEGER PRIMARY KEY, document_id INTEGER NOT NULL REFERENCES documents(id),
          chunk_key TEXT UNIQUE NOT NULL, section TEXT NOT NULL, ordinal INTEGER NOT NULL,
          page_start INTEGER, page_end INTEGER, token_count INTEGER NOT NULL,
          text TEXT NOT NULL, text_sha256 TEXT NOT NULL
        );
        CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
          text, content='chunks', content_rowid='id', tokenize='unicode61 remove_diacritics 2'
        );
        CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
          INSERT INTO chunks_fts(rowid,text) VALUES(new.id,new.text);
        END;
        CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
          INSERT INTO chunks_fts(chunks_fts,rowid,text) VALUES('delete',old.id,old.text);
        END;
        CREATE TABLE IF NOT EXISTS entities(
          id INTEGER PRIMARY KEY, kind TEXT NOT NULL, canonical TEXT NOT NULL,
          label TEXT NOT NULL, UNIQUE(kind,canonical)
        );
        CREATE TABLE IF NOT EXISTS relations(
          id INTEGER PRIMARY KEY, document_id INTEGER NOT NULL REFERENCES documents(id),
          chunk_id INTEGER REFERENCES chunks(id), source_entity_id INTEGER NOT NULL REFERENCES entities(id),
          relation TEXT NOT NULL, target_entity_id INTEGER NOT NULL REFERENCES entities(id),
          evidence TEXT NOT NULL, method TEXT NOT NULL, confidence REAL NOT NULL,
          review_status TEXT NOT NULL DEFAULT 'NAO_REVISADO',
          UNIQUE(document_id,chunk_id,source_entity_id,relation,target_entity_id)
        );
        CREATE TABLE IF NOT EXISTS index_metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL);
        """
    )
    return con


def seed_url() -> str:
    text = URL_FILE.read_text(encoding="utf-8-sig", errors="replace")
    for line in text.splitlines():
        if line.startswith("URL="):
            return line[4:].strip()
    raise RuntimeError(f"URL ausente em {URL_FILE}")


def page_url(base: str, start: int, rows: int = 25) -> str:
    u = urlparse(base)
    q = dict(parse_qsl(u.query, keep_blank_values=True))
    q.update(start=str(start), rows=str(rows))
    return urlunparse(u._replace(query=urlencode(q)))


def field(text: str, label: str) -> str | None:
    pat = rf"{re.escape(label)}\s*:\s*(.+?)(?=\s+(?:Relator\(a\)|Revisor\(a\)|Assuntos|Data de Autuação|Data de Julgamento|Data de Publicação|EMENTA)\s*:|$)"
    m = re.search(pat, text, flags=re.I | re.S)
    return " ".join(m.group(1).split()) if m else None


def parse_results(page_html: str, source_url: str) -> list[dict]:
    soup = BeautifulSoup(page_html, "html.parser")
    out = []
    for panel in soup.select("div.panel.panel-default"):
        button = panel.find("button", attrs={"title": re.compile("Inteiro Teor", re.I)})
        if not button:
            continue
        onclick = button.get("onclick", "")
        murl = re.search(r"openInteiroTeor\('([^']+)'\)", onclick)
        if not murl:
            continue
        pdf_url = html.unescape(murl.group(1))
        uuid = dict(parse_qsl(urlparse(pdf_url).query)).get("uuid")
        text = " ".join(panel.get_text(" ", strip=True).split())
        proc = re.search(r"([A-ZÁÉÍÓÚÇ ]+?)\s+N[.º°]*\s*([0-9]{7}-[0-9]{2}\.[0-9]{4}\.[0-9]\.[0-9]{2}\.[0-9]{4})", text)
        process_number = proc.group(2) if proc else None
        case_class = proc.group(1).strip() if proc else None
        summary = field(text, "EMENTA") or (text[:8000] if text else None)
        out.append({
            "uuid": uuid, "process_number": process_number, "case_class": case_class,
            "rapporteur": field(text, "Relator(a)"), "reviewer": field(text, "Revisor(a)"),
            "subjects": field(text, "Assuntos"), "filing_date": field(text, "Data de Autuação"),
            "judgment_date": field(text, "Data de Julgamento"),
            "publication_date": field(text, "Data de Publicação"), "summary": summary,
            "official_pdf_url": pdf_url, "source_search_url": source_url,
        })
    return out


def eligible(d: dict) -> tuple[bool, str]:
    dt = parse_date(d.get("publication_date")) or parse_date(d.get("judgment_date"))
    if not dt or not START_DATE <= dt <= END_DATE:
        return False, "fora_do_periodo_2021_2025"
    hay = norm(" ".join(str(d.get(k) or "") for k in ("subjects", "summary")))
    if "DESERCAO" not in hay:
        return False, "sem_desercao_nos_metadados_ou_ementa"
    if "ESTADO DE NECESSIDADE" not in hay and "INEXIGIBILIDADE DE CONDUTA" not in hay:
        return False, "sem_exculpante_ou_inexigibilidade_nos_metadados_ou_ementa"
    return True, "incluido"


def collect(max_pages: int | None = None, delay: float = 0.25) -> None:
    ensure_dirs()
    con = db_conn()
    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT
    retry = Retry(total=5, connect=5, read=5, backoff_factor=1.0,
                  status_forcelist=(429, 500, 502, 503, 504),
                  allowed_methods=frozenset(("GET",)))
    session.mount("https://", HTTPAdapter(max_retries=retry))
    base = seed_url()
    starts = list(range(0, 225, 25))
    if max_pages:
        starts = starts[:max_pages]
    included = excluded = 0
    for start in starts:
        url = page_url(base, start)
        response = session.get(url, timeout=60)
        response.raise_for_status()
        page_file = PAGES / f"results_{start:04d}.html"
        page_file.write_bytes(response.content)
        results = parse_results(response.text, url)
        if not results:
            break
        for d in results:
            ok, reason = eligible(d)
            exists = con.execute("SELECT 1 FROM collection_log WHERE uuid=? AND decision=?", (d["uuid"], "INCLUIDO" if ok else "EXCLUIDO")).fetchone()
            if not exists:
                con.execute("INSERT INTO collection_log(uuid,process_number,decision,reason,source_page,logged_at) VALUES(?,?,?,?,?,?)",
                    (d["uuid"], d["process_number"], "INCLUIDO" if ok else "EXCLUIDO", reason, start, datetime.now().isoformat()))
            if not ok:
                excluded += 1
                continue
            included += 1
            pdf_path = PDFS / f"{d['process_number'] or d['uuid']}.pdf"
            if not pdf_path.exists():
                try:
                    pdf = session.get(d["official_pdf_url"], timeout=(20, 120))
                    pdf.raise_for_status()
                    if not pdf.content.startswith(b"%PDF"):
                        raise RuntimeError("resposta_nao_pdf")
                    pdf_path.write_bytes(pdf.content)
                    time.sleep(delay)
                except Exception as exc:
                    con.execute("INSERT INTO collection_log(uuid,process_number,decision,reason,source_page,logged_at) VALUES(?,?,?,?,?,?)",
                        (d["uuid"], d["process_number"], "FALHA_DOWNLOAD", f"{type(exc).__name__}:{exc}", start, datetime.now().isoformat()))
                    con.commit()
                    print(f"aviso: falha no PDF {d['process_number']}: {type(exc).__name__}", file=sys.stderr)
                    continue
            raw = pdf_path.read_bytes()
            meta = json.dumps(d, ensure_ascii=False)
            con.execute("""INSERT INTO documents(uuid,process_number,case_class,rapporteur,reviewer,subjects,
                filing_date,judgment_date,publication_date,summary,source_search_url,official_pdf_url,pdf_path,
                pdf_sha256,collected_at,metadata_json,extraction_status)
                VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(uuid) DO UPDATE SET process_number=excluded.process_number, subjects=excluded.subjects,
                summary=excluded.summary, pdf_path=excluded.pdf_path, pdf_sha256=excluded.pdf_sha256,
                metadata_json=excluded.metadata_json""",
                (d["uuid"], d["process_number"], d["case_class"], d["rapporteur"], d["reviewer"], d["subjects"],
                 d["filing_date"], d["judgment_date"], d["publication_date"], d["summary"], d["source_search_url"],
                 d["official_pdf_url"], str(pdf_path.relative_to(ROOT)), sha256_bytes(raw), datetime.now().isoformat(), meta, "PENDENTE"))
        con.commit()
        print(f"página start={start}: {len(results)} resultados")
    print(f"coleta concluída: {included} inclusões observadas; {excluded} exclusões observadas")


SECTION_PATTERNS = [
    ("EMENTA", r"^\s*EMENTA\b"), ("RELATORIO", r"^\s*RELAT[ÓO]RIO\b"),
    ("VOTO", r"^\s*VOTO\b"), ("DISPOSITIVO", r"^\s*(?:DISPOSITIVO|AC[ÓO]RD[ÃA]O)\b"),
]


def page_sections(doc: fitz.Document) -> list[tuple[int, str, str]]:
    current = "OUTROS"
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text("text", sort=True)
        top = "\n".join(text.splitlines()[:20])
        for section, pat in SECTION_PATTERNS:
            if re.search(pat, top, flags=re.I | re.M):
                current = section
                break
        pages.append((i + 1, current, text))
    return pages


def tokens(text: str) -> list[str]:
    return re.findall(r"\S+", text)


def make_chunks(pages: list[tuple[int, str, str]]) -> list[dict]:
    grouped: list[tuple[str, list[tuple[int, str]]]] = []
    for page, section, text in pages:
        if grouped and grouped[-1][0] == section:
            grouped[-1][1].append((page, text))
        else:
            grouped.append((section, [(page, text)]))
    out = []
    ordinal = 0
    for section, items in grouped:
        full = "\n".join(t for _, t in items).strip()
        if not full:
            continue
        words = tokens(full)
        windows = [(0, len(words))] if section in {"EMENTA", "DISPOSITIVO"} else [
            (i, min(i + MAX_TOKENS, len(words))) for i in range(0, len(words), MAX_TOKENS - OVERLAP_TOKENS)
        ]
        for a, b in windows:
            if a >= b:
                continue
            chunk_text = " ".join(words[a:b])
            # Aproximação conservadora de páginas dentro do grupo.
            frac_a, frac_b = a / max(1, len(words)), b / max(1, len(words))
            p0 = items[min(len(items)-1, int(frac_a * len(items)))][0]
            p1 = items[min(len(items)-1, max(0, int(np.ceil(frac_b * len(items))) - 1))][0]
            ordinal += 1
            out.append({"section": section, "ordinal": ordinal, "page_start": p0, "page_end": p1,
                        "token_count": b-a, "text": chunk_text})
    return out


def chunk_documents() -> None:
    con = db_conn()
    docs = con.execute("SELECT * FROM documents ORDER BY id").fetchall()
    total = 0
    for d in docs:
        anon = con.execute("SELECT * FROM anonymized_documents WHERE document_id=? AND status='OK'", (d["id"],)).fetchone()
        path = ROOT / (anon["anonymized_pdf_path"] if anon else d["pdf_path"])
        try:
            pdf = fitz.open(path)
            chunks = make_chunks(page_sections(pdf))
            con.execute("DELETE FROM chunks WHERE document_id=?", (d["id"],))
            for c in chunks:
                h = sha256_bytes(c["text"].encode("utf-8"))
                key = f"{d['uuid']}:{c['ordinal']:04d}:{h[:12]}"
                con.execute("INSERT INTO chunks(document_id,chunk_key,section,ordinal,page_start,page_end,token_count,text,text_sha256) VALUES(?,?,?,?,?,?,?,?,?)",
                    (d["id"], key, c["section"], c["ordinal"], c["page_start"], c["page_end"], c["token_count"], c["text"], h))
            con.execute("UPDATE documents SET page_count=?, extraction_status=? WHERE id=?", (len(pdf), "OK" if chunks else "SEM_TEXTO", d["id"]))
            total += len(chunks)
        except Exception as exc:
            con.execute("UPDATE documents SET extraction_status=? WHERE id=?", (f"ERRO:{type(exc).__name__}", d["id"]))
        con.commit()
    print(f"segmentação concluída: {len(docs)} documentos, {total} chunks")


ENTITY_PATTERNS = {
    "PROCESSO": re.compile(r"\b\d{7}-\d{2}\.\d{4}\.\d\.\d{2}\.\d{4}\b"),
    "NORMA": re.compile(r"\b(?:art(?:igo)?\.?\s*\d+[ºo]?(?:[,.-]\s*(?:§\s*\d+[ºo]?|[IVXLCDM]+|[a-z]))?(?:\s+do)?\s+(?:CPM|CPPM|Código Penal Militar|Código de Processo Penal Militar))", re.I),
    "SUMULA": re.compile(r"\bS[úu]mula\s+(?:n[º°.]*\s*)?\d+\b", re.I),
}


def entity(con: sqlite3.Connection, kind: str, label: str) -> int:
    canonical = norm(label)
    con.execute("INSERT OR IGNORE INTO entities(kind,canonical,label) VALUES(?,?,?)", (kind, canonical, label))
    return con.execute("SELECT id FROM entities WHERE kind=? AND canonical=?", (kind, canonical)).fetchone()[0]


def build_graph() -> None:
    con = db_conn()
    con.execute("DELETE FROM relations")
    con.execute("DELETE FROM entities")
    docs = con.execute("SELECT * FROM documents ORDER BY id").fetchall()
    for d in docs:
        source = entity(con, "ACORDAO", d["process_number"] or d["uuid"])
        for c in con.execute("SELECT * FROM chunks WHERE document_id=?", (d["id"],)):
            for kind, pattern in ENTITY_PATTERNS.items():
                for m in pattern.finditer(c["text"]):
                    label = m.group(0)
                    if kind == "PROCESSO" and label == d["process_number"]:
                        continue
                    target = entity(con, kind, label)
                    relation = {"PROCESSO": "CITA", "NORMA": "APLICA_OU_MENCIONA", "SUMULA": "CITA"}[kind]
                    a, b = max(0, m.start()-120), min(len(c["text"]), m.end()+180)
                    evidence = c["text"][a:b]
                    con.execute("INSERT OR IGNORE INTO relations(document_id,chunk_id,source_entity_id,relation,target_entity_id,evidence,method,confidence) VALUES(?,?,?,?,?,?,?,?)",
                        (d["id"], c["id"], source, relation, target, evidence, "REGEX", 0.95))
    con.commit()
    g = nx.MultiDiGraph()
    for e in con.execute("SELECT * FROM entities"):
        g.add_node(str(e["id"]), kind=e["kind"], label=e["label"], canonical=e["canonical"])
    for r in con.execute("SELECT * FROM relations"):
        g.add_edge(str(r["source_entity_id"]), str(r["target_entity_id"]), relation=r["relation"],
                   document_id=r["document_id"], chunk_id=r["chunk_id"], confidence=r["confidence"], review_status=r["review_status"])
    nx.write_graphml(g, DATA / "graph.graphml")
    print(f"grafo concluído: {g.number_of_nodes()} nós, {g.number_of_edges()} relações")


@dataclass
class VectorIndex:
    backend: str
    ids: np.ndarray
    vectors: np.ndarray
    encoder: object


def build_vectors() -> None:
    con = db_conn()
    rows = con.execute("SELECT id,text FROM chunks ORDER BY id").fetchall()
    if not rows:
        raise RuntimeError("Não há chunks. Execute o pipeline.")
    texts = [r["text"] for r in rows]
    ids = np.array([r["id"] for r in rows], dtype=np.int64)
    backend = "lsa_tfidf"
    try:
        from sentence_transformers import SentenceTransformer
        model_name = "intfloat/multilingual-e5-small"
        encoder = SentenceTransformer(model_name)
        vectors = encoder.encode(["passage: " + t for t in texts], normalize_embeddings=True, show_progress_bar=True)
        joblib.dump({"backend": "sentence_transformers", "model_name": model_name}, DATA / "vectorizer.joblib")
        backend = f"sentence_transformers:{model_name}"
    except Exception as exc:
        print(f"aviso: embeddings neurais indisponíveis ({exc}); usando LSA local", file=sys.stderr)
        tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_df=0.98, sublinear_tf=True, strip_accents="unicode")
        matrix = tfidf.fit_transform(texts)
        n = max(2, min(256, matrix.shape[0]-1, matrix.shape[1]-1))
        svd = TruncatedSVD(n_components=n, random_state=42)
        normalizer = Normalizer(copy=False)
        vectors = normalizer.fit_transform(svd.fit_transform(matrix)).astype(np.float32)
        joblib.dump({"backend": "lsa_tfidf", "tfidf": tfidf, "svd": svd, "normalizer": normalizer}, DATA / "vectorizer.joblib")
    np.savez_compressed(DATA / "vectors.npz", ids=ids, vectors=np.asarray(vectors, dtype=np.float32))
    con.execute("INSERT OR REPLACE INTO index_metadata(key,value) VALUES('vector_backend',?)", (backend,))
    con.execute("INSERT OR REPLACE INTO index_metadata(key,value) VALUES('indexed_at',?)", (datetime.now().isoformat(),))
    con.commit()
    print(f"índice vetorial concluído: {len(ids)} chunks; backend={backend}")


def lexical_search(con: sqlite3.Connection, query: str, k: int) -> list[tuple[int, float]]:
    terms = re.findall(r"[\wÀ-ÿ]+", query)
    fts = " OR ".join(f'"{t}"' for t in terms if len(t) > 2)
    if not fts:
        return []
    rows = con.execute("SELECT rowid,bm25(chunks_fts) score FROM chunks_fts WHERE chunks_fts MATCH ? ORDER BY score LIMIT ?", (fts, k)).fetchall()
    return [(r[0], -float(r[1])) for r in rows]


def vector_search(query: str, k: int) -> list[tuple[int, float]]:
    meta = joblib.load(DATA / "vectorizer.joblib")
    arr = np.load(DATA / "vectors.npz")
    vectors, ids = arr["vectors"], arr["ids"]
    if meta["backend"] == "sentence_transformers":
        from sentence_transformers import SentenceTransformer
        q = SentenceTransformer(meta["model_name"]).encode(["query: " + query], normalize_embeddings=True)[0]
    else:
        q = meta["normalizer"].transform(meta["svd"].transform(meta["tfidf"].transform([query])))[0]
    scores = vectors @ np.asarray(q, dtype=np.float32)
    top = np.argsort(-scores)[:k]
    return [(int(ids[i]), float(scores[i])) for i in top]


def rrf(*rankings: list[tuple[int, float]], k: int = 60) -> list[tuple[int, float]]:
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, (cid, _) in enumerate(ranking, 1):
            scores[cid] = scores.get(cid, 0) + 1 / (k + rank)
    return sorted(scores.items(), key=lambda x: -x[1])


def search(query: str, mode: str, k: int) -> None:
    con = db_conn()
    lex = lexical_search(con, query, max(k*4, 20))
    vec = vector_search(query, max(k*4, 20)) if mode in {"vector", "hybrid"} else []
    ranked = lex if mode == "lexical" else vec if mode == "vector" else rrf(lex, vec)
    for rank, (cid, score) in enumerate(ranked[:k], 1):
        r = con.execute("""SELECT c.*,d.process_number,d.official_pdf_url,d.judgment_date,d.publication_date
          FROM chunks c JOIN documents d ON d.id=c.document_id WHERE c.id=?""", (cid,)).fetchone()
        excerpt = r["text"][:700].replace("\n", " ")
        print(f"\n[{rank}] score={score:.5f} | {r['process_number']} | {r['section']} | pp. {r['page_start']}-{r['page_end']}")
        print(excerpt + ("…" if len(r["text"]) > 700 else ""))
        print(r["official_pdf_url"])


def stats() -> None:
    con = db_conn()
    for label, sql in {
        "documentos": "SELECT count(*) FROM documents", "chunks": "SELECT count(*) FROM chunks",
        "entidades": "SELECT count(*) FROM entities", "relações": "SELECT count(*) FROM relations",
        "excluídos": "SELECT count(*) FROM collection_log WHERE decision='EXCLUIDO'",
    }.items():
        print(f"{label}: {con.execute(sql).fetchone()[0]}")
    print("status de extração:")
    for r in con.execute("SELECT extraction_status,count(*) n FROM documents GROUP BY extraction_status"):
        print(f"  {r[0]}: {r[1]}")


def pipeline(args) -> None:
    collect(args.max_pages, args.delay)
    from anonymize_pdfs import anonymize_corpus
    anonymize_corpus(ROOT, db_conn())
    chunk_documents()
    build_graph()
    build_vectors()
    stats()


def main() -> None:
    p = argparse.ArgumentParser(description="RAG + grafos para jurisprudência do STM")
    sub = p.add_subparsers(dest="command", required=True)
    pp = sub.add_parser("pipeline")
    pp.add_argument("--max-pages", type=int)
    pp.add_argument("--delay", type=float, default=0.25)
    pc = sub.add_parser("collect"); pc.add_argument("--max-pages", type=int); pc.add_argument("--delay", type=float, default=0.25)
    pa = sub.add_parser("anonymize"); pa.add_argument("--force", action="store_true"); pa.add_argument("--limit", type=int)
    sub.add_parser("chunk"); sub.add_parser("graph"); sub.add_parser("vectors"); sub.add_parser("stats")
    ps = sub.add_parser("search"); ps.add_argument("query"); ps.add_argument("--mode", choices=["lexical","vector","hybrid"], default="hybrid"); ps.add_argument("-k", type=int, default=5)
    args = p.parse_args()
    def run_anonymize(a):
        from anonymize_pdfs import anonymize_corpus
        anonymize_corpus(ROOT, db_conn(), force=a.force, limit=a.limit)
    {"pipeline": pipeline, "collect": lambda a: collect(a.max_pages,a.delay), "anonymize": run_anonymize, "chunk": lambda a: chunk_documents(),
     "graph": lambda a: build_graph(), "vectors": lambda a: build_vectors(), "stats": lambda a: stats(),
     "search": lambda a: search(a.query,a.mode,a.k)}[args.command](args)


if __name__ == "__main__":
    main()
