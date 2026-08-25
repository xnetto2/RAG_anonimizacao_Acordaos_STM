"""Interface local para revisão humana das relações do grafo."""
from __future__ import annotations

from datetime import datetime
from rag_stm import db_conn

DECISIONS = ["CONFIRMADA", "RECLASSIFICADA", "REJEITADA", "EVIDENCIA_INSUFICIENTE"]
RELATION_TYPES = ["CITA", "MENCIONA", "APLICA", "AFASTA", "ADOTA_TESE", "DISTINGUE", "DIVERGE_DE", "SUPERA"]


def record_review(con, relation_id: int, decision: str, reviewed_relation: str,
                  reviewer: str, notes: str = "") -> None:
    if decision not in DECISIONS or reviewed_relation not in RELATION_TYPES:
        raise ValueError("Decisão ou relação inválida")
    row = con.execute("SELECT relation FROM relations WHERE id=?", (relation_id,)).fetchone()
    if not row:
        raise ValueError("Relação inexistente")
    previous = row["relation"]
    effective = reviewed_relation if decision == "RECLASSIFICADA" else previous
    con.execute("""INSERT INTO relation_review_events
      (relation_id,previous_relation,reviewed_relation,decision,reviewer,notes,reviewed_at)
      VALUES(?,?,?,?,?,?,?)""",
      (relation_id, previous, effective, decision, reviewer.strip(), notes.strip(), datetime.now().astimezone().isoformat()))
    con.execute("UPDATE relations SET relation=?,review_status=? WHERE id=?", (effective, decision, relation_id))
    con.commit()


def relation_query(status: str, kind: str, relation_type: str, process: str):
    where, params = [], []
    for value, column in ((status, "r.review_status"), (kind, "te.kind"), (relation_type, "r.relation")):
        if value != "TODOS":
            where.append(f"{column}=?"); params.append(value)
    if process.strip():
        where.append("d.process_number LIKE ?"); params.append(f"%{process.strip()}%")
    clause = " WHERE " + " AND ".join(where) if where else ""
    sql = """SELECT r.id,r.relation,r.review_status,r.confidence,r.method,r.evidence,
      d.process_number,d.official_pdf_url,c.section,c.page_start,c.page_end,c.text chunk_text,
      se.label source_label,te.kind target_kind,te.label target_label,te.canonical target_canonical
      FROM relations r JOIN documents d ON d.id=r.document_id LEFT JOIN chunks c ON c.id=r.chunk_id
      JOIN entities se ON se.id=r.source_entity_id JOIN entities te ON te.id=r.target_entity_id""" + clause + " ORDER BY r.id"
    return sql, params


def main() -> None:
    import streamlit as st
    st.set_page_config(page_title="Revisão do grafo STM", layout="wide")
    st.title("Revisão humana das relações do grafo")
    st.caption("Confirmada/reclassificada: peso 1,0; não revisada: 0,25; rejeitada/insuficiente: 0.")
    con = db_conn()
    counts = dict(con.execute("SELECT review_status,count(*) FROM relations GROUP BY review_status").fetchall())
    for col, state in zip(st.columns(5), ["NAO_REVISADO", "CONFIRMADA", "RECLASSIFICADA", "REJEITADA", "EVIDENCIA_INSUFICIENTE"]):
        col.metric(state.replace("_", " ").title(), counts.get(state, 0))
    with st.sidebar:
        st.header("Filtros")
        status = st.selectbox("Estado", ["NAO_REVISADO", "CONFIRMADA", "RECLASSIFICADA", "REJEITADA", "EVIDENCIA_INSUFICIENTE", "TODOS"])
        kind = st.selectbox("Entidade", ["TODOS", "PROCESSO", "NORMA", "SUMULA"])
        relation_type = st.selectbox("Relação", ["TODOS", "APLICA_OU_MENCIONA"] + RELATION_TYPES)
        process = st.text_input("Número do processo")
        reviewer = st.text_input("Revisor", value="Alexandre")
    sql, params = relation_query(status, kind, relation_type, process)
    rows = con.execute(sql, params).fetchall()
    if not rows:
        st.info("Nenhuma relação corresponde aos filtros."); return
    st.session_state.setdefault("review_index", 0)
    st.session_state.review_index = min(st.session_state.review_index, len(rows)-1)
    prev, nxt, counter = st.columns([1, 1, 5])
    if prev.button("← Anterior", disabled=st.session_state.review_index == 0):
        st.session_state.review_index -= 1; st.rerun()
    if nxt.button("Próxima →", disabled=st.session_state.review_index >= len(rows)-1):
        st.session_state.review_index += 1; st.rerun()
    counter.write(f"Relação {st.session_state.review_index + 1} de {len(rows)}")
    row = rows[st.session_state.review_index]
    left, right = st.columns([2, 3])
    with left:
        st.subheader(f"Processo {row['process_number']}")
        st.write(f"**Relação:** `{row['relation']}` → **{row['target_label']}**")
        st.code(row["target_canonical"])
        st.write(f"**Tipo:** {row['target_kind']} | **Método:** {row['method']} | **Confiança:** {row['confidence']:.2f}")
        st.write(f"**Seção:** {row['section']} | **Páginas:** {row['page_start']}–{row['page_end']}")
        if row["official_pdf_url"]:
            st.link_button("Abrir inteiro teor oficial", row["official_pdf_url"])
    with right:
        st.subheader("Evidência")
        st.info(row["evidence"])
        with st.expander("Chunk completo"):
            st.write(row["chunk_text"])
    with st.form("review_form", clear_on_submit=True):
        decision = st.radio("Decisão", DECISIONS, horizontal=True)
        current_index = RELATION_TYPES.index(row["relation"]) if row["relation"] in RELATION_TYPES else 0
        reviewed_relation = st.selectbox("Relação correta", RELATION_TYPES, index=current_index,
                                         disabled=decision != "RECLASSIFICADA")
        notes = st.text_area("Justificativa/observações")
        if st.form_submit_button("Registrar revisão", type="primary"):
            if not reviewer.strip():
                st.error("Informe o revisor.")
            else:
                record_review(con, row["id"], decision, reviewed_relation, reviewer, notes)
                st.success("Revisão registrada com histórico."); st.rerun()
    with st.expander("Histórico"):
        history = con.execute("SELECT * FROM relation_review_events WHERE relation_id=? ORDER BY id DESC", (row["id"],)).fetchall()
        st.dataframe([dict(item) for item in history], use_container_width=True)


if __name__ == "__main__":
    main()
