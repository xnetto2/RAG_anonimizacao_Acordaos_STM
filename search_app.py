"""Interface gráfica para consulta da jurisprudência indexada."""
from __future__ import annotations

from rag_stm import search_results


MODE_LABELS = {
    "Híbrida — texto, semântica e grafo": "hybrid",
    "Lexical — correspondência de termos": "lexical",
    "Vetorial — similaridade semântica": "vector",
}


def result_markdown(query: str, results: list[dict]) -> str:
    lines = [f"# Resultados da pesquisa\n\n**Argumento:** {query}\n"]
    for item in results:
        pages = f"{item['page_start']}–{item['page_end']}"
        lines.extend([
            f"## {item['rank']}. Processo {item['process_number']}",
            f"Seção: {item['section']} | páginas: {pages} | score: {item['score']:.5f}",
            "",
            item["text"].strip(),
            "",
            f"Inteiro teor: {item['official_pdf_url'] or 'não informado'}",
            "",
        ])
    return "\n".join(lines)


def main() -> None:
    import streamlit as st

    st.set_page_config(page_title="Consulta jurisprudencial — STM", page_icon="⚖️", layout="wide")
    st.title("Consulta jurisprudencial — STM")
    st.caption("Busca auditável por texto, similaridade semântica e relações revisadas do grafo.")

    with st.form("search_form"):
        query = st.text_area(
            "Argumento ou questão de pesquisa",
            placeholder="Ex.: A dificuldade financeira, sem prova documental, configura estado de necessidade exculpante?",
            height=110,
        )
        col_mode, col_k = st.columns([3, 1])
        mode_label = col_mode.selectbox("Modo de busca", list(MODE_LABELS))
        k = col_k.number_input("Resultados", min_value=1, max_value=30, value=10, step=1)
        submitted = st.form_submit_button("Pesquisar", type="primary", use_container_width=True)

    if submitted:
        if not query.strip():
            st.warning("Informe um argumento ou uma questão de pesquisa.")
        else:
            try:
                with st.spinner("Consultando o corpus…"):
                    st.session_state.search_query = query.strip()
                    st.session_state.search_mode = MODE_LABELS[mode_label]
                    st.session_state.search_results = search_results(
                        st.session_state.search_query, st.session_state.search_mode, int(k)
                    )
            except Exception as exc:
                st.error(f"Não foi possível executar a consulta: {exc}")

    results = st.session_state.get("search_results", [])
    effective_query = st.session_state.get("search_query", "")
    if not results:
        if submitted and query.strip():
            st.info("Nenhum resultado foi encontrado.")
        else:
            st.info("Digite o argumento acima para iniciar a pesquisa.")
        return

    st.success(f"{len(results)} resultados recuperados para: “{effective_query}”")
    sections = sorted({item["section"] for item in results})
    selected_sections = st.multiselect("Filtrar seções nos resultados", sections, default=sections)
    visible = [item for item in results if item["section"] in selected_sections]

    st.download_button(
        "Exportar resultados em Markdown",
        result_markdown(effective_query, visible),
        file_name="consulta_jurisprudencial_stm.md",
        mime="text/markdown",
    )

    for item in visible:
        pages = f"{item['page_start']}–{item['page_end']}"
        with st.container(border=True):
            st.subheader(f"{item['rank']}. Processo {item['process_number']}")
            meta, score, date = st.columns([3, 1, 1])
            meta.write(f"**Seção:** {item['section']} · **Páginas:** {pages}")
            score.metric("Score final", f"{item['score']:.5f}")
            date.write(f"**Julgamento:**  \n{item['judgment_date'] or 'não informado'}")
            excerpt = item["text"][:1200]
            st.write(excerpt + ("…" if len(item["text"]) > 1200 else ""))
            if item["official_pdf_url"]:
                st.link_button("Abrir inteiro teor oficial", item["official_pdf_url"])

            with st.expander("Ver texto completo e explicação do ranking"):
                st.write(item["text"])
                d1, d2, d3 = st.columns(3)
                d1.metric("Ranking lexical", item["lexical_rank"] or "—")
                d2.metric("Ranking vetorial", item["vector_rank"] or "—")
                d3.metric("Ranking do grafo", item["graph_rank"] or "—")
                if item["entities"]:
                    st.write("**Relações jurídicas identificadas no trecho**")
                    st.dataframe(item["entities"], use_container_width=True, hide_index=True)

    st.caption("Os resultados auxiliam a pesquisa e devem ser conferidos no inteiro teor oficial.")


if __name__ == "__main__":
    main()
