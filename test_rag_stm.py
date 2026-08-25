import tempfile
import unittest

import pymupdf as fitz

import rag_stm
from anonymize_pdfs import find_sensitive_values
from normalization import canonical_entity


class PipelineTests(unittest.TestCase):
    def test_parse_results_and_eligibility(self):
        page = """<div class='panel panel-default'><div>APELAÇÃO CRIMINAL N.º 7000000-00.2023.7.00.0000
        Relator(a): FULANO Revisor(a): BELTRANO Assuntos: DESERÇÃO Data de Julgamento: 10/05/2023
        Data de Publicação: 12/05/2023 EMENTA: DESERÇÃO. ESTADO DE NECESSIDADE EXCULPANTE.</div>
        <button title='Exibir Inteiro Teor' onclick="tracker.functions.openInteiroTeor('https://x.test/?uuid=abc')">Inteiro</button></div>"""
        rows = rag_stm.parse_results(page, "https://fonte")
        self.assertEqual(rows[0]["uuid"], "abc")
        self.assertTrue(rag_stm.eligible(rows[0])[0])

    def test_chunk_overlap_and_sections(self):
        pages = [(1, "EMENTA", "EMENTA " + "x " * 600), (2, "VOTO", "VOTO " + "y " * 1100)]
        chunks = rag_stm.make_chunks(pages)
        self.assertEqual(chunks[0]["section"], "EMENTA")
        self.assertGreaterEqual(len(chunks), 4)
        self.assertTrue(all(c["token_count"] <= rag_stm.MAX_TOKENS or c["section"] == "EMENTA" for c in chunks))

    def test_sensitive_detection_preserves_public_roles(self):
        text = "Acusado: João da Silva, CPF 123.456.789-00. Relator: Artur Vidigal de Oliveira."
        items = find_sensitive_values([text])
        values = {i["value"] for i in items}
        self.assertIn("João da Silva", values)
        self.assertIn("123.456.789-00", values)
        self.assertNotIn("Artur Vidigal de Oliveira", values)

    def test_party_header_detects_name_and_abbreviated_reference(self):
        text = ("EMBARGANTE: ALBERTINI FERNANDES SOUSA\n"
                "APELANTE: MINISTÉRIO PÚBLICO MILITAR\nRELATOR: CARLOS VUYK DE AQUINO")
        items = find_sensitive_values([text])
        person = next(i for i in items if i["value"] == "ALBERTINI FERNANDES SOUSA")
        self.assertIn("Albertini".casefold(), {v.casefold() for v in person["search_values"]})
        self.assertNotIn("CARLOS VUYK DE AQUINO", {i["value"] for i in items})
        self.assertNotIn("MINISTÉRIO PÚBLICO MILITAR", {i["value"] for i in items})


    def test_legal_normalization(self):
        self.assertEqual(canonical_entity("NORMA", "art. 39 do CPM"), "CPM:ARTIGO_39")
        self.assertEqual(canonical_entity("PROCESSO", "7000000-00.2023.7.00.0000"),
                         "PROCESSO:7000000-00.2023.7.00.0000")
        self.assertEqual(canonical_entity("SUMULA", "Súmula 3"), "STM:SUMULA_3")
        self.assertEqual(canonical_entity("SUMULA", "Súmula 3 do STM"), "STM:SUMULA_3")
        self.assertEqual(canonical_entity("SUMULA", "Súmula 3 do STF"), "STF:SUMULA_3")
        self.assertEqual(canonical_entity("SUMULA", "Súmula 231",
                         "pena mínima, conforme a Súmula 231 do STJ"), "STJ:SUMULA_231")
        self.assertEqual(canonical_entity("SUMULA", "Súmula 339",
                         "SUPREMO TRIBUNAL FEDERAL - SÚMULA 339/STF"), "STF:SUMULA_339")
        self.assertEqual(canonical_entity("SUMULA", "Súmula 12",
                         "Súmula 12 desta Corte Castrense"), "STM:SUMULA_12")

    def test_review_weights(self):
        self.assertEqual(rag_stm.REVIEW_WEIGHTS["CONFIRMADA"], 1.0)
        self.assertEqual(rag_stm.REVIEW_WEIGHTS["RECLASSIFICADA"], 1.0)
        self.assertEqual(rag_stm.REVIEW_WEIGHTS["NAO_REVISADO"], 0.25)
        self.assertEqual(rag_stm.REVIEW_WEIGHTS["REJEITADA"], 0.0)
        fused = dict(rag_stm.rrf_with_weighted_graph([], [], [(1, 1.0), (2, 0.25)]))
        self.assertGreater(fused[1], fused[2])


if __name__ == "__main__":
    unittest.main()
