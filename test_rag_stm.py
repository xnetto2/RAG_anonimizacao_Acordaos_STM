import tempfile
import unittest

import fitz

import rag_stm
from anonymize_pdfs import find_sensitive_values


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


if __name__ == "__main__":
    unittest.main()
