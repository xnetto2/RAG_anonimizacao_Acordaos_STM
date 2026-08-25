# Pesquisa jurisprudencial aumentada — STM

Protótipo auditável para acórdãos sobre deserção e estado de necessidade
exculpante. A solução coleta resultados e inteiros teores oficiais, segmenta os
documentos em **chunks jurídicos com proveniência**, cria um índice lexical,
embeddings vetoriais e um grafo de normas/precedentes.

## Execução

```powershell
python -m pip install -r requirements.txt
python rag_stm.py anonymize
python rag_stm.py chunk
python rag_stm.py pipeline
python rag_stm.py search "Quais provas são exigidas para o estado de necessidade exculpante?" --mode hybrid
python rag_stm.py stats
```

O pipeline é incremental: PDFs existentes e registros com o mesmo UUID não são
baixados novamente. Os artefatos são gravados em `data/`:

- `raw/pdfs/`: inteiros teores oficiais;
- `anonymized/pdfs/`: cópias com redação efetiva, usadas no chunking;
- `anonymized/reports/`: auditoria sem armazenamento dos valores em claro;
- `raw/search_pages/`: páginas HTML que comprovam a coleta;
- `stm_rag.sqlite3`: documentos, chunks, entidades, relações e FTS5;
- `vectorizer.joblib` e `vectors.npz`: índice vetorial local;
- `graph.graphml`: grafo interoperável.

## Critérios do corpus

- julgamento ou publicação entre 01.01.2021 e 31.12.2025;
- presença de deserção e de estado de necessidade/inexigibilidade de conduta;
- fonte oficial do STM;
- exclusões registradas na tabela `collection_log`.

O arquivo `.url` fornecido é a semente da coleta. Como a consulta também retorna
itens fora do recorte, os critérios acima são reaplicados localmente.

## Chunks

Cada chunk registra `document_id`, processo, seção jurídica, páginas inicial e
final, posição, hash, texto e contagem aproximada de tokens. Ementa e dispositivo
são preservados integralmente; relatório e voto são segmentados em até 500 tokens,
com sobreposição de 75 tokens. Nenhuma resposta deve ser tratada como substituta
da consulta ao inteiro teor.

## Modos de busca

- `lexical`: BM25 do SQLite FTS5;
- `vector`: embeddings do `sentence-transformers` quando disponível; caso
  contrário, LSA local explicitamente identificado no resultado;
- `hybrid`: fusão por Reciprocal Rank Fusion e expansão por entidades do grafo.

O comando retorna sempre processo, seção, páginas, URL oficial e excerto. A
geração textual por LLM foi deixada desacoplada: primeiro se valida a recuperação,
depois se conecta um modelo com obrigação de citar apenas as evidências retornadas.

## Anonimização

`python rag_stm.py anonymize` preserva os originais, remove CPF, e-mail,
telefone, CEP, identidade/matrícula e nomes encontrados em contexto de papéis
privados (acusado, réu, testemunha, familiares etc.). Relator, revisor e agentes
públicos não são removidos por padrão. A redação elimina o texto subjacente e os
metadados do PDF; não é uma tarja meramente visual. Como reconhecimento de nomes
pode produzir falsos positivos ou omissões, os relatórios ficam com estado
`NAO_REVISADO` até conferência humana.
