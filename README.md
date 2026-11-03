# Preentrega 4

Pinecone Serverless con embeddings locales de 384 dimensiones, ingesta por lotes y metadatos de texto, fuente, posición y categoría. Recuperación híbrida BM25 + vectorial mediante EnsembleRetriever, namespace propio y top-5.

```sh
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Completar PINECONE_API_KEY
python ingest.py
python evaluate.py
```

La ingesta crea el índice si no existe y verifica su dimensión. Los embeddings se calculan localmente con all-MiniLM-L6-v2, tanto al indexar como al consultar. No requiere créditos de OpenAI.

`golden_set.json` contiene cinco consultas con sus documentos relevantes. La evaluación imprime Precision@5 y Recall@5 y guarda el reporte en `metrics.json`. Cada consulta tiene una fuente relevante: un acierto equivale a Precision@5 de 0,2 y Recall@5 de 1.
