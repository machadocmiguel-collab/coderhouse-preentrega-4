import hashlib
import json
import os
import time
from pathlib import Path
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_text_splitters import RecursiveCharacterTextSplitter
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")
NAMESPACE = os.getenv("PINECONE_NAMESPACE", "manual-tecnico")
DIMENSION = 384


def embed(texts: list[str]) -> list[list[float]]:
    return [vector.tolist() for vector in DefaultEmbeddingFunction()(texts)]


def connect_index():
    client = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    name = os.getenv("INDEX_NAME", "curso-ai-engineering")
    if name not in client.list_indexes().names():
        client.create_index(name=name, dimension=DIMENSION, metric="cosine", spec=ServerlessSpec(cloud="aws", region="us-east-1"))
    deadline = time.monotonic() + 120
    while True:
        info = client.describe_index(name)
        if info.dimension != DIMENSION:
            raise ValueError(f"Se requieren {DIMENSION} dimensiones; el índice tiene {info.dimension}")
        if info.status["ready"]:
            return client.Index(host=info.host)
        if time.monotonic() > deadline:
            raise TimeoutError("El índice no quedó listo en 120 segundos")
        time.sleep(2)


def prepare_documents(data_dir=ROOT / "data") -> list[dict]:
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base", chunk_size=600, chunk_overlap=50
    )
    documents = []
    for file in sorted(Path(data_dir).glob("*.md")):
        for page, text in enumerate(splitter.split_text(file.read_text(encoding="utf-8")), 1):
            key = hashlib.sha256(f"{file.name}:{page}:{text}".encode()).hexdigest()
            documents.append({"id": key, "text": text, "source": file.name, "page": page, "category": "python"})
    if not documents:
        raise ValueError("No hay documentos Markdown en data")
    return documents


def ingest(index=None) -> int:
    docs = prepare_documents()
    index = index if index is not None else connect_index()
    for start in range(0, len(docs), 100):
        batch = docs[start:start + 100]
        embeddings = embed([doc["text"] for doc in batch])
        vectors = [{"id": doc["id"], "values": vector,
                    "metadata": {"text": doc["text"], "source": doc["source"], "page": doc["page"], "category": doc["category"]}}
                   for doc, vector in zip(batch, embeddings)]
        index.upsert(vectors=vectors, namespace=NAMESPACE)
    (ROOT / "corpus.json").write_text(json.dumps(docs, ensure_ascii=False, indent=2), encoding="utf-8")
    return len(docs)


if __name__ == "__main__":
    print(f"Documentos indexados: {ingest()}")
