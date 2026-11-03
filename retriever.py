import asyncio
import json
import re
from typing import Any
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from ingest import ROOT, NAMESPACE, connect_index, embed


def tokens(text: str) -> list[str]:
    return re.findall(r"[\w]+", text.casefold())


class PineconeRetriever(BaseRetriever):
    index: Any
    k: int = 5

    def _get_relevant_documents(self, query, *, run_manager):
        result = self.index.query(vector=embed([query])[0], top_k=self.k,
                                  namespace=NAMESPACE, include_metadata=True,
                                  filter={"category": {"$eq": "python"}})
        return [Document(page_content=match.metadata["text"],
                         metadata={**match.metadata, "id": match.id}) for match in result.matches]

    async def _aget_relevant_documents(self, query, *, run_manager):
        return await asyncio.to_thread(self._get_relevant_documents, query, run_manager=run_manager)


class RAGSystem:
    def __init__(self, index=None):
        corpus = json.loads((ROOT / "corpus.json").read_text(encoding="utf-8"))
        docs = [Document(page_content=doc["text"], metadata={key: value for key, value in doc.items() if key != "text"}) for doc in corpus]
        lexical = BM25Retriever.from_documents(docs, preprocess_func=tokens, k=5)
        dense = PineconeRetriever(index=index if index is not None else connect_index())
        self.retriever = EnsembleRetriever(retrievers=[lexical, dense], weights=[0.5, 0.5], id_key="id")

    async def search(self, query: str) -> list[Document]:
        return (await self.retriever.ainvoke(query))[:5]
