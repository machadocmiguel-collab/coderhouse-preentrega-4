import asyncio
import json
from ingest import ROOT
from retriever import RAGSystem


def metrics(retrieved: list[str], expected: list[str], k: int = 5) -> dict[str, float]:
    relevant = set(expected)
    hits = len(set(retrieved[:k]) & relevant)
    return {"precision@5": hits / k, "recall@5": hits / len(relevant) if relevant else 0.0}


async def evaluate():
    system = RAGSystem()
    benchmark = json.loads((ROOT / "golden_set.json").read_text(encoding="utf-8"))
    rows = []
    for case in benchmark:
        docs = await system.search(case["pregunta"])
        ids = [doc.metadata["source"] for doc in docs]
        rows.append({"pregunta": case["pregunta"], "recuperados": ids,
                     **metrics(ids, case["documentos_esperados"])})
    result = {"consultas": rows,
              "precision@5": sum(row["precision@5"] for row in rows) / len(rows),
              "recall@5": sum(row["recall@5"] for row in rows) / len(rows)}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    (ROOT / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    asyncio.run(evaluate())
