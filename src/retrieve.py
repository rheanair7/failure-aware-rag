import os
from dotenv import load_dotenv
load_dotenv()

from anthropic import Anthropic
from src.ingest import get_chroma_client, get_embedding_fn

client = Anthropic()

SYSTEM_PROMPT = """You are a research assistant with expertise in autonomous driving 
and lane detection. Answer questions using ONLY the provided context. If the context 
does not contain enough information to answer confidently, say so explicitly. 
Do not hallucinate paper names, results, methods, or numbers not present in the context."""


def retrieve_and_answer(question: str, strategy: str, top_k: int = 5) -> dict:
    chroma = get_chroma_client()
    ef = get_embedding_fn()
    collection = chroma.get_collection(strategy, embedding_function=ef)

    results = collection.query(query_texts=[question], n_results=top_k)
    context_chunks = results["documents"][0]
    sources = [m["source"] for m in results["metadatas"][0]]
    context = "\n\n---\n\n".join(context_chunks)

    response = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {question}"
        }]
    )

    return {
        "question": question,
        "strategy": strategy,
        "context": context,
        "answer": response.content[0].text,
        "context_chunks": context_chunks,
        "sources": sources,
    }


if __name__ == "__main__":
    from rich.console import Console
    console = Console()

    test_q = "What is CLRNet's F1 score on the crossroad scenario and why does it fail?"

    for strategy in ["fixed", "semantic"]:
        console.print(f"\n[bold cyan]Strategy: {strategy}[/bold cyan]")
        result = retrieve_and_answer(test_q, strategy)
        console.print(f"[bold]Sources:[/bold] {result['sources']}")
        console.print(f"[bold]Answer:[/bold] {result['answer']}")