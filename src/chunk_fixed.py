from src.ingest import load_pdfs, get_chroma_client, get_embedding_fn

CHUNK_SIZE = 512
CHUNK_OVERLAP = 64


def chunk_text_fixed(text: str) -> list[str]:
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i + CHUNK_SIZE])
        if chunk.strip():
            chunks.append(chunk)
        i += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


def build_fixed_collection():
    client = get_chroma_client()
    ef = get_embedding_fn()

    try:
        client.delete_collection("fixed")
        print("  deleted existing fixed collection")
    except Exception:
        pass

    collection = client.create_collection("fixed", embedding_function=ef)
    docs = load_pdfs("corpus")

    for doc in docs:
        chunks = chunk_text_fixed(doc["text"])
        print(f"  {doc['name']}: {len(chunks)} chunks")
        collection.add(
            documents=chunks,
            ids=[f"{doc['name']}_fixed_{i}" for i in range(len(chunks))],
            metadatas=[{"source": doc["name"]} for _ in chunks]
        )

    print(f"\nFixed collection ready: {collection.count()} total chunks")
    return collection


if __name__ == "__main__":
    build_fixed_collection()