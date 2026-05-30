from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from src.ingest import load_pdfs, get_chroma_client, get_embedding_fn

BREAKPOINT_THRESHOLD = 0.35
MIN_CHUNK_WORDS = 30

model = SentenceTransformer("all-MiniLM-L6-v2")


def chunk_text_semantic(text: str) -> list[str]:
    sentences = [s.strip() for s in text.split(".") if len(s.strip().split()) > 5]
    if len(sentences) < 2:
        return [text]

    embeddings = model.encode(sentences, show_progress_bar=False)
    chunks, current = [], [sentences[0]]

    for i in range(1, len(sentences)):
        sim = cosine_similarity([embeddings[i - 1]], [embeddings[i]])[0][0]
        if sim < (1 - BREAKPOINT_THRESHOLD):
            joined = ". ".join(current)
            if len(joined.split()) >= MIN_CHUNK_WORDS:
                chunks.append(joined)
            current = [sentences[i]]
        else:
            current.append(sentences[i])

    if current:
        chunks.append(". ".join(current))

    return chunks


def build_semantic_collection():
    client = get_chroma_client()
    ef = get_embedding_fn()

    try:
        client.delete_collection("semantic")
        print("  deleted existing semantic collection")
    except Exception:
        pass

    collection = client.create_collection("semantic", embedding_function=ef)
    docs = load_pdfs("corpus")

    for doc in docs:
        chunks = chunk_text_semantic(doc["text"])
        print(f"  {doc['name']}: {len(chunks)} chunks")
        collection.add(
            documents=chunks,
            ids=[f"{doc['name']}_sem_{i}" for i in range(len(chunks))],
            metadatas=[{"source": doc["name"]} for _ in chunks]
        )

    print(f"\nSemantic collection ready: {collection.count()} total chunks")
    return collection


if __name__ == "__main__":
    build_semantic_collection()