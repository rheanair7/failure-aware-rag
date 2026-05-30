import os
from pypdf import PdfReader
import chromadb
from chromadb.utils import embedding_functions


def load_pdfs(corpus_dir: str) -> list[dict]:
    docs = []
    for fname in sorted(os.listdir(corpus_dir)):
        if not fname.endswith(".pdf"):
            continue
        path = os.path.join(corpus_dir, fname)
        print(f"  reading {fname}...")
        try:
            reader = PdfReader(path)
            text = "\n".join(
                page.extract_text() or "" for page in reader.pages
            )
            if len(text.strip()) < 100:
                print(f"    warning: very little text extracted from {fname}")
            docs.append({
                "name": fname.replace(".pdf", ""),
                "text": text,
                "pages": len(reader.pages)
            })
            print(f"    {len(reader.pages)} pages, {len(text)} chars")
        except Exception as e:
            print(f"    failed to read {fname}: {e}")
    return docs


def get_chroma_client():
    return chromadb.PersistentClient(path=".chromadb")


def get_embedding_fn():
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )