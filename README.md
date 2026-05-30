# failure-aware-rag

Most RAG tutorials measure whether retrieval "works."
This one measures where it fails — and why semantic chunking wins on grounding
but the margin depends entirely on whether your chunks preserve causal reasoning.

Built over 11 autonomous driving and lane detection research papers.
Evaluated with a custom LLM judge that scores grounding, hallucination, and
citation fidelity across 30 domain-expert questions.

---

## What this project does

Standard RAG pipelines retrieve documents and generate answers. This project
adds an evaluation layer that measures answer quality across three dimensions:

- **Grounding** — is every claim in the answer supported by the retrieved context?
- **Hallucination** — does the answer introduce facts not present in the context?
- **Citation fidelity** — are paper names, methods, and numbers accurate?

Two chunking strategies are compared head-to-head on the same corpus and the
same 30 questions. The results are not close.

---

## Results

| Metric | Fixed chunking | Semantic chunking |
|---|---|---|
| Grounding | 0.458 | 0.930 |
| Hallucination | 0.412 | 0.930 |
| Citation fidelity | 0.545 | 0.997 |

Semantic chunking outperforms fixed chunking by roughly 2x on every metric.

### Why the gap is this large

Fixed chunking splits documents every 512 words regardless of meaning.
A question asking for both a quantitative result and its explanation will
often retrieve the explanation chunk but miss the results table chunk —
they got separated at the 512-word boundary.

Semantic chunking splits on embedding cosine distance drops between sentences,
keeping topically coherent content together. When the retriever finds a chunk,
it contains everything needed to answer the question.

The most revealing failure: q01 asks for CLRNet's F1 score on the crossroad
scenario. Fixed chunking retrieved the causal explanation (row-anchor priors
incompatible with intersection geometry) but not the number (F1 = 0.000).
The judge correctly scored this as low grounding — the answer was partially
right but missing the quantitative claim.

---

## Corpus

11 research papers on lane detection and autonomous driving perception,
downloaded from arXiv:

- CLRNet (Zheng et al., CVPR 2022)
- LaneATT (Tabelini et al., CVPR 2021)
- UFLD (Pan et al., ECCV 2018)
- BEVFormer (Li et al., ECCV 2022)
- DETR (Carion et al., ECCV 2020)
- CondLaneNet (Liu et al., ICCV 2021)
- Anchor3DLane (Chen et al., CVPR 2023)
- CULane dataset paper (Pan et al., 2018)
- LaneGCN (Liang et al., ECCV 2020)
- VPGNet (Lee et al., ICCV 2017)
- Scenario-based failure analysis of CLRNet on CULane (local)

---

## Eval design

### The 30 questions

Questions are written by a domain expert across five categories:

- **failure_analysis** — where and why does CLRNet fail per scenario?
- **cross_paper_comparison** — how do LaneATT, CondLaneNet, CLRNet differ architecturally?
- **methodology** — Grad-CAM target functions, augmentation parameters, eval protocols
- **intervention_analysis** — TTA vs fine-tuning, which scenarios recover and why?
- **safety_implications** — what do these failures mean for real-world deployment?

Questions are deliberately hard. Cross-paper comparison and safety implication
questions require the retriever to surface the right chunk from the right paper.
Generic or hallucinated answers score near zero on citation fidelity.

### The custom judge

A Claude-based judge scores each answer on grounding, hallucination, and
citation fidelity using structured JSON output. The judge has access to the
retrieved context and the ground truth answer, and flags specific hallucinated
claims rather than returning a single scalar.

This catches failure modes that aggregate metrics miss — for example, a model
that correctly describes a method but attributes it to the wrong paper, or one
that uses the right terminology but invents a number.

---

## Quickstart

```bash
git clone https://github.com/rheanair7/failure-aware-rag
cd failure-aware-rag
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# download corpus
python download_corpus.py

# build vector indices (first run only)
python run_eval.py --build-index --strategy both

# run full eval
python run_eval.py --strategy both

# test with 3 questions first
python run_eval.py --strategy both --limit 3
```

Add your Anthropic API key to `.env`:
```
ANTHROPIC_API_KEY=sk-ant-...
```

---

## Project structure

```
failure-aware-rag/
├── src/
│   ├── ingest.py          # PDF loading and ChromaDB client
│   ├── chunk_fixed.py     # 512-token fixed chunking
│   ├── chunk_semantic.py  # embedding cosine-drop semantic chunking
│   ├── retrieve.py        # query -> top-k -> Claude answer
│   └── eval_judge.py      # custom LLM judge: grounding + hallucination
├── questions.json         # 30 domain-expert questions with ground truth
├── download_corpus.py     # reproducible corpus downloader
├── run_eval.py            # CLI entrypoint
└── results/               # eval CSVs (gitignored)
```

---

## Key findings

**1. Semantic chunking is not marginally better — it is categorically better.**
The 2x improvement across all three metrics suggests that chunk boundary
placement is the single highest-leverage decision in a RAG pipeline for
technical domains where quantitative results and their explanations are
co-located in text.

**2. Fixed chunking fails specifically on questions that require co-located evidence.**
Questions about specific F1 scores, augmentation parameters, and experimental
results score near zero under fixed chunking. The retriever finds related chunks
but not the chunk containing the specific claim being asked about.

**3. The custom judge catches hallucination that aggregate metrics miss.**
In several cases the answer was directionally correct but introduced a specific
term or number not present in the retrieved context. The custom judge flags
the specific claim rather than returning a passing aggregate score.

**4. Citation fidelity is the hardest metric to satisfy under fixed chunking.**
A model answering about lane detection methods will confidently attribute
techniques to papers not in the retrieved context. Semantic chunking, by keeping
paper-specific content together, reduces cross-paper contamination significantly.

---

## Stack

`chromadb` `sentence-transformers` `anthropic` `pypdf` `rich` `pydantic` `python-dotenv`