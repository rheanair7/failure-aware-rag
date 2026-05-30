import json
from dotenv import load_dotenv
load_dotenv()

from anthropic import Anthropic

client = Anthropic()

JUDGE_PROMPT = """You are evaluating a RAG system's answer on autonomous driving research.

Question: {question}
Retrieved context: {context}
System answer: {answer}
Ground truth: {ground_truth}

Score on three dimensions (0.0 to 1.0):
1. grounding: every claim in the answer is supported by the retrieved context (1.0 = fully grounded)
2. hallucination: answer introduces NO facts absent from context (1.0 = no hallucination at all)
3. citation_fidelity: paper names, methods, and numbers mentioned exist in the context (1.0 = perfect)

Also list any specific hallucinated claims if present.

Respond in JSON only, no other text, no markdown:
{{"grounding": float, "hallucination": float, "citation_fidelity": float, "flags": [list of hallucinated claims, empty list if none]}}"""


def judge_answer(result: dict, ground_truth: str) -> dict:
    prompt = JUDGE_PROMPT.format(
        question=result["question"],
        context=result["context"][:3000],
        answer=result["answer"],
        ground_truth=ground_truth,
    )

    response = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}]
    )

    raw = response.content[0].text.strip()
    # strip markdown code fences if model ignores instructions
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    try:
        scores = json.loads(raw)
    except json.JSONDecodeError:
        scores = {
            "grounding": 0.0,
            "hallucination": 0.0,
            "citation_fidelity": 0.0,
            "flags": ["json parse error -- raw response: " + raw[:200]]
        }

    scores["question"] = result["question"]
    scores["strategy"] = result["strategy"]
    scores["answer"] = result["answer"]
    scores["sources"] = result.get("sources", [])
    return scores


if __name__ == "__main__":
    from src.retrieve import retrieve_and_answer
    from rich.console import Console
    console = Console()

    test_q = "What is CLRNet's F1 score on the crossroad scenario and why does it fail?"
    ground_truth = "CLRNet achieves F1 = 0.000 on the crossroad scenario. The failure is structural: CLRNet's row-anchor priors assume lanes cross each horizontal row at predictable positions, which is fundamentally incompatible with intersection geometry."

    for strategy in ["fixed", "semantic"]:
        console.print(f"\n[bold cyan]Judging: {strategy}[/bold cyan]")
        result = retrieve_and_answer(test_q, strategy)
        scores = judge_answer(result, ground_truth)
        console.print(f"grounding:         {scores['grounding']}")
        console.print(f"hallucination:     {scores['hallucination']}")
        console.print(f"citation_fidelity: {scores['citation_fidelity']}")
        console.print(f"flags:             {scores['flags']}")