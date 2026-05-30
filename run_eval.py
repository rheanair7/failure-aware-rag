import json
import csv
import argparse
import os
from dotenv import load_dotenv
load_dotenv()

from rich.console import Console
from rich.table import Table
from src.retrieve import retrieve_and_answer
from src.eval_judge import judge_answer

console = Console()


def run(strategy: str, questions: list[dict]) -> list[dict]:
    console.print(f"\n[bold]Running eval: {strategy} chunking ({len(questions)} questions)[/bold]")
    judge_scores = []

    for q in questions:
        console.print(f"  [{q['id']}] {q['question'][:70]}...")
        try:
            result = retrieve_and_answer(q["question"], strategy)
            score = judge_answer(result, q["ground_truth"])
            score["id"] = q["id"]
            score["category"] = q.get("category", "unknown")
            score["difficulty"] = q.get("difficulty", "unknown")
            judge_scores.append(score)
        except Exception as e:
            console.print(f"    [red]error: {e}[/red]")
            judge_scores.append({
                "id": q["id"],
                "question": q["question"],
                "strategy": strategy,
                "grounding": 0.0,
                "hallucination": 0.0,
                "citation_fidelity": 0.0,
                "flags": [str(e)],
                "category": q.get("category", "unknown"),
                "difficulty": q.get("difficulty", "unknown"),
            })

    os.makedirs("results", exist_ok=True)
    out_path = f"results/{strategy}_results.csv"
    fieldnames = [
        "id", "question", "strategy", "category", "difficulty",
        "grounding", "hallucination", "citation_fidelity", "flags", "sources", "answer"
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(judge_scores)

    console.print(f"\n  saved to {out_path}")

    table = Table(title=f"{strategy} chunking -- mean judge scores")
    table.add_column("Metric", style="cyan")
    table.add_column("Mean", style="green")
    table.add_column("Min", style="yellow")
    table.add_column("Max", style="blue")

    for metric in ["grounding", "hallucination", "citation_fidelity"]:
        vals = [s[metric] for s in judge_scores if isinstance(s[metric], float)]
        if vals:
            table.add_row(
                metric,
                f"{sum(vals)/len(vals):.3f}",
                f"{min(vals):.3f}",
                f"{max(vals):.3f}",
            )
    console.print(table)
    return judge_scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RAG evals over lane detection corpus")
    parser.add_argument("--strategy", choices=["fixed", "semantic", "both"], default="both")
    parser.add_argument("--build-index", action="store_true", help="rebuild ChromaDB from PDFs")
    parser.add_argument("--limit", type=int, default=None, help="limit to first N questions (for testing)")
    args = parser.parse_args()

    if args.build_index:
        from src.chunk_fixed import build_fixed_collection
        from src.chunk_semantic import build_semantic_collection
        console.print("[bold]Building fixed index...[/bold]")
        build_fixed_collection()
        console.print("\n[bold]Building semantic index...[/bold]")
        build_semantic_collection()

    with open("questions.json", encoding="utf-8") as f:
        questions = json.load(f)

    if args.limit:
        questions = questions[:args.limit]
        console.print(f"[yellow]Limited to {args.limit} questions[/yellow]")

    strategies = ["fixed", "semantic"] if args.strategy == "both" else [args.strategy]

    all_scores = {}
    for strat in strategies:
        all_scores[strat] = run(strat, questions)

    if len(strategies) == 2:
        console.print("\n[bold]Strategy comparison[/bold]")
        comp = Table()
        comp.add_column("Metric", style="cyan")
        comp.add_column("Fixed", style="blue")
        comp.add_column("Semantic", style="green")
        comp.add_column("Winner", style="yellow")

        for metric in ["grounding", "hallucination", "citation_fidelity"]:
            fixed_mean = sum(s[metric] for s in all_scores["fixed"] if isinstance(s[metric], float)) / len(all_scores["fixed"])
            sem_mean = sum(s[metric] for s in all_scores["semantic"] if isinstance(s[metric], float)) / len(all_scores["semantic"])
            winner = "semantic" if sem_mean > fixed_mean else "fixed"
            comp.add_row(metric, f"{fixed_mean:.3f}", f"{sem_mean:.3f}", winner)

        console.print(comp)
        console.print("\n[green]Done. Results saved to results/[/green]")