"""
ROUGE Evaluation Script for FAQ Agent
Compares model answers with reference answers from the FAQ document.
Requires: pip install rouge-score
Usage: python scripts/evaluate_faq.py
"""

import json
import os
import re
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv(override=True)


def extract_qa_from_pdf(pdf_path: str) -> list[dict]:
    """
    Extract Q&A pairs from FAQ PDF using heuristics.
    Assumes format like "Pertanyaan:" or "Q:" followed by "Jawaban:" or "A:".
    """
    from langchain_community.document_loaders import PyPDFLoader

    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    full_text = "\n\n".join(d.page_content for d in docs)

    qa_pairs = []
    # Try "Pertanyaan:" / "Jawaban:" (Indonesian)
    pattern = re.compile(
        r"Pertanyaan\s*[:\.]\s*(.+?)(?=Jawaban\s*[:\.]|$)",
        re.DOTALL | re.IGNORECASE
    )
    answers_pattern = re.compile(
        r"Jawaban\s*[:\.]\s*(.+?)(?=Pertanyaan\s*[:\.]|$)",
        re.DOTALL | re.IGNORECASE
    )

    questions = pattern.findall(full_text)
    answers = answers_pattern.findall(full_text)

    if questions and answers:
        for q, a in zip(questions, answers):
            q = q.strip()
            a = a.strip()
            if len(q) > 5 and len(a) > 5:
                qa_pairs.append({"question": q, "reference": a})
    else:
        # Fallback: try "Q:" / "A:"
        q_pat = re.compile(r"Q\s*[:\.]\s*(.+?)(?=A\s*[:\.]|$)", re.DOTALL | re.IGNORECASE)
        a_pat = re.compile(r"A\s*[:\.]\s*(.+?)(?=Q\s*[:\.]|$)", re.DOTALL | re.IGNORECASE)
        questions = q_pat.findall(full_text)
        answers = a_pat.findall(full_text)
        if questions and answers:
            for q, a in zip(questions, answers):
                q, a = q.strip(), a.strip()
                if len(q) > 5 and len(a) > 5:
                    qa_pairs.append({"question": q, "reference": a})

    return qa_pairs


def load_ground_truth(gt_path: str) -> list[dict]:
    """Load ground truth from JSON file. Format: [{"question": "...", "reference": "..."}, ...]"""
    with open(gt_path, encoding="utf-8") as f:
        return json.load(f)


def run_faq_agent(question: str, thread_id: str = "eval") -> str:
    """Invoke FAQ agent and return the generated answer."""
    from langchain_core.messages import HumanMessage
    import agents.FAQ as FAQ

    config = {"configurable": {"thread_id": thread_id}}
    result = FAQ.graph.invoke(
        {"messages": [HumanMessage(content=question)]},
        config=config,
    )
    return result["messages"][-1].content if result.get("messages") else ""


def compute_rouge(predictions: list[str], references: list[str]) -> dict:
    """Compute ROUGE-1, ROUGE-2, ROUGE-L scores."""
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    r1_p, r1_r, r1_f = [], [], []
    r2_p, r2_r, r2_f = [], [], []
    rl_p, rl_r, rl_f = [], [], []

    for pred, ref in zip(predictions, references):
        if not pred:
            pred = " "
        if not ref:
            ref = " "
        scores = scorer.score(ref, pred)
        r1_p.append(scores["rouge1"].precision)
        r1_r.append(scores["rouge1"].recall)
        r1_f.append(scores["rouge1"].fmeasure)
        r2_p.append(scores["rouge2"].precision)
        r2_r.append(scores["rouge2"].recall)
        r2_f.append(scores["rouge2"].fmeasure)
        rl_p.append(scores["rougeL"].precision)
        rl_r.append(scores["rougeL"].recall)
        rl_f.append(scores["rougeL"].fmeasure)

    n = len(predictions)
    return {
        "rouge1": {"precision": sum(r1_p) / n, "recall": sum(r1_r) / n, "f1": sum(r1_f) / n},
        "rouge2": {"precision": sum(r2_p) / n, "recall": sum(r2_r) / n, "f1": sum(r2_f) / n},
        "rougeL": {"precision": sum(rl_p) / n, "recall": sum(rl_r) / n, "f1": sum(rl_f) / n},
    }


def main():
    pdf_path = os.path.join(os.path.dirname(__file__), "..", "docs", "FAQ Dexa Medica.pdf")
    gt_path = os.path.join(os.path.dirname(__file__), "..", "scripts", "faq_ground_truth.json")

    # Load ground truth: prefer JSON file, else try extracting from PDF
    if os.path.exists(gt_path):
        print(f"Loading ground truth from {gt_path}")
        qa_pairs = load_ground_truth(gt_path)
    elif os.path.exists(pdf_path):
        print(f"Extracting Q&A from PDF: {pdf_path}")
        qa_pairs = extract_qa_from_pdf(pdf_path)
        if not qa_pairs:
            print("Could not extract Q&A from PDF. Create scripts/faq_ground_truth.json with format:")
            print('[{"question": "...", "reference": "..."}, ...]')
            return 1
    else:
        print("No ground truth found. Create scripts/faq_ground_truth.json or ensure PDF exists.")
        return 1

    print(f"Evaluating on {len(qa_pairs)} Q&A pairs...")

    predictions = []
    references = []
    for i, pair in enumerate(qa_pairs):
        q = pair["question"]
        ref = pair["reference"]
        print(f"  [{i+1}/{len(qa_pairs)}] Running: {q[:50]}...")
        pred = run_faq_agent(q, thread_id=f"eval_{i}")
        predictions.append(pred)
        references.append(ref)

    # Save results for analysis
    results_path = os.path.join(os.path.dirname(__file__), "..", "scripts", "evaluation_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(
            [
                {"question": qa["question"], "reference": r, "prediction": p}
                for qa, r, p in zip(qa_pairs, references, predictions)
            ],
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"\nResults saved to {results_path}")

    # Compute ROUGE
    metrics = compute_rouge(predictions, references)
    print("\n--- ROUGE Scores ---")
    for name, vals in metrics.items():
        print(f"  {name}: F1={vals['f1']:.4f} P={vals['precision']:.4f} R={vals['recall']:.4f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
