#!/usr/bin/env python3
"""Build a COVID-19 question set for the MedGemma + MultiVerS evaluation.

Each question is written by MedGemma from one HealthVer test claim, so the
HealthVer corpus actually contains evidence that can answer it. The source
claim, its gold label and its gold evidence documents are kept with the
question; they are reference material for the human annotator and for
retrieval metrics, not labels for MedGemma's own claims.
"""
import argparse
import json
import random
import re
import sys
from pathlib import Path

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from medgemma_multivers_pipeline import generate


def read_jsonl(path):
    return [json.loads(x) for x in Path(path).open(encoding="utf-8") if x.strip()]


def gold_label(claim):
    labels = {e["label"] for v in claim["evidence"].values() for e in v}
    if not labels:
        return "NEI"
    return labels.pop() if len(labels) == 1 else "MIXED"


def question_prompt(claim):
    return f"""Rewrite the statement below as one neutral question that a member of the public
might ask a doctor about COVID-19. The question must be answerable with yes/no or a short
explanation, must not reveal whether the statement is true, and must keep the key medical
entities (drug, symptom, intervention, population). Return only the question.

Statement: {claim.strip()}"""


def clean_question(text):
    text = text.strip().strip('"').strip()
    text = re.sub(r"^\s*(question|q)\s*:\s*", "", text, flags=re.I)
    return text.splitlines()[0].strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--claims", default="data_train/target/healthver/claims_test.jsonl")
    ap.add_argument("--corpus", default="data_train/target/healthver/corpus.jsonl")
    ap.add_argument("--output", default="data/medgemma_eval/covid_questions.jsonl")
    ap.add_argument("--model_id", default="google/medgemma-4b-it")
    ap.add_argument("--per_label", default="SUPPORT=20,CONTRADICT=15,MIXED=10,NEI=15")
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    claims = read_jsonl(args.claims)
    quotas = {k: int(v) for k, v in (x.split("=") for x in args.per_label.split(","))}
    rng = random.Random(args.seed)
    selected = []
    for label, n in quotas.items():
        pool = [c for c in claims if gold_label(c) == label and c.get("cited_doc_ids")]
        selected += rng.sample(pool, min(n, len(pool)))
    selected.sort(key=lambda c: c["id"])

    processor = AutoProcessor.from_pretrained(args.model_id)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None).eval()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for i, c in enumerate(selected, 1):
            question = clean_question(generate(processor, model, question_prompt(c["claim"]), 64))
            row = {
                "id": i,
                "question": question,
                "source_healthver_id": c["id"],
                "source_claim": c["claim"].strip(),
                "gold_label": gold_label(c),
                "gold_evidence": c["evidence"],
                "gold_doc_ids": [int(d) for d in c["evidence"]] or c["cited_doc_ids"],
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"[{i}/{len(selected)}] {row['gold_label']:10s} {question}", flush=True)


if __name__ == "__main__":
    main()
