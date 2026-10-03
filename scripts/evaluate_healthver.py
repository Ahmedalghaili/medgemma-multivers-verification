#!/usr/bin/env python3
"""Evaluate MultiVerS predictions against the HealthVer entailment split."""

import argparse
import json
from collections import defaultdict
from pathlib import Path


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def f1(p, r):
    return 0.0 if p + r == 0 else 2 * p * r / (p + r)


def prf(correct, retrieved, relevant):
    p = correct / retrieved if retrieved else 0.0
    r = correct / relevant if relevant else 0.0
    return p, r, f1(p, r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--claims", required=True, help="HF-derived test claims JSONL")
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    gold_rows = read_jsonl(args.claims)
    predictions = {int(x["id"]): x.get("evidence", {}) for x in read_jsonl(args.predictions)}
    gold_pairs = {}
    for row in gold_rows:
        label = row["verdict"]
        gold_pairs[(int(row["claim_id"]), int(row["abstract_id"]))] = (
            label,
            set(row["evidence"]),
        )

    labels = ["SUPPORT", "CONTRADICT", "NEI"]
    correct = 0
    confusion = {g: {p: 0 for p in labels} for g in labels}
    predicted_non_nei = 0
    gold_non_nei = 0
    correct_non_nei = 0
    evidence_retrieved = evidence_correct = evidence_relevant = 0
    evidence_labeled_correct = 0

    for (claim_id, doc_id), (gold_label, gold_ev) in gold_pairs.items():
        pred_doc = predictions.get(claim_id, {}).get(str(doc_id), {})
        pred_label = pred_doc.get("label", "NEI")
        pred_ev = set(int(i) for i in pred_doc.get("sentences", []))
        if pred_label not in labels:
            raise ValueError(f"Unknown prediction label: {pred_label}")
        correct += pred_label == gold_label
        confusion[gold_label][pred_label] += 1
        predicted_non_nei += pred_label != "NEI"
        gold_non_nei += gold_label != "NEI"
        correct_non_nei += (gold_label != "NEI") and (pred_label == gold_label)
        if gold_label != "NEI":
            evidence_relevant += len(gold_ev)
            evidence_retrieved += len(pred_ev)
            evidence_correct += len(pred_ev & gold_ev)
            evidence_labeled_correct += len(pred_ev & gold_ev) if pred_label == gold_label else 0

    total = len(gold_pairs)
    accuracy = correct / total
    macro_f1 = 0.0
    per_label = {}
    for label in labels:
        tp = confusion[label][label]
        fp = sum(confusion[g][label] for g in labels if g != label)
        fn = sum(confusion[label][p] for p in labels if p != label)
        p, r, score = prf(tp, tp + fp, tp + fn)
        per_label[label] = {"precision": p, "recall": r, "f1": score}
        macro_f1 += score
    macro_f1 /= len(labels)
    non_nei_p, non_nei_r, non_nei_f1 = prf(
        correct_non_nei, predicted_non_nei, gold_non_nei
    )
    ev_p, ev_r, ev_f1 = prf(evidence_correct, evidence_retrieved, evidence_relevant)
    lev_p, lev_r, lev_f1 = prf(
        evidence_labeled_correct, evidence_retrieved, evidence_relevant
    )
    result = {
        "n_pairs": total,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "non_nei_abstract_precision": non_nei_p,
        "non_nei_abstract_recall": non_nei_r,
        "non_nei_abstract_f1": non_nei_f1,
        "sentence_selection_precision": ev_p,
        "sentence_selection_recall": ev_r,
        "sentence_selection_f1": ev_f1,
        "sentence_label_precision": lev_p,
        "sentence_label_recall": lev_r,
        "sentence_label_f1": lev_f1,
        "per_label": per_label,
        "confusion": confusion,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
