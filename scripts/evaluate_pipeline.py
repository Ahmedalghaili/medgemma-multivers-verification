#!/usr/bin/env python3
"""Evaluate an annotated MedGemma -> MultiVerS pipeline run."""

import argparse
import json
from pathlib import Path


LABELS = ["SUPPORT", "CONTRADICT", "NEI"]
LABEL_INDEX = {"CONTRADICT": 0, "NEI": 1, "SUPPORT": 2}


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).open(encoding="utf-8")]


def prf(correct, retrieved, relevant):
    precision = correct / retrieved if retrieved else 0.0
    recall = correct / relevant if relevant else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def aggregate_document_predictions(predicted_docs):
    """Choose the strongest non-NEI evidence, using label probability * rationale confidence."""
    candidates = []
    for doc_id, item in predicted_docs.items():
        probs = item.get("label_probs", [])
        if len(probs) != 3:
            # Backward compatibility with old prediction files.
            if item.get("sentences") and item.get("label") in LABELS:
                candidates.append((1.0, item.get("label"), doc_id))
            continue
        sentence_probs = item.get("rationale_probs", [])
        rationale_conf = max((float(x) for x in sentence_probs), default=0.0)
        for label in ("SUPPORT", "CONTRADICT"):
            score = float(probs[LABEL_INDEX[label]]) * max(rationale_conf, 1e-6)
            candidates.append((score, label, doc_id))
    if not candidates:
        return "NEI", None
    score, label, doc_id = max(candidates, key=lambda x: x[0])
    # A very low evidence score is treated as NEI.
    if score < 0.20:
        return "NEI", None
    return label, doc_id


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", required=True)
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    gold = {int(x["claim_id"]): x for x in read_jsonl(args.gold)}
    pred = {int(x["id"]): x.get("evidence", {}) for x in read_jsonl(args.predictions)}
    confusion = {g: {p: 0 for p in LABELS} for g in LABELS}
    correct_labels = 0
    evidence_correct = evidence_retrieved = evidence_relevant = 0
    unsupported = 0

    for claim_id, row in gold.items():
        gold_label = row["gold_label"]
        if gold_label not in LABELS:
            raise ValueError(f"Fill gold_label for claim_id={claim_id}")
        gold_ev = {
            (int(item["doc_id"]), int(sentence))
            for item in row.get("gold_evidence", [])
            for sentence in item.get("sentences", [])
        }
        predicted_docs = pred.get(claim_id, {})
        predicted = {
            (int(doc_id), int(sentence))
            for doc_id, item in predicted_docs.items()
            for sentence in item.get("sentences", [])
        }
        pred_label, _ = aggregate_document_predictions(predicted_docs)
        if pred_label not in LABELS:
            raise ValueError(f"Unknown predicted label: {pred_label}")
        correct_labels += pred_label == gold_label
        confusion[gold_label][pred_label] += 1
        unsupported += int(gold_label == "NEI" and bool(predicted))
        evidence_relevant += len(gold_ev)
        evidence_retrieved += len(predicted)
        evidence_correct += len(gold_ev & predicted)

    n = len(gold)
    per_label = {}
    macro_f1 = 0.0
    for label in LABELS:
        tp = confusion[label][label]
        fp = sum(confusion[g][label] for g in LABELS if g != label)
        fn = sum(confusion[label][p] for p in LABELS if p != label)
        p, r, f1 = prf(tp, tp + fp, tp + fn)
        per_label[label] = {"precision": p, "recall": r, "f1": f1}
        macro_f1 += f1
    ev_p, ev_r, ev_f1 = prf(evidence_correct, evidence_retrieved, evidence_relevant)
    result = {
        "n_claims": n,
        "label_accuracy": correct_labels / n if n else 0.0,
        "label_macro_f1": macro_f1 / len(LABELS),
        "evidence_precision": ev_p,
        "evidence_recall": ev_r,
        "evidence_f1": ev_f1,
        "unsupported_claim_rate": unsupported / n if n else 0.0,
        "per_label": per_label,
        "confusion": confusion,
    }
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
