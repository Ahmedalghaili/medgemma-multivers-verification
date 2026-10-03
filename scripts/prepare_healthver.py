#!/usr/bin/env python3
"""Convert dwadden/healthver_entailment to MultiVerS JSONL formats."""

import argparse
import json
from collections import OrderedDict
from pathlib import Path

from datasets import load_dataset


LABELS = {"SUPPORT": "SUPPORT", "CONTRADICT": "CONTRADICT", "NEI": "NEI"}


def dump_jsonl(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_rows(dataset):
    """Return rows with plain Python values and validate evidence indices."""
    result = []
    for row in dataset:
        abstract = list(row["abstract"])
        evidence = sorted(set(int(i) for i in row["evidence"]))
        if any(i < 0 or i >= len(abstract) for i in evidence):
            raise ValueError(
                f"Invalid evidence index for claim={row['claim_id']} "
                f"document={row['abstract_id']}: {evidence}"
            )
        verdict = LABELS[row["verdict"]]
        if verdict == "NEI" and evidence:
            raise ValueError("NEI example contains evidence: " + repr(row))
        result.append(
            {
                "claim_id": int(row["claim_id"]),
                "claim": str(row["claim"]),
                "abstract_id": int(row["abstract_id"]),
                "title": str(row["title"]),
                "abstract": abstract,
                "verdict": verdict,
                "evidence": evidence,
            }
        )
    return result


def make_corpus(all_rows):
    corpus = OrderedDict()
    for row in all_rows:
        doc_id = row["abstract_id"]
        document = {
            "doc_id": doc_id,
            "title": row["title"],
            "abstract": row["abstract"],
        }
        if doc_id in corpus and corpus[doc_id] != document:
            raise ValueError(f"Conflicting versions of abstract_id={doc_id}")
        corpus[doc_id] = document
    return list(corpus.values())


def make_claims(rows, training):
    grouped = OrderedDict()
    for row in rows:
        claim_id = row["claim_id"]
        claim = grouped.setdefault(
            claim_id,
            {"id": claim_id, "claim": row["claim"], "doc_ids": [], "evidence": {}},
        )
        if claim["claim"] != row["claim"]:
            raise ValueError(f"Conflicting text for claim_id={claim_id}")
        doc_id = row["abstract_id"]
        if doc_id not in claim["doc_ids"]:
            claim["doc_ids"].append(doc_id)
        if row["verdict"] != "NEI":
            claim["evidence"][str(doc_id)] = [
                {"label": row["verdict"], "sentences": row["evidence"]}
            ]
    output = []
    for claim in grouped.values():
        if training:
            claim["cited_doc_ids"] = claim.pop("doc_ids")
        else:
            claim.pop("evidence", None)
        output.append(claim)
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=".", help="Project root for data_train/ and data/")
    parser.add_argument("--dataset", default="dwadden/healthver_entailment")
    args = parser.parse_args()
    output = Path(args.output)

    raw = load_dataset(args.dataset, trust_remote_code=True)
    splits = {name: normalize_rows(raw[name]) for name in ("train", "validation", "test")}
    all_rows = [row for rows in splits.values() for row in rows]
    corpus = make_corpus(all_rows)

    train_root = output / "data_train" / "target" / "healthver"
    prediction_root = output / "data" / "healthver"
    dump_jsonl(corpus, train_root / "corpus.jsonl")
    dump_jsonl(corpus, prediction_root / "corpus.jsonl")

    for split, rows in splits.items():
        fold = "dev" if split == "validation" else split
        dump_jsonl(make_claims(rows, training=True), train_root / f"claims_{fold}.jsonl")
        dump_jsonl(make_claims(rows, training=False), prediction_root / f"claims_{split}.jsonl")
        dump_jsonl(rows, prediction_root / f"gold_{split}.jsonl")

    metadata = {
        "dataset": args.dataset,
        "splits": {name: len(rows) for name, rows in splits.items()},
        "unique_documents": len(corpus),
        "training_root": str(train_root),
        "prediction_root": str(prediction_root),
    }
    (output / "healthver_conversion.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
