#!/usr/bin/env python3
"""Create a human-label template for a MedGemma claim-verification run."""

import argparse
import json
from pathlib import Path


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).open(encoding="utf-8")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--claims", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    claims = read_jsonl(args.claims)
    with Path(args.output).open("w", encoding="utf-8") as f:
        for row in claims:
            f.write(json.dumps({
                "claim_id": int(row["id"]),
                "claim": row["claim"],
                "gold_label": "",  # SUPPORT, CONTRADICT, or NEI
                "gold_evidence": [],  # [{"doc_id": 123, "sentences": [0, 1]}]
            }) + "\n")
    print(f"Created {args.output}; fill in gold_label and gold_evidence.")


if __name__ == "__main__":
    main()
