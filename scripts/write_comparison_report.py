#!/usr/bin/env python3
"""Write the reproducible experiment comparison tables."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def load(p): return json.loads((ROOT / p).read_text())

def pct(x): return "N/A" if x is None else f"{100*x:.2f}%"

def main():
    original = load("multivers/results/healthver_science_init/healthver_test_metrics.json")
    finetuned = load("multivers/finetune_results/healthver_finetuned_resume/healthver_test_metrics.json")
    orig_pipe = load("results/pipeline_eval_100/original_metrics.json")
    ft_pipe = load("results/pipeline_eval_100/metrics.json")
    sem_pipe = load("results/pipeline_eval_100_semantic/metrics.json")
    lines=["# Pipeline experiment comparison", "",
           "The standalone scores are the frozen HealthVer test results. The end-to-end", 
           "scores use the 100-claim HealthVer grounding benchmark; its answer field is", 
           "a claim proxy, not a newly generated MedGemma answer.", "",
           "## Table 1 — Standalone verifier", "",
           "| System | Accuracy | Macro-F1 | Abstract F1 | Sentence selection F1 | Sentence label F1 |",
           "|---|---:|---:|---:|---:|---:|",
           f"| Original MultiVerS | {pct(original['accuracy'])} | {pct(original['macro_f1'])} | {pct(original['non_nei_abstract_f1'])} | {pct(original['sentence_selection_f1'])} | {pct(original['sentence_label_f1'])} |",
           f"| Fine-tuned MultiVerS | {pct(finetuned['accuracy'])} | {pct(finetuned['macro_f1'])} | {pct(finetuned['non_nei_abstract_f1'])} | {pct(finetuned['sentence_selection_f1'])} | {pct(finetuned['sentence_label_f1'])} |",
           "", "## Table 2 — Grounding pipeline benchmark", "",
           "| System | Accuracy | Macro-F1 | Evidence precision | Evidence recall | Evidence F1 | Unsupported rate |",
           "|---|---:|---:|---:|---:|---:|---:|",
           "| MedGemma only | N/A | N/A | N/A | N/A | N/A | N/A |",
           f"| MedGemma + original MultiVerS + TF-IDF | {pct(orig_pipe['label_accuracy'])} | {pct(orig_pipe['label_macro_f1'])} | {pct(orig_pipe['evidence_precision'])} | {pct(orig_pipe['evidence_recall'])} | {pct(orig_pipe['evidence_f1'])} | {pct(orig_pipe['unsupported_claim_rate'])} |",
           f"| MedGemma + fine-tuned MultiVerS + TF-IDF | {pct(ft_pipe['label_accuracy'])} | {pct(ft_pipe['label_macro_f1'])} | {pct(ft_pipe['evidence_precision'])} | {pct(ft_pipe['evidence_recall'])} | {pct(ft_pipe['evidence_f1'])} | {pct(ft_pipe['unsupported_claim_rate'])} |",
           f"| MedGemma + fine-tuned MultiVerS + semantic retrieval | {pct(sem_pipe['label_accuracy'])} | {pct(sem_pipe['label_macro_f1'])} | {pct(sem_pipe['evidence_precision'])} | {pct(sem_pipe['evidence_recall'])} | {pct(sem_pipe['evidence_f1'])} | {pct(sem_pipe['unsupported_claim_rate'])} |",
           "", "## Interpretation", "",
           "Semantic retrieval improved claim accuracy from 62.00% to 69.00% and macro-F1",
           "from 54.25% to 59.57% compared with fine-tuned MultiVerS plus TF-IDF. Evidence",
           "recall also increased, while evidence precision decreased slightly. The next",
           "formal experiment should replace the proxy answers with manually checked",
           "MedGemma answers and claims.", ""]
    out=ROOT/"results/pipeline_comparison.md"; out.write_text("\n".join(lines), encoding="utf-8"); print(out)

if __name__ == "__main__": main()
