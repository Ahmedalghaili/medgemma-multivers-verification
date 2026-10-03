# Final MedGemma + MultiVerS evaluation

## Dataset statistics

- Questions: 60
- Generated answers: 60
- Extracted claims: 187
- LLM JSON claims: 15
- Conservative sentence-fallback claims: 172
- Semantic retrieval model: `pritamdeka/S-PubMedBert-MS-MARCO`
- Verifier checkpoint: `multivers/finetune_results/healthver_finetuned_resume/checkpoint/epoch=3-step=2648.ckpt`

## Automatic output distribution

- SUPPORT: 112
- CONTRADICT: 17
- NEI: 58

## Human-evaluated performance

Accuracy, Macro-F1, evidence precision, evidence recall, evidence F1, and unsupported claim rate are **pending manual annotation**. The annotation file is `manual_review.csv`; human labels must be filled before calculating these metrics.

| System | Accuracy | Macro-F1 |
|---|---:|---:|
| MedGemma only | Pending human review | Pending |
| MedGemma + original MultiVerS | Not run by design | Not run |
| MedGemma + fine-tuned MultiVerS | Pending human review | Pending |

## Error analysis form

For each incorrect claim, mark the primary error as: MedGemma hallucination, claim extraction error, retrieval error, or MultiVerS verification error.
