# Final MedGemma + MultiVerS evaluation

## Dataset statistics

- Questions: 60
- Generated answers: 60
- Extracted claims: 177
- Claims extracted by MedGemma JSON: 177
- Claims from sentence fallback: 0
- Claims whose top-10 retrieved documents include a gold document of the source HealthVer claim: 104/177
- Semantic retrieval model: `pritamdeka/S-PubMedBert-MS-MARCO`
- Verifier checkpoint: `multivers/results/healthver_from_fever_sci/checkpoint/epoch=15-step=10592.ckpt`

## Automatic output distribution

- SUPPORT: 148
- CONTRADICT: 20
- NEI: 9

## Human-evaluated performance

Accuracy, Macro-F1, evidence precision, evidence recall, evidence F1, and unsupported claim rate are **pending manual annotation**. The annotation file is `manual_review.csv`; human labels must be filled before calculating these metrics.

| System | Accuracy | Macro-F1 |
|---|---:|---:|
| MedGemma only | Pending human review | Pending |
| MedGemma + original MultiVerS | Not run by design | Not run |
| MedGemma + fine-tuned MultiVerS | Pending human review | Pending |

## Error analysis form

For each incorrect claim, mark the primary error as: MedGemma hallucination, claim extraction error, retrieval error, or MultiVerS verification error.
