# Pipeline experiment comparison

The standalone scores are the frozen HealthVer test results. The end-to-end
scores use the 100-claim HealthVer grounding benchmark; its answer field is
a claim proxy, not a newly generated MedGemma answer.

## Table 1 — Standalone verifier

| System | Accuracy | Macro-F1 | Abstract F1 | Sentence selection F1 | Sentence label F1 |
|---|---:|---:|---:|---:|---:|
| Original MultiVerS | 74.86% | 74.37% | 74.90% | 84.58% | 73.52% |
| Fine-tuned MultiVerS | 75.30% | 74.89% | 75.43% | 85.41% | 74.82% |

## Table 2 — Grounding pipeline benchmark

| System | Accuracy | Macro-F1 | Evidence precision | Evidence recall | Evidence F1 | Unsupported rate |
|---|---:|---:|---:|---:|---:|---:|
| MedGemma only | N/A | N/A | N/A | N/A | N/A | N/A |
| MedGemma + original MultiVerS + TF-IDF | 64.00% | 57.03% | 21.02% | 42.57% | 28.15% | 17.00% |
| MedGemma + fine-tuned MultiVerS + TF-IDF | 62.00% | 54.25% | 19.72% | 43.69% | 27.17% | 18.00% |
| MedGemma + fine-tuned MultiVerS + semantic retrieval | 69.00% | 59.57% | 18.49% | 47.30% | 26.58% | 18.00% |

## Interpretation

Semantic retrieval improved claim accuracy from 62.00% to 69.00% and macro-F1
from 54.25% to 59.57% compared with fine-tuned MultiVerS plus TF-IDF. Evidence
recall also increased, while evidence precision decreased slightly. The next
formal experiment should replace the proxy answers with manually checked
MedGemma answers and claims.
