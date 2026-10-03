# HealthVer model comparison

All models were evaluated on the same 903-example test set.

| Model | Accuracy | Macro-F1 | Abstract F1 | Sentence selection F1 | Sentence label F1 |
|---|---:|---:|---:|---:|---:|
| Original MultiVerS | 74.86 | 74.37 | 74.90 | 84.58 | 73.52 |
| Fine-tuned MultiVerS | **75.30** | **74.89** | **75.43** | **85.41** | **74.82** |
| Label-aware MultiVerS | 74.75 | 74.37 | 74.23 | 82.91 | 73.04 |

The label-aware model used differentiable label probabilities as soft context
for the rationale selector. It did not improve over the fine-tuned baseline
on this test set, so the fine-tuned model remains the preferred model.
