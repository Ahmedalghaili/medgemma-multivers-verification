# HealthVer fine-tuning results

Best checkpoint: `checkpoint/epoch=3-step=2648.ckpt`.

| Metric | Original MultiVerS | Fine-tuned | Change |
|---|---:|---:|---:|
| Accuracy | 74.86 | 75.30 | +0.44 |
| Macro-F1 | 74.37 | 74.89 | +0.52 |
| Abstract F1 | 74.90 | 75.43 | +0.53 |
| Sentence selection F1 | 84.58 | 85.41 | +0.83 |
| Sentence label F1 | 73.52 | 74.82 | +1.30 |

Evaluation uses the same 903-example HealthVer test set and the same local
evaluator as the original experiment.
