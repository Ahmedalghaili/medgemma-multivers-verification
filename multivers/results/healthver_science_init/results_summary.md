# HealthVer MultiVerS results

Checkpoint: `checkpoint/epoch=17-step=11916.ckpt` (selected on validation
`valid_sentence_label_f1`). The model was trained for 20 epochs from the
scientific Longformer initialization, without the released HealthVer
checkpoint.

## Test results

| Metric | Result |
|---|---:|
| Pair accuracy | 74.86 |
| Pair macro-F1 | 74.37 |
| Non-NEI abstract F1 | 74.90 |
| Sentence selection precision | 91.82 |
| Sentence selection recall | 78.40 |
| Sentence selection F1 | 84.58 |
| Sentence label precision | 79.81 |
| Sentence label recall | 68.15 |
| Sentence label F1 | 73.52 |

The complete metrics, per-label scores, and confusion matrix are in
`healthver_test_metrics.json`.

## Comparison with the original MultiVerS paper

| System | Abstract P | Abstract R | Abstract F1 | Sentence P | Sentence R | Sentence F1 |
|---|---:|---:|---:|---:|---:|---:|
| This experiment | 74.34 | 75.46 | 74.90 | 79.81 | 68.15 | 73.52 |
| Original MultiVerS | 78.90 | 76.30 | 77.60 | 71.40 | 67.00 | 69.10 |

The paper's abstract F1 and sentence F1 use MultiVerS's original evaluator.
The local evaluator additionally reports pair accuracy, macro-F1, and
sentence-selection F1.
