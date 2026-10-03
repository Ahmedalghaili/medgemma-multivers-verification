# HealthVer MultiVerS from fever_sci pretraining

Checkpoint: `checkpoint/epoch=15-step=10592.ckpt` (selected on validation
`valid_sentence_label_f1`). Trained 20 epochs on HealthVer starting from the
released `fever_sci` model (FEVER + PubMedQA + EvidenceInference), as in the
original MultiVerS recipe. Same hyperparameters as `healthver_science_init`.

## HealthVer test (903 pairs, local evaluator)

| Metric | science_init (old) | fever_sci (this run) | Paper |
|---|---:|---:|---:|
| Pair accuracy | 74.86 | **76.97** | – |
| Pair macro-F1 | 74.37 | **76.29** | – |
| Abstract F1 | 74.90 | **76.51** | 77.60 |
| Sentence selection F1 | 84.58 | **86.08** | – |
| Sentence label F1 | 73.52 | **74.51** | 69.10 |

## MedGemma + MultiVerS pipeline (60 COVID questions, 177 MedGemma claims)

Scored against LLM-annotator labels, not human labels.

| Verifier | Accuracy | Macro-F1 | CONTRADICT F1 | NEI F1 |
|---|---:|---:|---:|---:|
| Fine-tuned science_init | 65.5 | 44.1 | 24.2 | 27.3 |
| fever_sci (this run) | **68.9** | **53.5** | **51.9** | 27.3 |
