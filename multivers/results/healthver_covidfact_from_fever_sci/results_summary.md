# Joint HealthVer + COVID-Fact fine-tuning (from fever_sci)

Checkpoint: `checkpoint/epoch=8-step=7011.ckpt`, selected on HealthVer validation
only (`--valid_datasets healthver`). 20 epochs, same hyperparameters as
`healthver_from_fever_sci`, default dataset reweighting.

| Metric (HealthVer test, 903 pairs) | HealthVer only (fever_sci) | HealthVer + COVID-Fact |
|---|---:|---:|
| Accuracy | **76.97** | 75.30 |
| Macro-F1 | **76.29** | 75.07 |
| Abstract F1 | **76.51** | 75.21 |
| Sentence selection F1 | **86.08** | 83.72 |
| Sentence label F1 | **74.51** | 72.77 |
| Best validation sentence F1 | **64.7** | 61.1 |

Conclusion: adding COVID-Fact did not help HealthVer; the HealthVer-only
fever_sci model remains the best.
