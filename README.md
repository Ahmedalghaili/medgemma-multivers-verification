# Verifying MedGemma Answers with Scientific Evidence

A pipeline that checks the answers of a medical language model, **MedGemma**, claim by
claim against research abstracts, using a fine-tuned **MultiVerS** fact-checking model.
For every claim it returns a verdict (**SUPPORT**, **CONTRADICT** or **NEI** — not
enough information) and the exact evidence sentences.

```
Question ─► MedGemma answer ─► MedGemma splits it into claims
                                         │
            HealthVer abstracts ◄── retrieval (S-PubMedBert, top-10) ◄┘
                    │
                    ▼
        MultiVerS: verdict + evidence sentences for each claim
```

Only MultiVerS is fine-tuned. MedGemma (`google/medgemma-4b-it`) and the retrieval model
(`pritamdeka/S-PubMedBert-MS-MARCO`) are used as released.

## Results

### MultiVerS on the HealthVer test set (903 claim–abstract pairs)

Two fine-tuning runs with identical settings; only the starting weights differ.

| Model | Starts from | Accuracy | Macro-F1 | Abstract F1 | Sentence selection F1 | Sentence label F1 |
|---|---|---:|---:|---:|---:|---:|
| Baseline | Longformer-large-science | 74.86 | 74.37 | 74.90 | 84.58 | 73.52 |
| **Improved** | **fever_sci** (MultiVerS pre-trained on FEVER, PubMedQA, EvidenceInference) | **76.97** | **76.29** | **76.51** | **86.08** | **74.51** |

Scored with `scripts/evaluate_healthver.py`. With the official
[scifact-evaluator](https://github.com/allenai/scifact-evaluator) the improved model gets
76.51 abstract F1 and 69.11 sentence-label F1 (baseline: 74.90 / 68.12); see
`official_evaluator_metrics.json` in each run folder.

Other fine-tuning runs (abstract F1): +5 more epochs on the baseline 75.43; label-aware
evidence head 74.23; HealthVer + COVID-Fact from fever_sci 75.21.

### Full system: MedGemma + MultiVerS

60 COVID-19 questions written from HealthVer test claims; MedGemma extracted 177 claims.

| Verifier inside the pipeline | Accuracy | Macro-F1 | CONTRADICT F1 | NEI F1 |
|---|---:|---:|---:|---:|
| Earlier fine-tuned model (+5 epochs) | 65.5 | 44.1 | 24.2 | 27.3 |
| **Improved model** | **68.9** | **53.5** | **51.9** | 27.3 |

> **Note:** the reference labels for these 177 claims were produced with an LLM annotator
> and have **not yet been reviewed by humans**. Known weakness: claims without evidence are
> often labelled SUPPORT (only 9 of 57 NEI claims are caught).

## Repository layout

```
scripts/                     Pipeline, data preparation, training and evaluation scripts
  prepare_healthver.py         HealthVer (Hugging Face) -> MultiVerS format
  train_healthver.sh           MultiVerS fine-tuning (baseline / fever_sci / multi-dataset)
  evaluate_healthver.py        Test-set metrics
  prepare_covid_questions.py   Builds the 60 COVID questions with MedGemma
  run_final_medgemma_eval.py   Full pipeline: MedGemma -> claims -> retrieval -> MultiVerS
  evaluate_manual_review.py    End-to-end metrics from an annotated review CSV
multivers/                   MultiVerS code (vendored, with small changes — see below)
  results/                     healthver_science_init (baseline), healthver_from_fever_sci (improved),
                               healthver_covidfact_from_fever_sci
  finetune_results/            +5 epochs run
  modified_results/            label-aware run
data/medgemma_eval/          The 60 COVID questions (covid_questions.jsonl)
results/                     Pipeline outputs: MedGemma answers, claims, retrieval, verdicts
logs/                        Training and pipeline logs
```

Each run folder contains `hparams.yaml`, per-epoch `metrics.csv`, test predictions,
`healthver_test_metrics.json` and `results_summary.md`. Model checkpoints are **not**
included (about 5 GB each).

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

MedGemma is gated on Hugging Face: accept its terms, then run `huggingface-cli login`.

### Data

```bash
python scripts/prepare_healthver.py --output .      # writes data/healthver and data_train/target/healthver
```

### Pretrained MultiVerS weights

```bash
cd multivers
python script/get_checkpoint.py longformer_large_science
python script/get_checkpoint.py fever_sci
```

The released `fever_sci.ckpt` does not load with PyTorch Lightning 1.9.5 as-is; it was
re-saved as a weights-only file `checkpoints/fever_sci_weights.ckpt` (identical state dict).

## Training

```bash
# Baseline: from the plain encoder (empty STARTING_CHECKPOINT)
STARTING_CHECKPOINT= RESULT_DIR=results EXPERIMENT_NAME=healthver_science_init bash scripts/train_healthver.sh

# Improved: from fever_sci (default in the script)
RESULT_DIR=results EXPERIMENT_NAME=healthver_from_fever_sci bash scripts/train_healthver.sh

# Multi-dataset, checkpoint selection on HealthVer only
DATASETS=healthver,covidfact VALID_DATASETS=healthver \
RESULT_DIR=results EXPERIMENT_NAME=healthver_covidfact_from_fever_sci bash scripts/train_healthver.sh
```

Settings: 20 epochs, learning rate 1e-5 (linear schedule, 10% warm-up), effective batch 8,
16-bit precision, seed 76, best epoch chosen on validation sentence-label F1.
About 25 minutes per epoch on one RTX 4080 SUPER.

## Full pipeline

```bash
python scripts/prepare_covid_questions.py
python scripts/run_final_medgemma_eval.py \
  --questions data/medgemma_eval/covid_questions.jsonl \
  --output_dir results/medgemma_eval_covid_fever_sci \
  --checkpoint multivers/results/healthver_from_fever_sci/checkpoint/epoch=15-step=10592.ckpt
```

## Changes to MultiVerS

Vendored from [dwadden/multivers](https://github.com/dwadden/multivers) at commit
`a6ce033`. Changes:

- `model.py`: `torch.optim.AdamW` (same settings) instead of the removed `transformers.AdamW`;
  optional label-aware evidence head (`--label_aware_rationale`).
- `data_train.py`: `--valid_datasets` to select checkpoints on a subset of datasets;
  `MULTIVERS_DATA_ROOT` environment variable for the training-data folder.
- `train.py`, `predict.py`, `metrics.py`: compatibility with newer PyTorch / Lightning.

## Acknowledgements

- MultiVerS — Wadden et al., Allen Institute for AI (MIT License, see `multivers/LICENSE`).
- HealthVer — Sarrouti et al., used via `dwadden/healthver_entailment`.
- MedGemma — Google, subject to the Health AI Developer Foundations terms.
